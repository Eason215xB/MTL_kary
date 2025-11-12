# mtl_losses.py
# [!! 最终修复: "Plan D" 损失 !!]
# 1. 移除 Matcher, Mask2FormerLoss (它们是为 "盲" 头准备的)
# 2. 添加 FCN 损失 (CE + Dice)
# 3. 保留 SimCCLoss (它是正确的)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Tuple

# ==================================
# 1. 辅助函数 (Dice Loss)
# ==================================

def dice_loss_fcn(
    inputs: Tensor,  # (B, C, H, W) - Logits
    targets: Tensor, # (B, H, W) - Long
    num_classes: int
) -> Tensor:
    """计算 FCN 语义分割的 Dice Loss"""
    # Logits -> Probs
    inputs_probs = F.softmax(inputs, dim=1)
    
    # (B, H, W) -> (B, C, H, W) - One-hot
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # (B, C, H*W)
    inputs_flat = inputs_probs.flatten(2)
    targets_flat = targets_one_hot.flatten(2)
    
    numerator = 2 * (inputs_flat * targets_flat).sum(2) # (B, C)
    denominator = inputs_flat.sum(2) + targets_flat.sum(2) # (B, C)
    
    loss_per_class = 1 - (numerator + 1e-4) / (denominator + 1e-4)
    
    return loss_per_class.mean()


# ==================================
# 2. RTMCC (SimCC) 损失 (不变)
# ==================================

class SimCCLoss(nn.Module):
    def __init__(
        self, 
        use_target_weight: bool = True, 
        reduction: str = 'mean',
        beta: float = 1.0,
        label_softmax: bool = False
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.reduction = reduction
        self.beta = beta
        self.label_softmax = label_softmax
        self.kld_loss = nn.KLDivLoss(reduction='none')

    def forward(
        self,
        pred_x: Tensor,  # (N_total, W_simcc)
        pred_y: Tensor,  # (N_total, H_simcc)
        target_x: Tensor,# (N_total, W_simcc)
        target_y: Tensor,# (N_total, H_simcc)
        keypoint_weights: Tensor # (N_total,)
    ) -> Tensor:
        
        if self.label_softmax:
            target_x = F.softmax(target_x, dim=1)
            target_y = F.softmax(target_y, dim=1)

        log_pred_x = F.log_softmax(pred_x * self.beta, dim=1)
        log_pred_y = F.log_softmax(pred_y * self.beta, dim=1)
        
        loss_x = self.kld_loss(log_pred_x, target_x).sum(dim=1)
        loss_y = self.kld_loss(log_pred_y, target_y).sum(dim=1)
        
        loss = (loss_x + loss_y) * keypoint_weights
        
        if self.reduction == 'mean':
            if keypoint_weights.sum() > 0:
                return loss.sum() / keypoint_weights.sum()
            else:
                return loss.sum() * 0
        else:
            return loss.sum()

# ==================================
# 3. 总的多任务损失 (Plan D)
# ==================================

class MTLAllLosses(nn.Module):
    """[!! 最终修复: "Plan D" 损失 !!]"""
    def __init__(
        self,
        num_classes: int, # Seg 类别数 (e.g., 26)
        weights: Dict[str, float],
        pose_beta: float = 1.0,
        pose_label_softmax: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 姿态损失 (不变)
        self.pose_loss = SimCCLoss(
            use_target_weight=True,
            beta=pose_beta,
            label_softmax=pose_label_softmax
        )
        self.weight_pose = weights.get("w_pose", 0.02)

        # 2. 分割损失 (FCN)
        # 交叉熵损失 (忽略索引 0，假设 0 是背景)
        # [!! 关键 !!] 假设你的 GT 标签 0 是背景, 1-25 是染色体。
        # 如果你的 GT 标签 1-26，没有 0，你需要调整这里。
        # 我们假设 0 是背景。
        self.seg_ce_loss = nn.CrossEntropyLoss(ignore_index=0) 
        self.weight_seg_ce = weights.get("w_seg_ce", 1.0)
        self.weight_seg_dice = weights.get("w_seg_dice", 3.0)

    def get_seg_loss(
        self, 
        seg_logits: Tensor, # (B, C, H, W)
        targets_list: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        
        B, C, H, W = seg_logits.shape
        
        # (B, H, W)
        gt_sem_mask = torch.zeros((B, H, W), dtype=torch.long, device=seg_logits.device)
        
        # [!! 关键 !!]
        # 从实例掩码 (N_gt, H, W) 创建语义掩码 (H, W)
        # 假设 0 是背景
        for b in range(B):
            gt_labels_b = targets_list[b]['labels'] # (N_gt,)
            gt_masks_b = targets_list[b]['masks']   # (N_gt, H, W)
            
            for i in range(len(gt_labels_b)):
                label_idx = gt_labels_b[i] # 你的 config 是 1-25?
                if 0 < label_idx < self.num_classes:
                    gt_sem_mask[b][gt_masks_b[i] > 0] = label_idx
        
        # 1. CE Loss
        loss_ce = self.seg_ce_loss(seg_logits, gt_sem_mask)
        
        # 2. Dice Loss
        loss_dice = dice_loss_fcn(seg_logits, gt_sem_mask, self.num_classes)
        
        return {
            "seg_loss_ce": loss_ce * self.weight_seg_ce,
            "seg_loss_dice": loss_dice * self.weight_seg_dice
        }


    def get_pose_loss(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """(与 "Plan C" 相同)"""
        src_pose_x = outputs["pose_pred_x"]
        src_pose_y = outputs["pose_pred_y"]
        
        if src_pose_x.numel() == 0:
            return {"pose_loss": src_pose_x.sum() * 0.0}

        gt_pose_x = torch.cat([t["pose_targets_x"] for t in targets], dim=0)
        gt_pose_y = torch.cat([t["pose_targets_y"] for t in targets], dim=0)
        gt_pose_w = torch.cat([t["pose_weights"] for t in targets], dim=0)
        
        B, K, Ws = src_pose_x.shape
        _, _, Hs = src_pose_y.shape
        
        if gt_pose_x.shape[0] != B:
            if gt_pose_x.numel() == 0:
                return {"pose_loss": src_pose_x.sum() * 0.0}
            return {"pose_loss": src_pose_x.sum() * 0.0}

        keypoint_weights = gt_pose_w.reshape(B * K)
        
        loss = self.pose_loss(
            src_pose_x.reshape(B * K, Ws),
            src_pose_y.reshape(B * K, Hs),
            gt_pose_x.reshape(B * K, Ws),
            gt_pose_y.reshape(B * K, Hs),
            keypoint_weights
        )
        
        return {"pose_loss": loss * self.weight_pose}


    def forward(self, outputs: Dict[str, Tensor], targets_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        
        # [!! 修复 !!] 移除 Matcher
        
        # 1. 计算分割损失 (FCN)
        seg_losses = self.get_seg_loss(outputs["seg_logits"], targets_list)
        
        # 2. 计算姿态损失 (RTMCC)
        pose_losses = self.get_pose_loss(outputs, targets_list)
        
        # 3. 组合
        all_losses = {}
        all_losses.update(seg_losses)
        all_losses.update(pose_losses)
        
        all_losses["total_loss"] = sum(all_losses.values())
        
        return all_losses
