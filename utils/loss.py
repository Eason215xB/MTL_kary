# mtl_losses.py
# [!! V2 - 修正了 Seg Class Loss 和 SimCC Loss !!]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Tuple

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("="*50)
    print("错误: 未找到 'scipy' 库。")
    print("Mask2Former 损失需要 'scipy' 来进行匈牙利匹配。")
    print("请运行: pip install scipy")
    print("="*50)
    exit(1)


# ==================================
# 1. 辅助函数 (Focal & Dice Loss)
# (这部分不变, 因为它们用于 Mask Loss, 是正确的)
# ==================================

def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    """
    Sigmoid Focal Loss 的 PyTorch 实现。
    (用于 Mask Loss)
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def dice_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
):
    """
    Dice Loss，用于计算 Mask 损失。
    """
    inputs = inputs.sigmoid().flatten(1) # (N, H*W)
    targets = targets.flatten(1)         # (N, H*W)
    
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1e-4) / (denominator + 1e-4)
    return loss.sum() / num_masks


# ==================================
# 2. RTMCC (SimCC) 损失 (修正版)
# ==================================

class SimCCLoss(nn.Module):
    """
    SimCC 损失 (使用 KLD)。
    [!! V2 修正: 增加了 beta 和 label_softmax !!]
    """
    def __init__(
        self, 
        use_target_weight: bool = True, 
        reduction: str = 'mean',
        beta: float = 1.0,               # [!! 新增 !!] Logit 缩放因子
        label_softmax: bool = False      # [!! 新增 !!] 是否对 Target (高斯)
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
        
        # [!! 修正 !!]
        if self.label_softmax:
            target_x = F.softmax(target_x, dim=1)
            target_y = F.softmax(target_y, dim=1)

        # [!! 修正 !!]
        # (N, Ws)
        log_pred_x = F.log_softmax(pred_x * self.beta, dim=1)
        # (N, Hs)
        log_pred_y = F.log_softmax(pred_y * self.beta, dim=1)
        
        loss_x = self.kld_loss(log_pred_x, target_x).sum(dim=1)
        loss_y = self.kld_loss(log_pred_y, target_y).sum(dim=1)
        
        # 应用权重 (只计算可见关键点的损失)
        loss = (loss_x + loss_y) * keypoint_weights
        
        # 求平均
        if self.reduction == 'mean':
            if keypoint_weights.sum() > 0:
                return loss.sum() / keypoint_weights.sum()
            else:
                return loss.sum() * 0
        else:
            return loss.sum()

# ==================================
# 3. Mask2Former 匈牙利匹配器 (不变)
# ==================================

class HungarianMatcher(nn.Module):
    """
    Mask2Former 的匹配器。
    (这个模块保持不变, 它的逻辑是正确的)
    """
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        
        B, Nq = outputs["class_logits"].shape[:2]

        # 1. 准备模型输出 (B, Nq, ...)
        # [!! 修正 !!] 匹配成本使用 CrossEntropy (softmax)
        out_prob = outputs["class_logits"].softmax(-1) # (B, Nq, C+1)
        out_mask = outputs["mask_logits"]              # (B, Nq, H, W)

        indices = []
        
        for b in range(B):
            tgt_ids = targets[b]["labels"]     # (N_gt,)
            tgt_mask = targets[b]["masks"]     # (N_gt, H, W)
            N_gt = len(tgt_ids)

            if N_gt == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            # 3. 计算成本矩阵 (Nq, N_gt)
            
            # [!! 修正 !!]
            # cost_class: (Nq, C+1) -> (Nq, N_gt)
            # 使用 -P(class) 作为成本 (来自 out_prob)
            cost_class = -out_prob[b][:, tgt_ids] * self.cost_class

            out_mask_b = out_mask[b]
            tgt_mask_b = tgt_mask
            
            H, W = out_mask_b.shape[1:]
            total_pixels = H * W
            num_points_sample = min(self.num_points, total_pixels)
            indices_sample = torch.randperm(total_pixels, device=out_mask_b.device)[:num_points_sample]

            out_mask_sample = out_mask_b.flatten(1)[:, indices_sample]
            tgt_mask_sample = tgt_mask_b.flatten(1)[:, indices_sample].float()
            
            out_mask_expanded = out_mask_sample.unsqueeze(1).expand(-1, N_gt, -1)
            tgt_mask_expanded = tgt_mask_sample.unsqueeze(0).expand(Nq, -1, -1)

            # Focal Loss (作为成本)
            cost_mask = sigmoid_focal_loss(
                out_mask_expanded,
                tgt_mask_expanded,
                reduction="none"
            )
            cost_mask = cost_mask.mean(dim=-1) * self.cost_mask

            # Dice Loss (作为成本)
            out_dice = out_mask_sample.sigmoid().unsqueeze(1)
            tgt_dice = tgt_mask_sample.unsqueeze(0)
            numerator = 2 * (out_dice * tgt_dice).sum(dim=-1)
            denominator = out_dice.sum(dim=-1) + tgt_dice.sum(dim=-1)
            cost_dice = (1 - (numerator + 1e-4) / (denominator + 1e-4))
            cost_dice = cost_dice * self.cost_dice

            # --- 最终成本 ---
            C = cost_class + cost_mask + cost_dice
            C = C.cpu() 

            row_ind, col_ind = linear_sum_assignment(C)
            
            row_ind = torch.from_numpy(row_ind).to(out_prob.device)
            col_ind = torch.from_numpy(col_ind).to(out_prob.device)
            indices.append((row_ind, col_ind))

        return indices


# ==================================
# 4. Mask2Former 损失 (修正版)
# ==================================

class Mask2FormerLoss(nn.Module):
    """
    [!! V2 修正 !!]
    """
    def __init__(
        self,
        num_classes: int,
        weight_class: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.eos_coef = eos_coef

        # [!! 修正 !!]
        # 创建用于 F.cross_entropy 的 class_weight
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef

    def _get_src_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _get_tgt_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return (batch_idx, tgt_idx)

    def get_loss_class(self, class_logits: Tensor, targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]]) -> Tensor:
        """
        [!! V2 修正: 使用 CrossEntropyLoss !!]
        """
        B, Nq, C_plus_1 = class_logits.shape
        
        # (N_total_matched,)
        idx = self._get_src_permutation_idx(indices)
        
        # (N_total_matched,)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # (B, Nq)
        target_classes = torch.full(
            (B, Nq), self.num_classes, # 默认设为 'no object'
            dtype=torch.int64, device=class_logits.device
        )
        # 在匹配的位置填入 GT 类别
        target_classes[idx] = target_classes_o

        # [!! 修正 !!]
        # (B, Nq, C+1) -> (B*Nq, C+1)
        class_logits = class_logits.flatten(0, 1)
        # (B, Nq) -> (B*Nq)
        target_classes = target_classes.flatten()
        
        # 计算 CrossEntropy Loss
        loss = F.cross_entropy(
            class_logits, 
            target_classes, 
            weight=self.empty_weight.to(class_logits.device),
            reduction="mean"
        )
        
        return loss

    def get_loss_mask(self, mask_logits: Tensor, targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        计算掩码损失 (Focal + Dice)。
        (这部分保持不变, 它是正确的)
        """
        idx = self._get_src_permutation_idx(indices)
        src_masks = mask_logits[idx]
        
        tgt_idx = self._get_tgt_permutation_idx(indices)
        tgt_masks_all = torch.cat([t["masks"] for t in targets], dim=0).float()
        tgt_masks = tgt_masks_all[tgt_idx[1]]

        N_masks = src_masks.shape[0]
        if N_masks == 0:
            return src_masks.sum() * 0, src_masks.sum() * 0

        loss_focal = sigmoid_focal_loss(
            src_masks, tgt_masks,
            alpha=0.25, gamma=2.0, reduction="mean"
        )
        loss_dice = dice_loss(
            src_masks, tgt_masks, num_masks=N_masks
        )
        return loss_focal, loss_dice

    def get_losses(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        """
        计算 Mask2Former 的总损失 (给定匹配索引)。
        """
        # 1. 计算类别损失 (已修正)
        loss_class = self.get_loss_class(outputs["class_logits"], targets, indices)
        
        # 2. 计算掩码损失 (不变)
        loss_mask, loss_dice = self.get_loss_mask(outputs["mask_logits"], targets, indices)
        
        return {
            "seg_loss_class": loss_class * self.weight_class,
            "seg_loss_mask": loss_mask * self.weight_mask,
            "seg_loss_dice": loss_dice * self.weight_dice,
        }

# ==================================
# 5. 总的多任务损失 (修正版)
# ==================================

class MTLAllLosses(nn.Module):
    """
    [!! V2 修正: 传递 SimCC 参数 !!]
    """
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weights: Dict[str, float],
        pose_beta: float = 1.0,           # [!! 新增 !!]
        pose_label_softmax: bool = False  # [!! 新增 !!]
    ):
        super().__init__()
        self.matcher = matcher 
        self.seg_loss = Mask2FormerLoss(
            num_classes=num_classes,
            weight_class=weights.get("w_seg_cls", 2.0),
            weight_mask=weights.get("w_seg_mask", 5.0),
            weight_dice=weights.get("w_seg_dice", 5.0)
        )
        # [!! 修正 !!]
        self.pose_loss = SimCCLoss(
            use_target_weight=True,
            beta=pose_beta,
            label_softmax=pose_label_softmax
        )
        self.weight_pose = weights.get("w_pose", 1.0)

    def get_pose_loss(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        """
        使用匹配的索引计算姿态损失。
        (这部分逻辑不变)
        """
        src_idx = self.seg_loss._get_src_permutation_idx(indices) 
        tgt_idx = self.seg_loss._get_tgt_permutation_idx(indices) 

        src_pose_x = outputs["pose_pred_x"][src_idx]
        src_pose_y = outputs["pose_pred_y"][src_idx]
        
        if src_pose_x.numel() == 0:
            # (修正: 保持计算图连接)
            return {"pose_loss": (outputs["pose_pred_x"].sum() + outputs["pose_pred_y"].sum()) * 0.0}

        gt_pose_x = torch.cat([t["pose_targets_x"] for t in targets], dim=0)
        gt_pose_y = torch.cat([t["pose_targets_y"] for t in targets], dim=0)
        gt_pose_w = torch.cat([t["pose_weights"] for t in targets], dim=0)
        
        tgt_pose_x = gt_pose_x[tgt_idx[1]]
        tgt_pose_y = gt_pose_y[tgt_idx[1]]
        tgt_pose_w = gt_pose_w[tgt_idx[1]]

        N_matched, K, Ws = src_pose_x.shape
        _, _, Hs = src_pose_y.shape
        
        # (N_matched, K) -> (N_matched * K)
        keypoint_weights = tgt_pose_w.reshape(N_matched * K)
        
        # [!! 修正 !!] SimCCLoss 期望 (N_total, C)
        loss = self.pose_loss(
            src_pose_x.reshape(N_matched * K, Ws),
            src_pose_y.reshape(N_matched * K, Hs),
            tgt_pose_x.reshape(N_matched * K, Ws),
            tgt_pose_y.reshape(N_matched * K, Hs),
            keypoint_weights # 传入 (N_total,)
        )
        
        return {"pose_loss": loss * self.weight_pose}


    def forward(self, outputs: Dict[str, Tensor], targets_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        
        # 1. 匈牙利匹配
        matcher_outputs = {
            "class_logits": outputs["seg_class_logits"],
            "mask_logits": outputs["seg_mask_logits"]
        }
        indices = self.matcher(matcher_outputs, targets_list)
        
        # 2. 计算分割损失 (已修正)
        seg_losses = self.seg_loss.get_losses(matcher_outputs, targets_list, indices)
        
        # 3. 计算姿态损失 (已修正)
        pose_losses = self.get_pose_loss(outputs, targets_list, indices)
        
        # 4. 组合
        all_losses = {}
        all_losses.update(seg_losses)
        all_losses.update(pose_losses)
        
        all_losses["total_loss"] = sum(all_losses.values())
        
        return all_losses

# ==================================
# 6. 单元测试 (修正版)
# ==================================
if __name__ == "__main__":
    print("开始 mtl_losses (V2 修正版) 单元测试...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, Nq, C, H, W = 2, 100, 23, 64, 64
    K, W_s, H_s = 5, 128, 128
    
    # 1. 模拟模型输出
    outputs = {
        "seg_class_logits": torch.randn(B, Nq, C + 1, device=device),
        "seg_mask_logits": torch.randn(B, Nq, H, W, device=device),
        "pose_pred_x": torch.randn(B, Nq, K, W_s, device=device),
        "pose_pred_y": torch.randn(B, Nq, K, H_s, device=device),
    }

    # 2. 模拟 GT
    N_gt_1 = 2
    N_gt_2 = 3
    
    targets_list = [
        {
            "labels": torch.randint(0, C, (N_gt_1,), device=device),
            "masks": torch.rand(N_gt_1, H, W, device=device) > 0.5,
            "pose_targets_x": torch.rand(N_gt_1, K, W_s, device=device),
            "pose_targets_y": torch.rand(N_gt_1, K, H_s, device=device),
            "pose_weights": torch.ones(N_gt_1, K, device=device),
        },
        {
            "labels": torch.randint(0, C, (N_gt_2,), device=device),
            "masks": torch.rand(N_gt_2, H, W, device=device) > 0.5,
            "pose_targets_x": torch.rand(N_gt_2, K, W_s, device=device),
            "pose_targets_y": torch.rand(N_gt_2, K, H_s, device=device),
            "pose_weights": torch.ones(N_gt_2, K, device=device),
        },
    ]

    # 3. 初始化损失
    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
    loss_weights = {"w_seg_cls": 2.0, "w_seg_mask": 5.0, "w_seg_dice": 5.0, "w_pose": 1.0}
    
    mtl_loss_fn = MTLAllLosses(
        num_classes=C,
        matcher=matcher,
        weights=loss_weights,
        pose_beta=15.0,               # [!! 测试 !!]
        pose_label_softmax=True       # [!! 测试 !!]
    ).to(device)
    
    # 4. 计算损失
    losses = mtl_loss_fn(outputs, targets_list)
    
    print("计算得到的损失:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
        
    assert "total_loss" in losses
    assert "pose_loss" in losses
    assert "seg_loss_class" in losses
    assert losses["total_loss"] > 0
    
    print("\n[!!] V2 损失计算成功 [!!]")
    print("所有测试通过！")