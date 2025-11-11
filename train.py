# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
import time
import json
import argparse
from typing import Dict, List, Tuple
import cv2
import numpy as np

# ----------------------------
# 1. 导入你的所有自定义模块
# ----------------------------
from model.MTL_model import ChromosomeMTLModel
from MTL_datasets import CocoMtlDataset, mtl_collate_fn
from utils.loss import MTLAllLosses, HungarianMatcher


class MeanMetric:
    """在 DDP 中安全计算平均值的类"""
    def __init__(self, device: torch.device):
        self.total = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.count = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.device = device
    
    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n
        
    def reduce_from_all_processes(self):
        """在所有 DDP 进程中同步 total 和 count"""
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return (self.total / self.count).item()

class ConfusionMatrix:
    """
    用于计算 mIoU 和 mAcc 的混淆矩阵。
    """
    def __init__(self, num_classes: int, device: torch.device):
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.num_classes = num_classes
        self.device = device

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        mask = (gt >= 0) & (gt < self.num_classes)
        hist = torch.bincount(
            self.num_classes * gt[mask] + pred[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.matrix += hist
    
    def reduce_from_all_processes(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.all_reduce(self.matrix, op=dist.ReduceOp.SUM)

    def compute(self) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        hist = self.matrix.float()
        
        acc_cls = torch.diag(hist) / (hist.sum(dim=1) + 1e-6)
        mAcc = acc_cls[~torch.isnan(acc_cls)].mean().item()
        
        iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + 1e-6)
        mIoU = iou[~torch.isnan(iou)].mean().item()
        
        return mIoU, mAcc, iou, acc_cls

# ----------------------------
# 3. 姿态指标辅助函数 (不变)
# ----------------------------

def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return (batch_idx, src_idx)

def _get_tgt_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return (batch_idx, tgt_idx)

def decode_simcc_batch(
    pred_x_logits: torch.Tensor,  # (N, K, Ws)
    pred_y_logits: torch.Tensor,  # (N, K, Hs)
    simcc_ratio: float
) -> torch.Tensor:
    device = pred_x_logits.device
    N, K, Ws = pred_x_logits.shape
    _, _, Hs = pred_y_logits.shape
    
    pred_x_probs = F.softmax(pred_x_logits, dim=-1)
    pred_y_probs = F.softmax(pred_y_logits, dim=-1)
    
    grid_x = torch.arange(Ws, device=device, dtype=torch.float32).view(1, 1, Ws)
    grid_y = torch.arange(Hs, device=device, dtype=torch.float32).view(1, 1, Hs)
    
    coords_x = (pred_x_probs * grid_x).sum(-1) / simcc_ratio
    coords_y = (pred_y_probs * grid_y).sum(-1) / simcc_ratio
    
    return torch.stack([coords_x, coords_y], dim=-1)

def compute_pck_auc_normalized(
    pred_kpts: torch.Tensor, # (N_matched, K, 2)
    gt_kpts: torch.Tensor,   # (N_matched, K, 2)
    gt_vis: torch.Tensor,    # (N_matched, K)
    gt_bboxes: torch.Tensor, # (N_matched, 4) [x1, y1, x_max, y_max]
    pck_thresholds: torch.Tensor # (T,)
) -> Tuple[torch.Tensor, float]:
    """
    计算 BBox 归一化的 PCK 和 AUC。
    """
    # 1. 计算归一化尺度
    gt_bboxes_wh = gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2] # [w, h]
    norm_scale = torch.max(gt_bboxes_wh, dim=1)[0].unsqueeze(1) # max(w, h)
    
    # 2. 计算像素距离
    distances = torch.sqrt(((pred_kpts - gt_kpts)**2).sum(dim=-1))
    
    # 3. 计算归一化距离
    normalized_distances = distances / (norm_scale + 1e-6)

    # 4. 应用可见性
    visible_mask = gt_vis > 0
    if visible_mask.sum() == 0:
        return torch.zeros(len(pck_thresholds), device=pred_kpts.device), 0.0
    
    # 只计算可见关键点的距离
    visible_distances = normalized_distances[visible_mask]

    # 5. PCK 计算 - 使用可见关键点的距离
    pck_matrix = (visible_distances.unsqueeze(0) <= pck_thresholds.unsqueeze(1))
    pck_values = pck_matrix.float().mean(dim=-1)
    
    # 6. AUC (在 [0, 0.5] 范围内)
    auc = pck_values.mean().item()
    
    return pck_values, auc


# ----------------------------
# 4. 训练和验证函数 (修正)
# ----------------------------

def train_one_epoch(
    model: DDP,
    loss_fn: MTLAllLosses,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    rank: int,
    grad_clip_norm: float
) -> float:
    model.train()
    loss_fn.train()
    data_loader.sampler.set_epoch(epoch)
    
    total_loss = 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", disable=(rank != 0))
    
    for images, targets_list in pbar:
        images = images.to(device, non_blocking=True)
        targets_list_device = []
        for t in targets_list:
            targets_list_device.append({k: v.to(device, non_blocking=True) for k, v in t.items()})

        outputs = model(images)
        losses = loss_fn(outputs, targets_list_device)
        total_loss_batch = losses['total_loss']
        
        if not torch.isfinite(total_loss_batch):
            print(f"\n[!! 错误 - Rank {rank} !!] 损失为 NaN 或 Inf。")
            continue

        optimizer.zero_grad()
        total_loss_batch.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        if rank == 0:
            pbar.set_postfix(
                loss=f"{total_loss_batch.item():.4f}",
                seg_cls=f"{losses['seg_loss_class'].item():.4f}",
                seg_mask=f"{losses['seg_loss_mask'].item():.4f}",
                seg_dice=f"{losses['seg_loss_dice'].item():.4f}",
                pose=f"{losses['pose_loss'].item():.4f}"
            )
            
    return total_loss / len(data_loader)


@torch.no_grad()
def validate(
    model: DDP,
    loss_fn: MTLAllLosses,
    data_loader: DataLoader,
    device: torch.device,
    rank: int,
    num_classes: int,
    config: Dict
) -> Dict[str, float]:
    """[!! V7: 修复了 NameError !!]"""
    model.eval()
    loss_fn.eval()
    
    val_loss = MeanMetric(device=device)
    conf_matrix = ConfusionMatrix(num_classes=num_classes, device=device)
    
    pck_thresholds = torch.tensor([0.05, 0.1, 0.2, 0.5], device=device)
    pck_metrics = [MeanMetric(device=device) for _ in pck_thresholds]
    auc_metric = MeanMetric(device=device)
    
    simcc_ratio = config['MODEL_PARAMS']['POSE_SIMCC_RATIO']
    
    pbar = tqdm(data_loader, desc="[Validate]", disable=(rank != 0))
    
    for images, targets_list in pbar:
        
        images = images.to(device, non_blocking=True)
        targets_list_device = []
        for t in targets_list:
            targets_list_device.append({k: v.to(device, non_blocking=True) for k, v in t.items()})

        # 1. 前向传播和损失
        outputs = model(images)
        losses = loss_fn(outputs, targets_list_device)
        val_loss.update(losses['total_loss'].item(), n=images.shape[0])
        
        # 2. 运行匹配器以获取索引
        matcher_outputs = {
            "class_logits": outputs["seg_class_logits"],
            "mask_logits": outputs["seg_mask_logits"]
        }
        indices = loss_fn.matcher(matcher_outputs, targets_list_device)
        
        src_idx = _get_src_permutation_idx(indices)
        tgt_idx = _get_tgt_permutation_idx(indices)
        
        num_matched = src_idx[0].shape[0]
        
        # [!! 调试信息 !!]
        # if rank == 0 and torch.rand(1).item() < 0.05:  # 5%概率打印调试信息
        #     total_instances = sum(len(t['labels']) for t in targets_list_device)
        #     print(f"[调试-匹配] 批次中总实例数: {total_instances}, 匹配实例数: {num_matched}")
        #     print(f"[调试-匹配] src_idx: {src_idx[1][:10] if len(src_idx[1]) > 10 else src_idx[1]}")  # 只显示前10个
        #     print(f"[调试-匹配] tgt_idx: {tgt_idx[1][:10] if len(tgt_idx[1]) > 10 else tgt_idx[1]}")  # 只显示前10个
            
        # 3. 计算分割指标 (mIoU, mAcc)
        B, Nq, H, W = outputs['seg_mask_logits'].shape
        C = num_classes
        class_probs = F.softmax(outputs["seg_class_logits"][..., :-1], dim=-1)
        sem_mask_probs = torch.einsum('bqc,bqhw->bchw', class_probs, outputs['seg_mask_logits'].sigmoid())
        sem_pred = sem_mask_probs.argmax(dim=1)

        sem_gt = torch.zeros((B, H, W), dtype=torch.long, device=device)
        for b_idx in range(B):
            gt_labels_b = targets_list_device[b_idx]['labels']
            gt_masks_b = targets_list_device[b_idx]['masks']
            for i in range(len(gt_labels_b)):
                label_idx = gt_labels_b[i]
                if 0 <= label_idx < num_classes:
                    sem_gt[b_idx][gt_masks_b[i] > 0] = label_idx
        
        conf_matrix.update(sem_pred, sem_gt)

        # 4. 计算姿态指标 (PCK, AUC)
        B, Nq, K, Ws = outputs['pose_pred_x'].shape
        _, _, _, Hs = outputs['pose_pred_y'].shape
        
        pred_kpts_all = decode_simcc_batch(
            outputs['pose_pred_x'].reshape(B*Nq, K, Ws), 
            outputs['pose_pred_y'].reshape(B*Nq, K, Hs), 
            simcc_ratio
        )
        pred_kpts_all = pred_kpts_all.reshape(B, Nq, K, 2)
        

        if num_matched > 0:
            src_kpts = pred_kpts_all[src_idx] 
            
            gt_bboxes_all = torch.cat([t["bboxes"] for t in targets_list_device], dim=0)
            gt_pose_x_all = torch.cat([t["pose_targets_x"] for t in targets_list_device], dim=0)
            gt_pose_y_all = torch.cat([t["pose_targets_y"] for t in targets_list_device], dim=0)
            gt_pose_w_all = torch.cat([t["pose_weights"] for t in targets_list_device], dim=0)
            
            if rank == 0 and torch.rand(1).item() < 0.02:  # 2%概率打印调试信息
                print(f"[调试-网络输出] pred_kpts_all.shape: {pred_kpts_all.shape}")
                # 检查网络原始输出
                pose_pred_x_sample = outputs['pose_pred_x'][0, :3, 0, :]  # 第一个样本，前3个查询，第一个关键点
                pose_pred_y_sample = outputs['pose_pred_y'][0, :3, 0, :]  
                print(f"[调试-网络输出] pose_pred_x前3个查询的logits范围: [{pose_pred_x_sample.min():.3f}, {pose_pred_x_sample.max():.3f}]")
                print(f"[调试-网络输出] pose_pred_y前3个查询的logits范围: [{pose_pred_y_sample.min():.3f}, {pose_pred_y_sample.max():.3f}]")
                # 检查softmax后的分布
                x_probs = F.softmax(pose_pred_x_sample, dim=-1)
                y_probs = F.softmax(pose_pred_y_sample, dim=-1)
                x_entropy = -(x_probs * torch.log(x_probs + 1e-8)).sum(dim=-1).mean()
                y_entropy = -(y_probs * torch.log(y_probs + 1e-8)).sum(dim=-1).mean()
                print(f"[调试-网络输出] X/Y概率分布熵: {x_entropy:.3f}, {y_entropy:.3f} (熵越低越集中)")
                print("-" * 40)
            
            tgt_kpts_x = gt_pose_x_all[tgt_idx[1]]
            tgt_kpts_y = gt_pose_y_all[tgt_idx[1]]
            tgt_kpts_vis = gt_pose_w_all[tgt_idx[1]] 
            tgt_bboxes = gt_bboxes_all[tgt_idx[1]] 
            
            # [!! 修复 !!] 直接使用原始GT坐标，不要解码SimCC目标
            gt_kpts_raw = torch.cat([t["pose_kpts_aug"] for t in targets_list_device], dim=0)
            gt_kpts = gt_kpts_raw[tgt_idx[1]][:, :, :2]  # 只取x,y坐标，去掉visibility 
            
            pck_values, auc_value = compute_pck_auc_normalized(
                src_kpts, gt_kpts, tgt_kpts_vis, tgt_bboxes, pck_thresholds
            )
            
            num_visible_kpts = tgt_kpts_vis.sum().item()
            

            # if rank == 0 and torch.rand(1).item() < 0.05:  # 5%概率打印调试信息
            #     print(f"[调试-可见性] tgt_kpts_vis.shape: {tgt_kpts_vis.shape}")
            #     print(f"[调试-可见性] num_visible_kpts: {num_visible_kpts}")
            #     print(f"[调试-可见性] 匹配实例数: {num_matched}, 每实例关键点数: {tgt_kpts_vis.shape[1] if len(tgt_kpts_vis.shape) > 1 else 'N/A'}")
            #     print(f"[调试-可见性] tgt_kpts_vis前几行: {tgt_kpts_vis[:3] if len(tgt_kpts_vis) > 0 else 'empty'}")
            
            if num_visible_kpts > 0:
                # [!! 修复后的调试信息 !!]
                if rank == 0 and torch.rand(1).item() < 0.05:  # 5%概率打印调试信息
                    distances = torch.sqrt(((src_kpts - gt_kpts)**2).sum(dim=-1))
                    # 计算bbox尺寸用于调试
                    gt_bboxes_wh = tgt_bboxes[:, 2:4] - tgt_bboxes[:, 0:2]
                    norm_scale = torch.max(gt_bboxes_wh, dim=1)[0].unsqueeze(1)
                    normalized_distances = distances / (norm_scale + 1e-6)
                    
                    print(f"[调试-修复后] 匹配实例数: {num_matched}, 可见关键点数: {num_visible_kpts}")
                    print(f"[调试-修复后] 预测坐标范围: x=[{src_kpts[:,:,0].min():.1f}, {src_kpts[:,:,0].max():.1f}], y=[{src_kpts[:,:,1].min():.1f}, {src_kpts[:,:,1].max():.1f}]")
                    print(f"[调试-修复后] GT坐标范围: x=[{gt_kpts[:,:,0].min():.1f}, {gt_kpts[:,:,0].max():.1f}], y=[{gt_kpts[:,:,1].min():.1f}, {gt_kpts[:,:,1].max():.1f}]")
                    print(f"[调试-修复后] 像素距离范围: [{distances.min():.2f}, {distances.max():.2f}]")
                    print(f"[调试-修复后] Bbox尺寸: {gt_bboxes_wh.max(dim=1)[0]}")
                    print(f"[调试-修复后] 归一化距离范围: [{normalized_distances.min():.4f}, {normalized_distances.max():.4f}]")
                    print(f"[调试-修复后] PCK@0.5: {pck_values[3].item():.4f}")
                    print("="*60)
                
                for i in range(len(pck_thresholds)):
                    pck_metrics[i].update(pck_values[i].item(), n=1)
                auc_metric.update(auc_value, n=1)

    # 5. 聚合所有 DDP 进程的指标
    val_loss.reduce_from_all_processes()
    conf_matrix.reduce_from_all_processes()
    for pck_metric in pck_metrics:
        pck_metric.reduce_from_all_processes()
    auc_metric.reduce_from_all_processes()

    # 6. 仅在 Rank 0 上计算最终结果
    if rank == 0:
        mIoU, mAcc, _, _ = conf_matrix.compute()
        
        final_metrics = {
            "loss": val_loss.avg,
            "mIoU": mIoU,
            "mAcc": mAcc,
            "AUC": auc_metric.avg,
        }
        final_metrics["PCK@0.05"] = pck_metrics[0].avg
        final_metrics["PCK@0.1"] = pck_metrics[1].avg
        final_metrics["PCK@0.2"] = pck_metrics[2].avg
        final_metrics["PCK@0.5"] = pck_metrics[3].avg
        
        return final_metrics
    else:
        return None

# ----------------------------
# 3. DDP 设置
# ----------------------------

def setup_ddp(rank: int, world_size: int, port: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()


# ----------------------------
# 5. 主工作函数 (DDP) (修正)
# ----------------------------

def main_worker(rank: int, world_size: int, config: Dict):
    print(f"DDP: 启动 Rank {rank}/{world_size}")
    setup_ddp(rank, world_size, config['DDP_PORT'])
    device = torch.device(f"cuda:{rank}")

    # 1. 数据增强
    IMG_H, IMG_W = config['IMG_SIZE']
    
    train_transforms = A.Compose([
        # 1. 核心几何变换
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Resize(height=IMG_H, width=IMG_W, interpolation=cv2.INTER_LINEAR),

        # 3. 模糊增强 (模拟焦距)
        A.OneOf([
            A.Blur(blur_limit=(1, 3), p=1.0),
            A.MedianBlur(blur_limit=(1, 3), p=1.0),
            A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        ], p=0.2),

        # 4. 颜色和对比度 (模拟染色)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),

        # 5. 噪声增强 (模拟传感器)
        A.OneOf([
            A.GaussNoise(var_limit=(5, 15), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.15), p=1.0),
        ], p=0.25), # 总共有 25% 的概率应用其中一种噪声
        
        # 7. 归一化 (必须在最后)
        A.Normalize(mean=config['NORM_MEAN'], std=config['NORM_STD']),
        ToTensorV2(),
    ], 
                           
                           
    keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ),
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    )
    )
    
    val_transforms = A.Compose([
        A.Resize(height=IMG_H, width=IMG_W, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=config['NORM_MEAN'], std=config['NORM_STD']),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )
    
    # 2. 数据集
    train_dataset = CocoMtlDataset(
        ann_file=config['DATA_PATHS']['TRAIN_ANN_FILE'],
        img_dir=config['DATA_PATHS']['TRAIN_IMG_DIR'],
        img_size=(IMG_H, IMG_W),
        simcc_ratio=config['MODEL_PARAMS']['POSE_SIMCC_RATIO'],
        transforms=train_transforms
    )
    val_dataset = CocoMtlDataset(
        ann_file=config['DATA_PATHS']['VAL_ANN_FILE'],
        img_dir=config['DATA_PATHS']['VAL_IMG_DIR'],
        img_size=(IMG_H, IMG_W),
        simcc_ratio=config['MODEL_PARAMS']['POSE_SIMCC_RATIO'],
        transforms=val_transforms
    )
    
    # 3. [DDP] Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['BATCH_SIZE'], sampler=train_sampler,
        num_workers=config['NUM_WORKERS'], collate_fn=mtl_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['BATCH_SIZE'], sampler=val_sampler,
        num_workers=config['NUM_WORKERS'], collate_fn=mtl_collate_fn, pin_memory=True
    )

    # 4. 初始化模型
    num_seg_classes = config['MODEL_PARAMS']['SEG_NUM_CLASSES']
    model = ChromosomeMTLModel(
        img_size=(IMG_H, IMG_W),
        num_queries=config['MODEL_PARAMS']['NUM_QUERIES'],
        seg_num_classes=num_seg_classes,
        pose_num_keypoints=config['MODEL_PARAMS']['POSE_NUM_KEYPOINTS'],
        pose_simcc_ratio=config['MODEL_PARAMS']['POSE_SIMCC_RATIO']
    ).to(device)
    
    # [!! V7 修复: 预训练权重加载 !!]
    if "PRETRAINED_PATH" in config and config["PRETRAINED_PATH"] and rank == 0:
        pretrain_path = config["PRETRAINED_PATH"]
        if os.path.exists(pretrain_path):
            print(f"Rank 0: 正在从 {pretrain_path} 加载主干网预训练权重...")
            try:
                pretrained_dict = torch.load(pretrain_path, map_location=device)
                
                # 自动剥离前缀
                if 'state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['state_dict']
                
                clean_dict = {}
                for k, v in pretrained_dict.items():
                    new_k = k
                    if new_k.startswith('module.'):
                        new_k = new_k[len('module.'):]
                    if new_k.startswith('model.'):
                        new_k = new_k[len('model.'):]
                    clean_dict[new_k] = v
                pretrained_dict = clean_dict
                
                model_dict = model.state_dict()
                
                # 过滤掉不匹配的键
                filtered_dict = {}
                for k_pretrained, v in pretrained_dict.items():
                    
                    # 1. 尝试添加 "backbone." 前缀 (e.g., 预训练键 "stage1...")
                    k_model = f"backbone.{k_pretrained}" 
                    if k_model in model_dict and v.shape == model_dict[k_model].shape:
                        filtered_dict[k_model] = v
                    
                    # 2. 尝试直接匹配 (e.g., 预训练键 "backbone.stage1...")
                    elif k_pretrained in model_dict and v.shape == model_dict[k_pretrained].shape:
                        filtered_dict[k_pretrained] = v

                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=False)
                
                if len(filtered_dict) > 0:
                    print(f"Rank 0: 成功加载 {len(filtered_dict)} 个预训练层。")
                else:
                    print(f"Rank 0: 警告: 成功加载 0 个预训练层。键名可能不匹配。")
                    print("         (模型键 示例: 'backbone.stem.conv.weight')")
                    print("         (权重键 示例: 'stem.conv.weight' 或 'backbone.stem.conv.weight')")

            except Exception as e:
                print(f"Rank 0: 加载预训练权重失败: {e}")
                print("         将从头开始训练。")
        else:
            print(f"Rank 0: 警告 - 预训练路径未找到: {pretrain_path}。将从头开始训练。")
    
    dist.barrier()

    # 5. [DDP] 包装模型
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    # 6. 初始化损失函数
    matcher = HungarianMatcher(
        cost_class=config['LOSS_WEIGHTS']['COST_CLASS'],
        cost_mask=config['LOSS_WEIGHTS']['COST_MASK'],
        cost_dice=config['LOSS_WEIGHTS']['COST_DICE']
    )
    loss_fn = MTLAllLosses(
        num_classes=num_seg_classes,
        matcher=matcher,
        weights=config['LOSS_WEIGHTS'],
        pose_beta=config.get('POSE_PARAMS', {}).get('BETA', 1.0),
        pose_label_softmax=config.get('POSE_PARAMS', {}).get('LABEL_SOFTMAX', False)
    ).to(device)

    # 7. 初始化优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(), lr=config['LR'] * world_size,
        weight_decay=config['WEIGHT_DECAY']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config['SCHEDULER_STEP'], 
        gamma=config['SCHEDULER_GAMMA']
    )

    # 8. 训练循环
    best_val_metric = -1.0 # [!! V6: 更改为 PCK@0.5 !!]
    
    if rank == 0:
        os.makedirs(config['CHECKPOINT_PATH'], exist_ok=True)
        with open(config['LOG_FILE'], 'w') as f:
            # [!! V6: 更改日志表头 !!]
            f.write("Epoch,TrainLoss,ValLoss,mIoU,mAcc,PCK@0.5,AUC,LR\n")
        print("\n[!! 开始训练 !!]")
    
    start_time = time.time()
    
    for epoch in range(1, config['EPOCHS'] + 1):
        
        train_loader.sampler.set_epoch(epoch) # 确保 DDP shuffle
        
        # [!! V7 修复: 传入 grad_clip_norm !!]
        avg_train_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch, rank,
            grad_clip_norm=config['GRAD_CLIP_NORM']
        )
        
        val_metrics = validate(
            model, loss_fn, val_loader, device, rank,
            num_classes=num_seg_classes,
            config=config
        )
        
        avg_train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
        avg_train_loss = avg_train_loss_tensor.item()
        
        scheduler.step()
        
        # 9. [日志] [保存] - 仅在主进程 (rank 0) 执行
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            
            avg_val_loss = val_metrics["loss"]
            mIoU = val_metrics["mIoU"]
            mAcc = val_metrics["mAcc"]
            pck5 = val_metrics["PCK@0.5"] # [!! V6: 更改键名 !!]
            auc = val_metrics["AUC"]

            print(f"--- Epoch {epoch} 总结 ---")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")
            print(f"  Avg Val Loss:   {avg_val_loss:.4f}")
            print(f"  Seg mIoU:       {mIoU:.4f}")
            print(f"  Seg mAcc:       {mAcc:.4f}")
            print(f"  Pose PCK@0.5:   {pck5:.4f}") # [!! V6: 更改 !!]
            print(f"  Pose AUC:       {auc:.4f}")
            print(f"  Current LR:     {current_lr:.6f}")
            
            with open(config['LOG_FILE'], 'a') as f:
                f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{mIoU:.4f},{mAcc:.4f},{pck5:.4f},{auc:.4f},{current_lr:.6f}\n")

            # [!! V6: 基于 PCK@0.5 保存 !!]
            if pck5 > best_val_metric:
                best_val_metric = pck5
                save_path = os.path.join(config['CHECKPOINT_PATH'], "best_model.pth")
                torch.save(model.module.state_dict(), save_path)
                print(f"  [!!] 新的最佳模型 (PCK@0.5: {pck5:.4f}) 已保存到: {save_path}")

            if epoch % config['SAVE_EVERY'] == 0:
                epoch_save_path = os.path.join(config['CHECKPOINT_PATH'], f"epoch_{epoch}_model.pth")
                torch.save(model.module.state_dict(), epoch_save_path)
                print(f"  [!!] 定期检查点已保存到: {epoch_save_path}")

            latest_path = os.path.join(config['CHECKPOINT_PATH'], "latest_model.pth")
            torch.save(model.module.state_dict(), latest_path)
        
        dist.barrier()

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n[!! 训练完成 !!] 总耗时: {total_time/3600:.2f} 小时")

    cleanup_ddp()


# ----------------------------
# 6. 启动器 (不变)
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="DDP 训练脚本 (V7 - 修复)")
    parser.add_argument("--config", type=str, default="/home/yxb/project/MTL_kary/config.json",
                        help="指向 config.json 文件的路径")
    parser.add_argument("--gpus", type=str, default="2,3", 
                        help="指定要使用的 GPU ID, 逗号分隔 (e.g., '0,1,2')")
    parser.add_argument("--log_file", type=str, default="training_log.csv",
                        help="保存日志的文件名")
    parser.add_argument("--save_every", type=int, default=1,
                        help="每 N 个 epoch 保存一次检查点")
    parser.add_argument("--port", type=str, default="29375",
                        help="DDP 主端口号")
    args = parser.parse_args()

    # 1. 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到: {args.config}")
        exit(1)

    # 2. 更新 config
    config['GPUS'] = args.gpus
    config['LOG_FILE'] = args.log_file
    config['SAVE_EVERY'] = args.save_every
    config['DDP_PORT'] = args.port
    # [!! V7: 为梯度裁剪添加默认值 !!]
    if 'GRAD_CLIP_NORM' not in config:
        config['GRAD_CLIP_NORM'] = 1.0 

    # 3. 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config['GPUS']
    world_size = len(config['GPUS'].split(','))
    
    if world_size > torch.cuda.device_count():
        print(f"错误: 请求了 {world_size} 个 GPU, 但只有 {torch.cuda.device_count()} 个可用。")
        print(f" (CUDA_VISIBLE_DEVICES='{config['GPUS']}')")
        exit(1)

    print(f"将使用 {world_size} 个 GPU: {config['GPUS']}")

    # 4. [DDP] 启动多进程
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()