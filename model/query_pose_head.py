# query_pose_head.py
# [!! 修复版 !!]
# 修复了导致模型崩溃的架构问题

import torch
import torch.nn as nn
from typing import Tuple, Dict
from .rtmcc_head_kary import RTMCCBlock, ScaleNorm
import torch.nn.functional as F

class QueryPoseHead(nn.Module):
    """
    Query-based Pose Head (修复版)
    
    接收 (B, Nq, D_hidden)，输出 (B, Nq, K, Ws/Hs)
    
    [修复]
    1. GAU (可选)
    2. 一个MLP，将 (B, Nq, D) 映射到 (B, Nq, K * D_kpt)
       (D_kpt 是每个关键点的内部特征维度)
    3. Reshape 为 (B, Nq, K, D_kpt)
    4. 两个 *共享* 的 Linear (D_kpt -> Ws) 和 (D_kpt -> Hs)
       分别应用于 (B*Nq*K, D_kpt)
    """
    def __init__(
        self,
        num_queries: int,         # Nq (e.g., 100)
        in_hidden_dim: int,     # D_hidden (e.g., 256)
        num_keypoints: int,       # K (e.g., 5)
        simcc_dims: Tuple[int, int], # (W_s, H_s)
        gau_cfg: Dict,             # GAU 模块的配置
        kpt_feat_dims: int = 128   # 每个关键点的内部特征维度
    ):
        super().__init__()
        self.num_queries = num_queries
        self.in_hidden_dim = in_hidden_dim
        self.num_keypoints = num_keypoints
        self.simcc_W, self.simcc_H = simcc_dims
        self.kpt_feat_dims = kpt_feat_dims

        # 1. GAU (可选, 但可以增强 query 间的交互)
        self.gau = RTMCCBlock(
            num_token=num_queries,
            in_token_dims=in_hidden_dim,
            out_token_dims=in_hidden_dim,
            **gau_cfg
        )
        self.ln = ScaleNorm(in_hidden_dim)
        
        # 2. 姿态预测 MLP
        # (B, Nq, D_hidden) -> (B, Nq, K * D_kpt)
        # 这个MLP为 K 个关键点中的每一个生成一个 D_kpt 维的特征
        self.kpt_feat_proj = nn.Sequential(
            nn.Linear(in_hidden_dim, in_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(in_hidden_dim * 2, num_keypoints * kpt_feat_dims)
        )

        # 3. SimCC 预测头
        # [!! 关键修复 !!]
        # 这两个头在所有 K 个关键点之间 *共享* 权重
        # (D_kpt) -> (Ws)
        self.cls_x = nn.Linear(kpt_feat_dims, self.simcc_W)
        # (D_kpt) -> (Hs)
        self.cls_y = nn.Linear(kpt_feat_dims, self.simcc_H)


    def forward(self, decoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_out (Tensor): (B, Nq, D_hidden)
        
        Returns:
            pred_x (Tensor): (B, Nq, K, Ws)
            pred_y (Tensor): (B, Nq, K, Hs)
        """
        B, Nq, _ = decoder_out.shape
        
        # 1. 通过 GAU (可选)
        x = self.ln(self.gau(decoder_out))
        
        # 2. 预测 (B, Nq, D) -> (B, Nq, K * D_kpt)
        kpt_feats = self.kpt_feat_proj(x)
        
        # 3. Reshape 为 (B, Nq, K, D_kpt)
        kpt_feats_reshaped = kpt_feats.view(B, Nq, self.num_keypoints, self.kpt_feat_dims)
        
        # 4. 展平以应用共享头
        # (B, Nq, K, D_kpt) -> (B*Nq*K, D_kpt)
        kpt_feats_flat = kpt_feats_reshaped.view(-1, self.kpt_feat_dims)
        
        # 5. [!! 关键修复 !!] 应用共享头
        # (B*Nq*K, D_kpt) -> (B*Nq*K, Ws)
        pred_x_flat = self.cls_x(kpt_feats_flat)
        # (B*Nq*K, D_kpt) -> (B*Nq*K, Hs)
        pred_y_flat = self.cls_y(kpt_feats_flat)
        
        # 6. Reshape 回 (B, Nq, K, Ws/Hs)
        pred_x = pred_x_flat.view(B, Nq, self.num_keypoints, self.simcc_W)
        pred_y = pred_y_flat.view(B, Nq, self.num_keypoints, self.simcc_H)
        
        return pred_x, pred_y
