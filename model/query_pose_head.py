# query_pose_head.py
# 一个 "Query-based" 姿态头，用于 Plan B
# 接收 (B, Nq, D_hidden) -> 输出 (B, Nq, K, Ws/Hs)

import torch
import torch.nn as nn
from typing import Tuple
from .rtmcc_head_kary import RTMCCBlock, ScaleNorm


class QueryPoseHead(nn.Module):
    """
    Query-based Pose Head
    
    接收来自 Transformer Decoder 的查询嵌入 (B, Nq, D_hidden)，
    并为每个查询预测一个姿态。
    """
    def __init__(
        self,
        num_queries: int,         # Nq (e.g., 100)
        in_hidden_dim: int,     # D_hidden (e.g., 256)
        num_keypoints: int,       # K (e.g., 5)
        simcc_dims: Tuple[int, int], # (W_s, H_s)
        gau_cfg: dict             # GAU 模块的配置
    ):
        super().__init__()
        self.num_queries = num_queries
        self.in_hidden_dim = in_hidden_dim
        self.num_keypoints = num_keypoints
        self.simcc_W, self.simcc_H = simcc_dims
        
        # 1. GAU (可选, 但可以增强 query 间的交互)
        self.gau = RTMCCBlock(
            num_token=num_queries,
            in_token_dims=in_hidden_dim,
            out_token_dims=in_hidden_dim,
            **gau_cfg
        )
        self.ln = ScaleNorm(in_hidden_dim)
        
        # 2. 姿态预测 MLP
        # (B, Nq, D_hidden) -> (B, Nq, K*W_s)
        self.cls_x = nn.Linear(in_hidden_dim, num_keypoints * self.simcc_W)
        # (B, Nq, D_hidden) -> (B, Nq, K*H_s)
        self.cls_y = nn.Linear(in_hidden_dim, num_keypoints * self.simcc_H)

    def forward(self, decoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_out (Tensor): (B, Nq, D_hidden)
        
        Returns:
            pred_x (Tensor): (B, Nq, K, Ws)
            pred_y (Tensor): (B, Nq, K, Hs)
        """
        B = decoder_out.shape[0]
        
        # 1. 通过 GAU (可选)
        x = self.ln(self.gau(decoder_out))
        
        # 2. 预测
        pred_x = self.cls_x(x) # (B, Nq, K*Ws)
        pred_y = self.cls_y(x) # (B, Nq, K*Hs)
        
        # 3. Reshape
        pred_x = pred_x.view(B, self.num_queries, self.num_keypoints, self.simcc_W)
        pred_y = pred_y.view(B, self.num_queries, self.num_keypoints, self.simcc_H)
        
        return pred_x, pred_y