# mtl_model.py
# [!! 最终修复: "Plan D" 架构 !!]
# 1. 彻底移除 "盲目" 的 Mask2FormerHeadKary。
# 2. 替换为简单的 FCN 式语义分割头 (SemanticSegHead)。
# 3. 保持解耦的 RTMCCHeadKary 不变。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from .cspnext_kary import CSPNeXt
from .mask2former_head_kary import PAFPN # 只保留 PAFPN
from .rtmcc_head_kary import RTMCCHeadKary

# [!! 新增 !!] 一个简单的 FCN 分割头
class SemanticSegHead(nn.Module):
    """一个简单的 FCN 风格分割头"""
    def __init__(self, in_channels: int, num_classes: int, target_size: Tuple[int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes # 注意: num_classes 应该包含背景类 (e.g., C+1)
        self.target_size = target_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, 1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H/4, W/4)
        x = self.conv(x)
        # (B, NumClasses, H/4, W/4) -> (B, NumClasses, H, W)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x


class ChromosomeMTLModel(nn.Module):
    """
    染色体多任务模型 (Plan D: FCN + RTMCC)
    """
    def __init__(
        self,
        backbone_channels: List[int] = [128, 256, 512, 1024],
        backbone_blocks: List[int] = [3, 3, 9, 3],
        pafpn_hidden_dim: int = 256,
        img_size: Tuple[int, int] = (512, 512),
        seg_num_classes: int = 23, # 你的 config 是 26? 
        pose_num_keypoints: int = 3,
        pose_simcc_ratio: float = 2.0
    ):
        super().__init__()
        
        self.img_size = img_size
        self.pafpn_hidden_dim = pafpn_hidden_dim
        self.pose_num_keypoints = pose_num_keypoints
        self.pose_simcc_ratio = pose_simcc_ratio
        
        # [!! 修复 !!]
        # 你的 config.json 说 seg_num_classes = 26。
        # 语义分割通常需要一个 "背景" 类。
        # 假设 26 个类 = 25 个染色体 + 1 个背景。
        # 如果是 26 个染色体，没有背景，你需要 C+1 = 27。
        # 我们这里使用 config.json 的 26。
        self.seg_num_classes = seg_num_classes 

        # 1. 主干网 (Backbone)
        self.backbone = CSPNeXt(
            in_channels=3,
            stem_channels=64,
            stage_channels=backbone_channels,
            num_blocks=backbone_blocks
        )
        
        # 2. 共享颈部 (Shared Neck)
        self.pafpn = PAFPN(
            in_channels=backbone_channels,
            out_channels=self.pafpn_hidden_dim
        )
        
        # 3. 分割头 (Seg Head)
        # [!! 修复 !!] 替换为 SemanticSegHead
        # 它需要知道目标大小 (H, W) 和类别数 C
        self.seg_head = SemanticSegHead(
            in_channels=self.pafpn_hidden_dim,
            num_classes=self.seg_num_classes, # 使用 config.json 中的 26
            target_size=self.img_size
        )
        
        # 4. 姿态头 (Pose Head)
        H_f, W_f = img_size[0] // 4, img_size[1] // 4
        W_s = int(img_size[1] * self.pose_simcc_ratio)
        H_s = int(img_size[0] * self.pose_simcc_ratio)
        
        gau_cfg_for_pose = dict(
            hidden_dims=256, s=128, expansion_factor=2,
            dropout_rate=0.0, drop_path=0.0, act_fn='SiLU',
            use_rel_bias=False, pos_enc=False 
        )

        self.pose_head = RTMCCHeadKary(
            in_channels=self.pafpn_hidden_dim,
            out_channels=self.pose_num_keypoints,
            input_size=img_size,
            in_featuremap_size=(H_f, W_f),
            simcc_split_ratio=self.pose_simcc_ratio,
            gau_cfg=gau_cfg_for_pose
        )


    def forward(self, img: torch.Tensor, img_metas: List[Dict] = None) -> Dict[str, torch.Tensor]:
        
        backbone_feats = self.backbone(img)
        pafpn_feats = self.pafpn(backbone_feats)
        
        # N2 特征图 (B, C, H/4, W/4)
        n2_features = pafpn_feats[0]
        
        # 3. Seg Head (使用 N2)
        # [!! 修复 !!]
        # (B, NumClasses, H, W)
        seg_logits = self.seg_head(n2_features)
        
        # 4. Pose Head (使用 N2)
        # (B, K, Ws), (B, K, Hs)
        pose_pred_x, pose_pred_y = self.pose_head(n2_features)
        
        # 5. 组合输出
        outputs = {
            # [!! 修复 !!] 键名改变
            "seg_logits": seg_logits,
            
            # 姿态输出
            "pose_pred_x": pose_pred_x,
            "pose_pred_y": pose_pred_y,
        }
        
        return outputs
