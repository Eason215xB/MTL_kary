# mtl_model.py
# 染色体多任务模型 (分割 + 姿态) - [!! Plan B 架构 !!]
# 组合了 CSPNeXt + 共享 PAFPN + Mask2FormerHead + QueryPoseHead

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .cspnext_kary import CSPNeXt
from .mask2former_head_kary import Mask2FormerHeadKary, PAFPN


# from .rtmcc_head_kary import RTMCCBlock

from .query_pose_head import QueryPoseHead



class ChromosomeMTLModel(nn.Module):
    """
    染色体多任务模型 (Plan B: Query-based)
    
    架构:
    1. Backbone: CSPNeXt -> [C2, C3, C4, C5]
    2. Shared Neck: PAFPN -> [N2, N3, N4, N5]
    3. Seg Head (Decoder): Mask2Former -> (logits, masks, decoder_out)
    4. Pose Head: QueryPoseHead (Input: decoder_out)
    """
    def __init__(
        self,
        # --- 模型配置 ---
        backbone_channels: List[int] = [128, 256, 512, 1024],
        backbone_blocks: List[int] = [3, 3, 9, 3],
        pafpn_hidden_dim: int = 256,
        
        # --- 图像尺寸 ---
        img_size: Tuple[int, int] = (512, 512),
        
        # --- 共享查询 (Query) ---
        num_queries: int = 100,
        
        # --- 分割头 (Mask2Former) ---
        seg_num_classes: int = 23,
        seg_decoder_layers: int = 6,
        
        # --- 姿态头 (RTMCC-based) ---
        pose_num_keypoints: int = 3, # [!! 修复: 匹配config.json中的设置 !!]
        pose_simcc_ratio: float = 2.0
    ):
        super().__init__()
        
        # (H, W)
        self.img_size = img_size
        
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
            out_channels=pafpn_hidden_dim
        )
        
        # 3. 分割头 (Seg Head)
        # (包含 Transformer Decoder)
        self.seg_head = Mask2FormerHeadKary(
            num_classes=seg_num_classes,
            hidden_dim=pafpn_hidden_dim, 
            num_queries=num_queries,
            num_decoder_layers=seg_decoder_layers,
            num_levels=len(backbone_channels), # 4
            num_heads=8,
            num_points=4,
            ffn_dim=2048
        )
        
        # 4. 姿态头 (Pose Head)
        # [!! 替换 !!]
        W_s = int(img_size[1] * pose_simcc_ratio)
        H_s = int(img_size[0] * pose_simcc_ratio)
        
        # RTMCCBlock (GAU) 的示例配置
        # (这来自你原始的 rtmcc_head_kary.py)
        gau_cfg_for_pose = dict(
            expansion_factor=2,
            s=128,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False, # Query-based GAU 通常不使用 rel_bias
            pos_enc=True      # RoPE 对 Query 很有用
        )

        self.pose_head = QueryPoseHead(
            num_queries=num_queries,
            in_hidden_dim=pafpn_hidden_dim, # 256
            num_keypoints=pose_num_keypoints,
            simcc_dims=(W_s, H_s),
            gau_cfg=gau_cfg_for_pose,
            kpt_feat_dims=128
        )

    def forward(self, img: torch.Tensor, img_metas: List[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # 1. Backbone: (B, 3, H, W) -> [C2, C3, C4, C5]
        backbone_feats = self.backbone(img)
        
        # 2. Shared Neck: [C...] -> [N2, N3, N4, N5]
        pafpn_feats = self.pafpn(backbone_feats)
        
        # 3. Seg Head (包含 Decoder)
        # [!! 更改 !!]
        # seg_outputs 现在是一个字典: {"class_logits", "mask_logits", "decoder_out"}
        seg_outputs = self.seg_head(pafpn_feats, img_metas)
        
        # 4. Pose Head
        # [!! 更改 !!]
        # 姿态头接收来自 Seg/Decoder 的查询嵌入
        decoder_out = seg_outputs["decoder_out"] # (B, Nq, D_hidden)
        
        # pred_x: (B, Nq, K, Ws), pred_y: (B, Nq, K, Hs)
        pose_pred_x, pose_pred_y = self.pose_head(decoder_out)
        
        # 5. 组合输出
        outputs = {
            # 分割输出
            "seg_class_logits": seg_outputs["class_logits"], # (B, Nq, C+1)
            "seg_mask_logits": seg_outputs["mask_logits"],   # (B, Nq, H, W)
            
            # 姿态输出
            "pose_pred_x": pose_pred_x,
            "pose_pred_y": pose_pred_y,
        }
        
        return outputs


# ==============================
# 单元测试 (已更新为 Plan B)
# ==============================
def test_mtl_model():
    print("\n开始 ChromosomeMTLModel (Plan B) 单元测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # --- 配置 ---
    B, H, W = 2, 512, 512
    SEG_CLASSES = 23
    POSE_KPTS = 3 # [!! 修复: 匹配config.json中的设置 !!]
    NUM_QUERIES = 100
    SIMCC_RATIO = 2.0
    
    img = torch.randn(B, 3, H, W).to(device)
    img_metas = [{'pad_shape': (H, W)}] * B

    # --- 初始化模型 ---
    try:
        model = ChromosomeMTLModel(
            img_size=(H, W),
            num_queries=NUM_QUERIES,
            seg_num_classes=SEG_CLASSES,
            pose_num_keypoints=POSE_KPTS,
            pose_simcc_ratio=SIMCC_RATIO
        ).to(device)
    except Exception as e:
        print(f"\n[!! 错误 !!] 模型初始化失败: {e}")
        raise e

    model.eval()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    
    # --- 前向传播 ---
    with torch.no_grad():
        outputs = model(img, img_metas)

    print("\n--- 输出维度检查 ---")
    assert isinstance(outputs, dict)
    
    # 1. 检查分割输出
    expected_class_shape = (B, NUM_QUERIES, SEG_CLASSES + 1)
    expected_mask_shape = (B, NUM_QUERIES, H, W)
    
    assert "seg_class_logits" in outputs
    assert outputs["seg_class_logits"].shape == expected_class_shape
    print(f"seg_class_logits: {outputs['seg_class_logits'].shape} -> 成功!")
    
    assert "seg_mask_logits" in outputs
    assert outputs["seg_mask_logits"].shape == expected_mask_shape
    print(f"seg_mask_logits: {outputs['seg_mask_logits'].shape} -> 成功!")

    # 2. 检查姿态输出
    W_simcc = int(W * SIMCC_RATIO)
    H_simcc = int(H * SIMCC_RATIO)
    
    expected_pose_x_shape = (B, NUM_QUERIES, POSE_KPTS, W_simcc)
    expected_pose_y_shape = (B, NUM_QUERIES, POSE_KPTS, H_simcc)
    
    assert "pose_pred_x" in outputs
    assert outputs["pose_pred_x"].shape == expected_pose_x_shape
    print(f"pose_pred_x: {outputs['pose_pred_x'].shape} -> 成功!")
    
    assert "pose_pred_y" in outputs
    assert outputs["pose_pred_y"].shape == expected_pose_y_shape
    print(f"pose_pred_y: {outputs['pose_pred_y'].shape} -> 成功!")

    print("\n所有测试通过！Plan B 模型已成功组合。")


if __name__ == "__main__":
    test_mtl_model()
