# cspnext_backbone.py
# 纯 PyTorch 实现 CSPNeXt (L)
# 输出 [C2, C3, C4, C5] (strides [4, 8, 16, 32])
# 通道数 [128, 256, 512, 1024]

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBNAct(nn.Module):
    """
    一个标准的 Conv-BN-Activation 模块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = -1,
        groups: int = 1,
        act: str = 'SiLU'
    ):
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2
            
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CSPNeXtBlock(nn.Module):
    """
    CSPNeXt (Cross-Stage-Partial) 核心模块
    它将输入在通道维度上拆分为两部分，分别通过主路径和短路径，然后融合。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float = 0.5, # 决定 "main" 路径的通道数
        add_identity: bool = True,    # 是否添加残差连接
        act: str = 'SiLU'
    ):
        super().__init__()
        
        # main 路径的通道数
        mid_channels = int(in_channels * expansion_ratio)
        
        # 1. 主路径 (main path)
        self.main_conv = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, 3, act=act),
            ConvBNAct(mid_channels, mid_channels, 3, act=act)
        )
        
        # 2. 短路径 (shortcut path)
        self.short_conv = ConvBNAct(in_channels, mid_channels, 1, act=act)

        # 3. 融合 (fusion)
        # 融合后通道数为 2 * mid_channels，通过 1x1 卷积将其压缩/扩展到 out_channels
        self.final_conv = ConvBNAct(2 * mid_channels, out_channels, 1, act=act)
        
        # 残差连接
        self.add_identity = add_identity and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 1. 主路径
        x_main = self.main_conv(x)
        
        # 2. 短路径
        x_short = self.short_conv(x)

        # 3. 融合 (Concat)
        x = torch.cat((x_main, x_short), dim=1)
        
        # 4. 最终卷积
        x = self.final_conv(x)

        if self.add_identity:
            return identity + x
        else:
            return x


class CSPNeXt(nn.Module):
    """
    CSPNeXt 主干网 (以 CSPNeXt-L 配置为例)
    
    输出:
    - C2 (x1): (B, 128, H/4, W/4)
    - C3 (x2): (B, 256, H/8, W/8)
    - C4 (x3): (B, 512, H/16, W/16)
    - C5 (x4): (B, 1024, H/32, W/32)
    """
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        stage_channels: List[int] = [128, 256, 512, 1024],
        num_blocks: List[int] = [3, 3, 9, 3], # CSPNeXt-L 块配置
        expansion_ratio: float = 0.5,
        act: str = 'SiLU'
    ):
        super().__init__()
        
        # 1. Stem (H, W) -> (H/2, W/2)
        self.stem = ConvBNAct(in_channels, stem_channels, 3, stride=2, act=act)

        # 2. Stage 1 (C2) (H/2, W/2) -> (H/4, W/4)
        self.stage1 = self._make_stage(
            stem_channels, stage_channels[0], num_blocks[0], 
            expansion_ratio, act, stride=2
        )
        
        # 3. Stage 2 (C3) (H/4, W/4) -> (H/8, W/8)
        self.stage2 = self._make_stage(
            stage_channels[0], stage_channels[1], num_blocks[1], 
            expansion_ratio, act, stride=2
        )

        # 4. Stage 3 (C4) (H/8, W/8) -> (H/16, W/16)
        self.stage3 = self._make_stage(
            stage_channels[1], stage_channels[2], num_blocks[2], 
            expansion_ratio, act, stride=2
        )
        
        # 5. Stage 4 (C5) (H/16, W/16) -> (H/32, W/32)
        self.stage4 = self._make_stage(
            stage_channels[2], stage_channels[3], num_blocks[3], 
            expansion_ratio, act, stride=2
        )

        # 初始化权重 (可选但推荐)
        self._init_weights()

    def _make_stage(
        self, in_c, out_c, n_blocks, exp, act, stride
    ):
        """辅助函数，用于构建一个 stage"""
        layers = []
        # 第一个块：下采样 + 通道变换
        layers.append(ConvBNAct(in_c, out_c, 3, stride=stride, act=act))
        
        # 中间的 CSPNeXt 块
        for _ in range(n_blocks):
            layers.append(CSPNeXtBlock(out_c, out_c, exp, add_identity=True, act=act))
            
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回 C2, C3, C4, C5 四个尺度的特征图列表
        """
        x_stem = self.stem(x)
        
        x1 = self.stage1(x_stem) # C2, (B, 128, H/4, W/4)
        x2 = self.stage2(x1)   # C3, (B, 256, H/8, W/8)
        x3 = self.stage3(x2)   # C4, (B, 512, H/16, W/16)
        x4 = self.stage4(x3)   # C5, (B, 1024, H/32, W/32)
        
        return [x1, x2, x3, x4]


# ==============================
# 单元测试
# ==============================
def test_backbone():
    print("\n开始 CSPNeXt-L 主干网单元测试...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    B, H, W = 2, 512, 512
    img = torch.randn(B, 3, H, W).to(device)
    
    # 期望的输出通道
    expected_channels = [128, 256, 512, 1024]
    
    # 期望的输出尺寸 (Strides 4, 8, 16, 32)
    expected_shapes = [
        (B, expected_channels[0], H // 4, W // 4),
        (B, expected_channels[1], H // 8, W // 8),
        (B, expected_channels[2], H // 16, W // 16),
        (B, expected_channels[3], H // 32, W // 32),
    ]

    backbone = CSPNeXt(
        stage_channels=expected_channels,
        num_blocks=[3, 3, 9, 3] # -L config
    ).to(device)
    backbone.eval()

    with torch.no_grad():
        out_feats = backbone(img)

    assert isinstance(out_feats, list)
    assert len(out_feats) == 4
    print("\n输出特征 (C2, C3, C4, C5):")
    
    for i, (feat, shape) in enumerate(zip(out_feats, expected_shapes)):
        assert feat.shape == shape
        print(f"  C{i+2}: {feat.shape} -> 成功!")
        
    print("\n所有测试通过！")


if __name__ == "__main__":
    test_backbone()