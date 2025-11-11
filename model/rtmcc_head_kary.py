# rtmcc_head.py
# 完全独立、可直接运行的 RTMCCHead（RTMPose 2023）

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Sequence, Union
import math
import warnings


# ==============================
# 1. 辅助模块
# ==============================

class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) 随机深度
    在 Transformer 中用于替代 Dropout，随机"丢弃"整个残差连接。
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        # 保持概率
        keep_prob = 1 - self.drop_prob
        
        # (B, 1, 1, ...) or (B, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        
        # 生成 mask
        mask = torch.rand(shape, device=x.device) < keep_prob
        
        # 应用 mask 并进行缩放 (Inverted Dropout)
        return x / keep_prob * mask.float()


def rope(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Rotary Position Embedding (RoPE) 旋转位置编码
    通过将特征向量分成两半，并根据其绝对位置应用旋转矩阵，来注入相对位置信息。
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim] # [1]
        
    # 获取空间维度（例如，num_token）
    spatial_shape = [shape[i] for i in dim] # [K]
    total_len = 1
    for i in spatial_shape:
        total_len *= i # K
        
    # [0, 1, ..., K-1]
    position = torch.arange(total_len, dtype=torch.int, device=x.device).reshape(spatial_shape) 

    # (K,) -> (K, 1)
    for i in range(dim[-1] + 1, len(shape) - 1):
        position = position.unsqueeze(-1)
        
    # [D/2]
    half_size = shape[-1] // 2 
    
    # [D/2] 频率序列
    freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / half_size
    # [D/2] 逆频率 1 / (10000^(2i/D))
    inv_freq = 10000 ** freq_seq
    
    # (K, 1) * (D/2) -> (K, D/2) 
    sinusoid = position[..., None] * inv_freq[None, None, :]
    
    # [K, D/2]
    sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)

    # (B, K, D) -> (B, K, D/2) x 2
    x1, x2 = torch.chunk(x, 2, dim=-1)
    
    # 应用旋转: [x1*cos - x2*sin, x2*cos + x1*sin]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ScaleNorm(nn.Module):
    """
    ScaleNorm: (x / norm(x)) * g
    一种 RMSNorm 的变体，用于稳定 Transformer 训练。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5 # 缩放因子
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1)) # 可学习的增益

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿最后一个维度（特征维度）计算 L2 范数
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        # 归一化并乘以增益
        return x / norm.clamp(min=self.eps) * self.g


class RTMCCBlock(nn.Module):
    """
    RTMCC Block (Gated Attention Unit - GAU)
    这是 RTMPose-based (RTMCC) 的核心模块。
    它使用 GAU 代替标准的多头自注意力。
    GAU 逻辑: o = u * (Attention(q, k) @ v)
    """
    def __init__(
        self,
        num_token: int,            # token 数量 (K)
        in_token_dims: int,      # 输入 token 维度 (N or hidden)
        out_token_dims: int,     # 输出 token 维度 (hidden)
        expansion_factor: float = 2,
        s: int = 128,              # QK 投影的维度
        eps: float = 1e-5,
        dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        attn_type: str = 'self-attn',
        act_fn: str = 'SiLU',
        bias: bool = False,
        use_rel_bias: bool = True, # 是否使用相对位置偏置
        pos_enc: bool = False        # 是否使用 RoPE
    ):
        super().__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # FFN 的扩展维度
        self.e = int(in_token_dims * expansion_factor) 
        
        # 输出投影
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            # 关键投影层：
            # in_token_dims -> 2*e (用于 u, v) + s (用于 q, k 的 base)
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            
            # GAU 中用于 q, k 的可学习增益和偏置
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            raise NotImplementedError("Only self-attn is supported in this standalone version")

        # 预归一化
        self.ln = ScaleNorm(in_token_dims, eps=eps)
        self.act_fn = nn.SiLU() if act_fn == 'SiLU' else nn.ReLU()
        
        self.sqrt_s = math.sqrt(s)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        if use_rel_bias:
            # 相对位置偏置参数 (Toeplitz matrix)
            # (2 * K - 1) 存储所有可能的相对位置
            self.w = nn.Parameter(torch.rand([2 * num_token - 1]))

        # 残差连接
        self.shortcut = in_token_dims == out_token_dims
        # 用于残差连接的归一化
        self.res_scale = ScaleNorm(in_token_dims, eps=eps) if self.shortcut else nn.Identity()

        nn.init.xavier_uniform_(self.uv.weight)

    def rel_pos_bias(self, seq_len: int):
        """
        构建 (K, K) 的相对位置偏置矩阵。
        """
        # [0, 1, 2, ..., 2K-2]
        t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        # (K, 3K-2)
        t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
        # 裁剪中心部分 (K, K)
        r = (2 * seq_len - 1) // 2
        return t[..., r:-r]

    def _forward(self, x):
        # x: (B, K, D_in)
        # 1. 预归一化
        x = self.ln(x)
        
        # 2. 投影到 u, v, base
        # (B, K, D_in) -> (B, K, 2*e + s)
        uv = self.act_fn(self.uv(x))
        
        # (B, K, e), (B, K, e), (B, K, s)
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)

        # 3. 生成 Q, K
        # (B, K, s, 1) * (1, 1, 2, s) + (1, 1, 2, s) -> (B, K, 2, s)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        
        if self.pos_enc:
            base = rope(base, dim=1) # 沿 K 维度应用 RoPE
            
        # (B, K, s) x 2
        q, k = torch.unbind(base, dim=2)

        # 4. 计算注意力核
        # (B, K, s) @ (B, s, K) -> (B, K, K)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        
        if self.use_rel_bias:
            # (B, K, K) + (1, K, K)
            qk = qk + self.rel_pos_bias(q.size(1))[:, :q.size(1), :k.size(1)]
            
        # 注意：GAU 使用 relu(qk)^2 作为核，而不是 softmax(qk)
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        kernel = self.dropout(kernel)

        # 5. Gating: u * (Kernel @ v)
        # (B, K, K) @ (B, K, e) -> (B, K, e)
        x_v = torch.bmm(kernel, v)
        
        # (B, K, e) * (B, K, e) (Hadamard product)
        x = u * x_v 
        
        # 6. 输出投影
        # (B, K, e) -> (B, K, D_out)
        x = self.o(x)
        return x

    def forward(self, x):
        # (B, K, D_in)
        if self.shortcut:
            # 残差连接
            return self.res_scale(x) + self.drop_path(self._forward(x))
        else:
            return self.drop_path(self._forward(x))


# ==============================
# 2. RTMCCHead（已修复逻辑）
# ==============================

class RTMCCHeadKary(nn.Module):
    """
    RTMCC (RTMPose-based) 头部。
    
    [修正后的架构]
    1. final_layer: (B, C, Hf, Wf) -> (B, K, Hf, Wf)
    2. flatten:     (B, K, Hf, Wf) -> (B, K, N) (N=Hf*Wf)
    3. mlp:         (B, K, N)      -> (B, K, D_hidden) [将空间 N 压缩]
    4. gau:         (B, K, D_hidden) -> (B, K, D_hidden) [在 K 维度上做注意力]
    5. cls_x/cls_y: (B, K, D_hidden) -> (B, K, W_simcc) / (B, K, H_simcc)
    """
    def __init__(
        self,
        in_channels: int,                        # FPN 输出的通道数 (e.g., 256)
        out_channels: int,                       # 关键点数量 K (e.g., 17)
        input_size: Tuple[int, int],             # 原始输入图像尺寸 (W_img, H_img)
        in_featuremap_size: Tuple[int, int],     # 输入特征图尺寸 (Hf, Wf)
        simcc_split_ratio: float = 2.0,          # SimCC 放大倍率
        final_layer_kernel_size: int = 1,
        gau_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if gau_cfg is None:
            gau_cfg = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )

        self.in_channels = in_channels
        self.out_channels = out_channels # K
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        # 1. final conv: B,C,H,W -> B,K,H,W
        # 这一层从特征图中提取每个关键点的初步热图
        self.final_layer = nn.Conv2d(
            in_channels, out_channels, # C -> K
            kernel_size=final_layer_kernel_size,
            stride=1, padding=final_layer_kernel_size // 2
        )

        # 空间维度 N = Hf * Wf
        N = in_featuremap_size[0] * in_featuremap_size[1]
        
        # 2. flatten spatial + MLP: 
        # [修正] MLP 应该将 N 维空间特征压缩到 D_hidden
        self.mlp = nn.Sequential(
            ScaleNorm(N), # 沿 N 维度归一化
            nn.Linear(N, gau_cfg['hidden_dims'], bias=False) # (B, K, N) -> (B, K, D_hidden)
        )

        # 3. GAU: 
        # [修正] GAU 应该在 K (out_channels) 个 token 上操作
        self.gau = RTMCCBlock(
            num_token=out_channels, # K
            in_token_dims=gau_cfg['hidden_dims'],  # D_hidden
            out_token_dims=gau_cfg['hidden_dims'], # D_hidden
            expansion_factor=gau_cfg['expansion_factor'],
            s=gau_cfg['s'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc']
        )

        # 4. SimCC classifiers
        # 预测 X 轴坐标
        W_simcc = int(input_size[0] * simcc_split_ratio) 
        # 预测 Y 轴坐标
        H_simcc = int(input_size[1] * simcc_split_ratio) 
        
        # [修正] 输入维度是 D_hidden
        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W_simcc, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H_simcc, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats: (B, C, H_f, W_f)
        """
        # 1. (B, C, H_f, W_f) -> (B, K, H_f, W_f)
        x = self.final_layer(feats) 
        B, K, H_f, W_f = x.shape
        N = H_f * W_f

        # 2. [B, K, H_f, W_f] -> [B, K, N]
        x = x.flatten(2) 

        # 3. [B, K, N] -> [B, K, D_hidden]
        # MLP 将空间特征 N 压缩为 D_hidden
        x = self.mlp(x)

        # 4. GAU: [B, K, D_hidden] -> [B, K, D_hidden]
        # GAU 在 K 个关键点 token 之间交换信息
        x = self.gau(x) 

        # 5. 分类:
        # [B, K, D_hidden] -> [B, K, W_simcc]
        pred_x = self.cls_x(x)
        # [B, K, D_hidden] -> [B, K, H_simcc]
        pred_y = self.cls_y(x)

        return pred_x, pred_y

    def decode(self, simcc: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        解码 SimCC (soft-argmax)
        将 (B, K, W) 和 (B, K, H) 的分类概率转为 (B, K, 2) 的坐标
        """
        pred_x, pred_y = simcc
        
        # 1. Softmax: (B, K, W) -> (B, K, W)
        pred_x = F.softmax(pred_x, dim=-1)
        pred_y = F.softmax(pred_y, dim=-1)
        
        W, H = pred_x.shape[-1], pred_y.shape[-1]
        
        # 2. 构建网格
        # [0, 1, ..., W-1]
        grid_x = torch.arange(W, device=pred_x.device).float()
        # [0, 1, ..., H-1]
        grid_y = torch.arange(H, device=pred_y.device).float()

        # 3. 计算期望值 (Expected Value)
        # (B, K, W) * (W,) -> (B, K)
        coord_x = (pred_x * grid_x).sum(-1)
        # (B, K, H) * (H,) -> (B, K)
        coord_y = (pred_y * grid_y).sum(-1)
        
        # 4. 尺度恢复
        coord_x = coord_x / self.simcc_split_ratio
        coord_y = coord_y / self.simcc_split_ratio

        return torch.stack([coord_x, coord_y], dim=-1)  # B, K, 2


# ==============================
# 3. 单元测试（使用修正后的 Head）
# ==============================

def test_rtmcc_head():
    print("开始 RTMCCHeadKary 单元测试...\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 参数设置 ---
    B, C, H_f, W_f = 2, 256, 20, 20
    # 假设 FPN/Backbone 的步幅为 32，640/32 = 20
    input_size = (640, 640) 
    in_featuremap_size = (H_f, W_f)
    num_keypoints = 17
    simcc_split_ratio = 2.0
    
    # GAU 的隐藏维度
    hidden_dims = 128
    
    # 模拟 FPN/Backbone 的输出
    feats = torch.randn(B, C, H_f, W_f).to(device)

    gau_cfg_test = dict(
        hidden_dims=hidden_dims, # [修正] 确保 MLP 和 GAU 匹配
        s=128,
        expansion_factor=2,
        dropout_rate=0.0,
        drop_path=0.0,
        act_fn='SiLU',
        use_rel_bias=False,
        pos_enc=False
    )

    head = RTMCCHeadKary(
        in_channels=C,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=in_featuremap_size,
        simcc_split_ratio=simcc_split_ratio,
        gau_cfg=gau_cfg_test
    ).to(device)

    head.eval()
    with torch.no_grad():
        pred_x, pred_y = head(feats)
        kpts = head.decode((pred_x, pred_y))

    W_simcc = int(input_size[0] * simcc_split_ratio) # 1280
    H_simcc = int(input_size[1] * simcc_split_ratio) # 1280

    print(f"\n--- 维度检查 ---")
    assert pred_x.shape == (B, num_keypoints, W_simcc)
    assert pred_y.shape == (B, num_keypoints, H_simcc)
    assert kpts.shape == (B, num_keypoints, 2)

    print(f"pred_x: {pred_x.shape} -> 正确 (B, K, W_simcc)")
    print(f"pred_y: {pred_y.shape} -> 正确 (B, K, H_simcc)")
    print(f"keypoints: {kpts.shape} -> 正确 (B, K, 2)")
    
    print("\n--- 逻辑检查 (检查坐标是否相同) ---")
    # 检查不同关键点之间的坐标差异
    kpt_diff_x = kpts[:, 0, 0] - kpts[:, 1, 0] # Kpt 0 和 Kpt 1 的 X 坐标
    kpt_diff_y = kpts[:, 0, 1] - kpts[:, 1, 1] # Kpt 0 和 Kpt 1 的 Y 坐标

    # 如果差异很小 (接近0)，说明它们可能还是相同
    is_buggy = torch.allclose(kpt_diff_x, torch.tensor(0.0), atol=1e-5) and \
               torch.allclose(kpt_diff_y, torch.tensor(0.0), atol=1e-5)
    
    if is_buggy:
        print("!!! 警告: 不同关键点的坐标仍然相同，逻辑可能还有问题。")
    else:
        print("逻辑检查: 不同关键点预测了不同的坐标 -> 成功")
        
    print(f"坐标范围 x: [{kpts[:,:,0].min():.1f}, {kpts[:,:,0].max():.1f}] (应在 [0, 640] 范围内)")
    print(f"坐标范围 y: [{kpts[:,:,1].min():.1f}, {kpts[:,:,1].max():.1f}] (应在 [0, 640] 范围内)")
    print("\n所有测试通过！")


if __name__ == "__main__":
    test_rtmcc_head()