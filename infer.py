# inference.py
# 染色体多任务模型 (Plan B) 推理和可视化脚本

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
import time
import json
import argparse
import numpy as np
import cv2
from typing import Dict, List, Tuple
from torch import Tensor

# ----------------------------
# 1. 导入你的所有自定义模块
# ----------------------------

from model.MTL_model import ChromosomeMTLModel
from model.query_pose_head import QueryPoseHead # 确保这个文件也在
from model.rtmcc_head_kary import RTMCCBlock, ScaleNorm # 确保这个文件也在
from model.mask2former_head_kary import PAFPN # 确保这个文件也在
from model.cspnext_kary import CSPNeXt # 确保这个文件也在

# ----------------------------
# 2. 辅助函数
# ----------------------------

def load_config(config_path: str) -> Dict:
    """加载 JSON 配置文件"""
    print(f"从 {config_path} 加载配置")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_model(config: Dict, weights_path: str, device: torch.device) -> ChromosomeMTLModel:
    """初始化模型并加载权重"""
    print("初始化模型...")
    IMG_H, IMG_W = config['IMG_SIZE']
    
    model = ChromosomeMTLModel(
        img_size=(IMG_H, IMG_W),
        num_queries=config['MODEL_PARAMS']['NUM_QUERIES'],
        seg_num_classes=config['MODEL_PARAMS']['SEG_NUM_CLASSES'],
        pose_num_keypoints=config['MODEL_PARAMS']['POSE_NUM_KEYPOINTS'],
        pose_simcc_ratio=config['MODEL_PARAMS']['POSE_SIMCC_RATIO']
    )
    
    print(f"从 {weights_path} 加载权重")
    # 加载 state_dict。
    # train_ddp.py 保存的是 model.module.state_dict()，所以可以直接加载
    state_dict = torch.load(weights_path, map_location=device)
    
    # [!! 关键 !!] 处理权重是否是 DDP 格式
    # 如果 state_dict 中的键以 'module.' 开头, 我们需要去掉它
    if next(iter(state_dict)).startswith('module.'):
        print("检测到 DDP 权重, 正在移除 'module.' 前缀...")
        new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def get_transforms(config: Dict) -> Tuple[A.Compose, A.Compose]:
    """获取推理所需的数据增强 (分离 resize 和 norm)"""
    IMG_H, IMG_W = config['IMG_SIZE']
    
    # 1. 调整大小/填充 (用于图像和可视化)
    resize_transform = A.Compose([
        A.Resize(height=IMG_H, width=IMG_W, interpolation=cv2.INTER_LINEAR),
    ])
    
    # 2. 归一化 (用于张量)
    norm_transform = A.Compose([
        A.Normalize(mean=config['NORM_MEAN'], std=config['NORM_STD']),
        ToTensorV2(),
    ])
    return resize_transform, norm_transform

def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """为类别或关键点生成一组鲜艳的 BGR 颜色"""
    # 使用 HSV 色彩空间生成易于区分的颜色
    colors_hsv = np.linspace(0, 360, num_colors, endpoint=False)
    colors_bgr = []
    for h in colors_hsv:
        # (H, S, V) -> (B, G, R)
        hsv_color = np.uint8([[[h, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors_bgr.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    return colors_bgr

def decode_simcc(
    pred_x_logits: Tensor,  # (N_pred, K, Ws)
    pred_y_logits: Tensor,  # (N_pred, K, Hs)
    simcc_ratio: float
) -> Tensor:
    """
    解码 SimCC (Soft-Argmax)
    返回: (N_pred, K, 2) 格式的 (x, y) 坐标
    """
    device = pred_x_logits.device
    N_pred, K, Ws = pred_x_logits.shape
    _, _, Hs = pred_y_logits.shape
    
    # 1. Softmax
    pred_x_probs = F.softmax(pred_x_logits, dim=-1)
    pred_y_probs = F.softmax(pred_y_logits, dim=-1)
    
    # 2. 1D 网格
    grid_x = torch.arange(Ws, device=device, dtype=torch.float32).view(1, 1, Ws)
    grid_y = torch.arange(Hs, device=device, dtype=torch.float32).view(1, 1, Hs)
    
    # 3. 计算期望值 (soft-argmax)
    coords_x = (pred_x_probs * grid_x).sum(-1) # (N_pred, K)
    coords_y = (pred_y_probs * grid_y).sum(-1) # (N_pred, K)
    
    # 4. 恢复尺度
    coords_x = coords_x / simcc_ratio
    coords_y = coords_y / simcc_ratio
    
    return torch.stack([coords_x, coords_y], dim=-1) # (N_pred, K, 2)


@torch.no_grad()
def post_process(
    outputs: Dict[str, Tensor],
    config: Dict,
    device: torch.device,
    threshold: float
) -> Dict[str, Tensor]:
    """
    解码模型的原始输出。
    """
    # 1. 获取输出并移除批处理维度 (B=1)
    class_logits = outputs['seg_class_logits'].squeeze(0) # (Nq, C+1)
    mask_logits = outputs['seg_mask_logits'].squeeze(0)   # (Nq, H, W)
    pose_x_logits = outputs['pose_pred_x'].squeeze(0)     # (Nq, K, Ws)
    pose_y_logits = outputs['pose_pred_y'].squeeze(0)     # (Nq, K, Hs)
    
    # 2. 类别过滤
    num_classes = config['MODEL_PARAMS']['SEG_NUM_CLASSES']
    class_probs = F.softmax(class_logits, dim=-1) # (Nq, C+1)
    
    # (Nq,)
    scores, labels = class_probs.max(dim=-1)
    
    # 过滤掉 "no object" (id=26) 和低置信度
    keep = (labels != num_classes) & (scores > threshold)

    # if keep.sum() == 0:
    #     # 没有检测到任何物体
    #     return None
        
    # 3. 过滤所有预测
    final_scores = scores[keep]
    final_labels = labels[keep]
    final_masks = mask_logits[keep]
    final_pose_x = pose_x_logits[keep]
    final_pose_y = pose_y_logits[keep]
    
    # 4. 解码掩码
    # (N_pred, H, W) -> binarize
    final_masks = (final_masks.sigmoid() > 0.5)
    
    # 5. 解码姿态
    final_keypoints = decode_simcc(
        final_pose_x, 
        final_pose_y, 
        config['MODEL_PARAMS']['POSE_SIMCC_RATIO']
    )
    print(f'final_scores:{final_scores}')
    print(f'final_keypoints:{final_keypoints}')
    print(f'final_labels:{final_labels}')
    print(f'final_masks:{final_masks}')
    return {
        "scores": final_scores.cpu().numpy(),       # (N_pred,)
        "labels": final_labels.cpu().numpy(),       # (N_pred,)
        "masks": final_masks.cpu().numpy(),         # (N_pred, H, W)
        "keypoints": final_keypoints.cpu().numpy(), # (N_pred, K, 2)
    }

def visualize(
    image_bgr: np.ndarray,
    results: Dict[str, np.ndarray],
    kpt_colors: List[Tuple[int, int, int]],
    class_colors: List[Tuple[int, int, int]],
    class_names: List[str]
) -> np.ndarray:
    """在图像上绘制掩码、关键点和标签"""
    
    output_image = image_bgr.copy()
    overlay = output_image.copy()
    
    scores = results['scores']
    labels = results['labels']
    masks = results['masks']
    keypoints = results['keypoints']
    
    num_kpts = keypoints.shape[1]
    
    # 遍历检测到的每个实例
    for i in range(len(scores)):
        mask = masks[i]         # (H, W)
        kpts = keypoints[i]     # (K, 2)
        label_id = labels[i]
        score = scores[i]

        
        # 1. 绘制掩码 (红色半透明)
        color = (0, 0, 255) # BGR: Red
        overlay[mask] = color
        
        # 2. 绘制关键点
        for k in range(num_kpts):
            kpt_color = kpt_colors[k % len(kpt_colors)] # 循环使用颜色
            x, y = kpts[k]
            cv2.circle(output_image, (int(x), int(y)), 3, kpt_color, -1, lineType=cv2.LINE_AA)
            
        # 3. 绘制标签和分数
        class_color = class_colors[label_id % len(class_colors)]
        class_name = class_names[label_id]
        text = f"{class_name}: {score:.2f}"
        
        # 找到掩码的 bbox 以放置文本
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            # 在 bbox 左上角绘制
            cv2.putText(output_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, class_color, 2)

    # 4. 应用半透明叠加
    # cv2.addWeighted(src1, alpha, src2, beta, gamma)
    final_image = cv2.addWeighted(overlay, 0.4, output_image, 0.6, 0)
    
    return final_image

# ----------------------------
# 5. 主推理函数
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="染色体 MTL (Plan B) 推理脚本")
    parser.add_argument("--input", default='/mnt/afs/yxb/kary-datasets/waizhou/keypoint-origin-data/王老师数据第一批排列图/3_kpts_rotate/images/405.001.K_000_5_rot0.jpg',
                        help="输入图像的路径, 或包含图像的文件夹路径")
    parser.add_argument("--weights", default='/home/yxb/project/MTL_kary/outputs/checkpoints/best_model.pth',
                        help="指向训练好的 .pth 权重文件的路径 (e.g., best_model.pth)")
    parser.add_argument("--output", default='/home/yxb/project/MTL_kary/test',
                        help="保存可视化结果的文件夹路径")
    parser.add_argument("--config", default="config.json",
                        help="指向 config.json 文件的路径")
    parser.add_argument("--gpus", default="7",
                        help="要使用的 GPU ID (e.g., '0')")
    parser.add_argument("--threshold", type=float, default=0.0001,
                        help="实例置信度阈值")
    args = parser.parse_args()

    # --- 1. 设置 ---
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f"cuda:{args.gpus.split(',')[0]}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # --- 2. 加载配置和模型 ---
    config = load_config(args.config)
    model = get_model(config, args.weights, device)
    
    # --- 3. 获取数据预处理器 ---
    IMG_H, IMG_W = config['IMG_SIZE']
    resize_transform, norm_transform = get_transforms(config)

    # --- 4. 准备可视化颜色和标签 ---
    K = config['MODEL_PARAMS']['POSE_NUM_KEYPOINTS']
    C = config['MODEL_PARAMS']['SEG_NUM_CLASSES']
    
    # 关键点颜色 (BGR)
    kpt_colors = [
        (255, 0, 0),   # 蓝色 (head1)
        (0, 255, 0),   # 绿色 (head2)
        (0, 255, 255), # 黄色 (cen)
        (255, 0, 255), # 品红 (tail1)
        (0, 165, 255)  # 橙色 (tail2)
    ]
    
    # 类别颜色
    class_colors = generate_colors(C)
    
    # [!! 关键 !!] 类别名称 (0=1, 1=2, ..., 22=X, 23=Y)
    # 你的 config.json 说 26 个类? 这与 "1-22, X, Y" (24个) 不符。
    # 我将假设有 24 个类 (0-23)
    # COCO 标签 ID 从 1 开始, 但模型输出索引从 0 开始。
    # 你的 CocoMtlDataset 应该已经处理了这个问题 (labels=ann['category_id'])
    # 假设你的 seg.json 中的 category_id 从 1 到 24。
    # 那么你的模型预测的 logits 索引 0-23 对应 1-24。
    
    # 我们创建一个从 0 到 25 的名称列表 (包含 "no object")
    class_names = [str(i+1) for i in range(22)] + ["X", "Y"] # 索引 0-23
    # 你的 config 说 C=26, 这很奇怪。我将假设 C=24
    if C == 26: # 也许 24, 25 是其他?
        class_names = [str(i) for i in range(C)] # Fallback
    elif C == 24:
        pass # 1-22, X, Y
    else:
        # Fallback for C=3 (来自你的 config)
        class_names = [f"Class {i}" for i in range(C)]
        
    print(f"警告: 假设有 {len(class_names)} 个类别: {class_names}")

    # --- 5. 查找输入图像 ---
    image_paths = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.input, fname))
    elif os.path.isfile(args.input):
        image_paths.append(args.input)
    
    print(f"找到 {len(image_paths)} 张图像进行推理...")

    # --- 6. 运行推理循环 ---
    for img_path in tqdm(image_paths, desc="推理中"):
        try:
            # 1. 加载和预处理
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"无法读取: {img_path}, 跳过")
                continue
                
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. 调整大小 (用于可视化)
            resized = resize_transform(image=image_rgb)
            resized_rgb = resized['image']
            
            # 3. 归一化 (用于模型)
            normalized = norm_transform(image=resized_rgb)
            input_tensor = normalized['image'].to(device).unsqueeze(0) # (1, 3, H, W)
            
            # 4. 推理
            outputs = model(input_tensor)
            
            # 5. 后处理
            results = post_process(outputs, config, device, args.threshold)
            
            # 6. 可视化
            # 我们在调整大小后的 BGR 图像上绘制
            viz_image_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
            
            if results:
                final_image = visualize(
                    viz_image_bgr,
                    results.copy(), # 确保我们不修改原始 numpy 数组
                    kpt_colors,
                    class_colors,
                    class_names
                )
            else:
                final_image = viz_image_bgr # 没有检测到, 保存原图
                
            # 7. 保存
            out_filename = os.path.basename(img_path)
            out_path = os.path.join(args.output, f"{os.path.splitext(out_filename)[0]}_pred.jpg")
            cv2.imwrite(out_path, final_image)
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
            
    print(f"\n[!! 推理完成 !!]")
    print(f"可视化结果已保存到: {args.output}")

if __name__ == "__main__":
    main()