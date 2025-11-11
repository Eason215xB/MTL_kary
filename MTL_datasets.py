# mtl_dataset.py
# [!! V9: 修复了 albumentations 关键点格式 (坐标错乱) !!]
# [!! V8: 修复了导致损失爆炸的 A.Normalize !!]

import torch
import torch.utils.data
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from typing import List, Dict, Tuple, Any
import traceback # 导入 traceback

# ==================================
# 1. 目标生成 (SimCC) - (不变)
# ==================================
def _generate_simcc_targets(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    img_size: Tuple[int, int],
    simcc_ratio: float,
    sigma: float = 6.0,
    kpt_dim: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ... (此函数 100% 正确, 保持不变) ...
    H, W = img_size
    N_inst, N_kpts = keypoints.shape[:2]
    W_simcc = int(W * simcc_ratio)
    H_simcc = int(H * simcc_ratio)
    target_x = np.zeros((N_inst, N_kpts, W_simcc), dtype=np.float32)
    target_y = np.zeros((N_inst, N_kpts, H_simcc), dtype=np.float32)
    keypoint_weights = visibility.astype(np.float32)
    if N_inst == 0:
        return target_x, target_y, keypoint_weights
    for n in range(N_inst):
        for k in range(N_kpts):
            if visibility[n, k] > 0:
                x, y = keypoints[n, k, :2]
                mu_x = int(x * simcc_ratio)
                mu_y = int(y * simcc_ratio)
                if not (0 <= mu_x < W_simcc) or not (0 <= mu_y < H_simcc):
                    keypoint_weights[n, k] = 0
                    continue
                x_range = np.arange(W_simcc, dtype=np.float32)
                y_range = np.arange(H_simcc, dtype=np.float32)
                gauss_x = np.exp(-((x_range - mu_x) ** 2) / (2 * sigma ** 2))
                gauss_y = np.exp(-((y_range - mu_y) ** 2) / (2 * sigma ** 2))
                target_x[n, k, :] = gauss_x
                target_y[n, k, :] = gauss_y
    return target_x, target_y, keypoint_weights


# ==================================
# 2. 数据集类 (修改)
# ==================================

class CocoMtlDataset(torch.utils.data.Dataset):
    """
    COCO 多任务 (分割+姿态) 数据集
    (加载逻辑已参考 BaseCocoStyleDataset 重构)
    """
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        simcc_ratio: float = 2.0,
        simcc_sigma: float = 6.0,
        transforms: Any = None,
        use_train_subset: int = 0,
        debug_vis_dir: str = None 
    ):
        super().__init__()
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"COCO 标注文件未找到: {ann_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像目录未找到: {img_dir}")
            
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_size = img_size
        self.simcc_ratio = simcc_ratio
        self.simcc_sigma = simcc_sigma
        
        self.debug_vis_dir = debug_vis_dir
        if self.debug_vis_dir:
            print(f"[调试] 可视化已启用, 将保存到: {self.debug_vis_dir}")
            if not os.path.exists(self.debug_vis_dir):
                os.makedirs(self.debug_vis_dir, exist_ok=True)
        
        print("正在加载标注文件到内存...")
        self.coco = COCO(ann_file)
        
        self.num_kpts = self._get_num_kpts()
        
        print("正在解析数据列表...")
        self.data_infos = self.load_data_list(use_train_subset)
            
        print(f"数据加载完成: {len(self.data_infos)} 张有效图像, {self.num_kpts} 个关键点")

    def _get_num_kpts(self) -> int:
        """从 COCO 标注中推断关键点数量"""
        cat_ids = self.coco.getCatIds()
        if not cat_ids:
            print("[警告] 标注文件中没有 'categories'。")
            return 3 

        cat_info_list = self.coco.loadCats(cat_ids[0])
        if cat_info_list:
            cat_info = cat_info_list[0]
            if 'keypoints' in cat_info and cat_info['keypoints']:
                detected_kpts = len(cat_info['keypoints'])
                return detected_kpts
        
        img_ids = list(sorted(self.coco.imgs.keys()))
        if not img_ids: raise ValueError("标注文件中没有 'images'。")
        
        for img_id in img_ids[:10]:
            sample_ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if sample_ann_ids:
                sample_ann = self.coco.loadAnns(sample_ann_ids[0])[0]
                if 'keypoints' in sample_ann and sample_ann['keypoints']:
                    detected_kpts = len(sample_ann['keypoints']) // 3
                    if detected_kpts > 0:
                        print(f"[调试] 从标注推断到 {detected_kpts} 个关键点")
                        return detected_kpts

        raise ValueError("无法从 'categories' 或 'annotations' 推断关键点数量。")


    def load_data_list(self, use_train_subset: int) -> List[Dict]:
        """[!! V9 修复 !!] 修正 gt_keypoints 的格式"""
        img_ids = list(sorted(self.coco.imgs.keys()))
        data_infos = []

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            if not anns: continue

            gt_labels = []
            gt_bboxes = []
            gt_keypoints = [] # [!! V9 !!] 格式: List[ List[Tuple[x,y,v]] ]
            raw_anns = []     

            for ann in anns:
                if 'keypoints' not in ann or len(ann['keypoints']) != self.num_kpts * 3:
                    continue 

                gt_labels.append(ann['category_id'])
                x, y, w, h = ann['bbox']
                gt_bboxes.append([x, y, x + w, y + h])
                
                kpts_flat = ann['keypoints']
                kpts_xyz = np.array(kpts_flat).reshape(-1, 3)
                
                # [!! V9 修复 !!]
                # 分别存储坐标和可见性
                inst_kpts_xy = []  # 只存储 (x, y) 给 Albumentations
                inst_kpts_vis = [] # 单独存储 visibility
                for kpt in kpts_xyz:
                    inst_kpts_xy.append((kpt[0], kpt[1]))  # 只有 (x, y)
                    # [!! 修复可见性值 !!] COCO: 0=不可见, 1=被遮挡, 2=可见 -> 转换为 0/1
                    vis_val = 1.0 if kpt[2] > 0 else 0.0
                    inst_kpts_vis.append(vis_val)          # 标准化的 visibility
                
                gt_keypoints.append({
                    'xy': inst_kpts_xy,      # List[Tuple[x,y]]
                    'visibility': inst_kpts_vis  # List[visibility]
                })
                raw_anns.append(ann)

            if not gt_labels:
                continue

            data_info = {
                'img_id': img_id,
                'img_path': img_path,
                'height': img_info['height'],
                'width': img_info['width'],
                'gt_labels': gt_labels,
                'gt_bboxes': gt_bboxes,
                'gt_keypoints': gt_keypoints, # [!! V9 !!] 现在的格式是 List[List[Tuple]]
                'raw_anns': raw_anns 
            }
            data_infos.append(data_info)

        if use_train_subset > 0:
            print(f"[调试] 仅使用 {use_train_subset} 张图像")
            data_infos = data_infos[:use_train_subset]
            
        return data_infos


    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # [!! 防止无限递归 !!]
        if _retry_count > 10:
            print(f"[!! 警告 !!] 递归深度超过10，返回None避免无限循环")
            return None
        
        data_info = self.data_infos[idx]
        
        try:
            # 1. 加载图像 (I/O)
            image = cv2.imread(data_info['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 2. 从预解析数据中获取标签
            gt_labels = data_info['gt_labels']
            gt_bboxes = data_info['gt_bboxes']
            # [!! V9 !!] gt_keypoints_data: List[Dict{'xy': List[Tuple], 'visibility': List}]
            gt_keypoints_data = data_info['gt_keypoints'] 
            raw_anns = data_info['raw_anns']
            
            # 3. 生成 Mask (按需)
            gt_masks = [self.coco.annToMask(ann) for ann in raw_anns]

            # 4. [!! 可视化 (增强前) !!]
            if self.debug_vis_dir and idx < 10: 
                # [!! V9 修复 !!] 重构 (x,y,v) 格式用于可视化
                parsed_kpts_for_vis = []
                for inst_data in gt_keypoints_data:
                    inst_kpts_array = []
                    for (x, y), v in zip(inst_data['xy'], inst_data['visibility']):
                        inst_kpts_array.append([x, y, v])
                    parsed_kpts_for_vis.append(np.array(inst_kpts_array))
                
                visualize_and_save_pre_aug(
                    image, 
                    data_info,
                    gt_masks,
                    gt_bboxes,
                    parsed_kpts_for_vis, # 传入 List[np.array(K, 3)]
                    save_dir=self.debug_vis_dir
                )

            # 5. 应用 Augmentations (Albumentations)
            transformed_image = image
            transformed_masks = gt_masks
            transformed_kpts_data = gt_keypoints_data # [!! V9 !!]
            transformed_labels = gt_labels
            transformed_bboxes = gt_bboxes

            if self.transforms:
                # [!! V9 修复 !!] 正确处理关键点增强
                # 提取所有关键点的 (x, y) 坐标，单独保存 visibility
                all_keypoints_xy = []
                all_visibility = []
                
                for inst_data in gt_keypoints_data:
                    for kpt_xy, kpt_vis in zip(inst_data['xy'], inst_data['visibility']):
                        all_keypoints_xy.append(kpt_xy)  # 只传递 (x, y)
                        all_visibility.append(kpt_vis)   # 单独保存 visibility
                
                transformed = self.transforms(
                    image=image,
                    masks=gt_masks,
                    keypoints=all_keypoints_xy,  # [!! V9 !!] 只传递 (x,y) 坐标
                    labels=gt_labels,
                    bboxes=gt_bboxes
                )
                transformed_image = transformed['image']
                transformed_masks = transformed['masks']
                transformed_keypoints_xy = transformed['keypoints']  # 变换后的 (x,y)
                transformed_labels = transformed['labels']
                transformed_bboxes = transformed['bboxes']
                
                # 重新组织关键点数据，将 visibility 添加回去
                transformed_kpts_data = []
                kpt_idx = 0
                n_kpts = self.num_kpts
                
                for inst_idx in range(len(transformed_labels)):
                    if kpt_idx + n_kpts <= len(transformed_keypoints_xy):
                        inst_xy = transformed_keypoints_xy[kpt_idx:kpt_idx + n_kpts]
                        inst_vis = all_visibility[kpt_idx:kpt_idx + n_kpts]  # visibility不变
                        transformed_kpts_data.append({
                            'xy': inst_xy,
                            'visibility': inst_vis
                        })
                        kpt_idx += n_kpts
                    else:
                        break
            else:
                transformed_kpts_data = gt_keypoints_data
                
            if not transformed_labels:
                print(f"[调试] 图像 {data_info['img_id']} 增强后所有实例丢失, 跳过。")
                return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

            # 6. 后处理和格式化
            # [!! 修复 !!] 确保masks是tensor格式
            if isinstance(transformed_masks[0], np.ndarray):
                gt_masks_tensor = torch.stack([torch.from_numpy(mask) for mask in transformed_masks], dim=0).long()
            else:
                gt_masks_tensor = torch.stack(transformed_masks, dim=0).long()
            gt_labels_tensor = torch.tensor(transformed_labels, dtype=torch.long)
            
            gt_bboxes_np = np.array(transformed_bboxes, dtype=np.float32)
            if gt_bboxes_np.shape[0] > 0:
                gt_bboxes_np[:, 2] = np.maximum(gt_bboxes_np[:, 2], gt_bboxes_np[:, 0] + 1)
                gt_bboxes_np[:, 3] = np.maximum(gt_bboxes_np[:, 3], gt_bboxes_np[:, 1] + 1)
            gt_bboxes_tensor = torch.from_numpy(gt_bboxes_np)
            
            N_inst = len(transformed_labels)
            N_kpts = self.num_kpts
            
            # [!! V9 修复 !!]
            # transformed_kpts_data 是 List[Dict{'xy': List[Tuple], 'visibility': List}]
            # 我们需要将其转换为 (N_inst, N_kpts, 3) 的 np.array
            
            try:
                gt_kpts_list = []
                for inst_data in transformed_kpts_data:
                    inst_array = []
                    for (x, y), v in zip(inst_data['xy'], inst_data['visibility']):
                        inst_array.append([x, y, v])
                    gt_kpts_list.append(inst_array)
                
                gt_kpts_xys_np = np.array(gt_kpts_list, dtype=np.float32).reshape(N_inst, N_kpts, 3)
            
                # [!! 调试信息 !!]
                # if idx < 3:  # 只对前3个样本打印调试信息
                #     print(f"[调试-数据集] 图像 {data_info['img_id']}: N_inst={N_inst}, N_kpts={N_kpts}")
                #     print(f"[调试-数据集] gt_kpts_xys_np.shape: {gt_kpts_xys_np.shape}")
                #     print(f"[调试-数据集] 关键点坐标范围: x=[{gt_kpts_xys_np[:,:,0].min():.1f}, {gt_kpts_xys_np[:,:,0].max():.1f}], y=[{gt_kpts_xys_np[:,:,1].min():.1f}, {gt_kpts_xys_np[:,:,1].max():.1f}]")
                #     print(f"[调试-数据集] 可见性: {gt_kpts_xys_np[:,:,2].sum():.0f}/{gt_kpts_xys_np.shape[0] * gt_kpts_xys_np.shape[1]} 个可见关键点")
                #     print(f"[调试-数据集] 可见性详细: {gt_kpts_xys_np[:,:,2].flatten()}")
                
            except (ValueError, TypeError) as e:
                print(f"[!! 错误 !!] 图像 {data_info['img_id']} 关键点 Reshape 失败: {e}")
                print(f"  N_inst={N_inst}, N_kpts={N_kpts}")
                print(f"  变换后的关键点数据: {transformed_kpts_data}")
                return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

            gt_kpts_np = gt_kpts_xys_np[:, :, :2] 
            gt_kpts_vis_np = gt_kpts_xys_np[:, :, 2]
            
            target_x, target_y, pose_weights = _generate_simcc_targets(
                gt_kpts_np,
                gt_kpts_vis_np,
                self.img_size,
                self.simcc_ratio,
                self.simcc_sigma
            )
            
            # [!! 调试信息 !!]
            # if idx < 3:  # 只对前3个样本打印调试信息
            #     print(f"[调试-SimCC] target_x.shape: {target_x.shape}, target_y.shape: {target_y.shape}")
            #     print(f"[调试-SimCC] pose_weights.shape: {pose_weights.shape}, 可见权重: {pose_weights.sum():.0f}")
            #     print(f"[调试-SimCC] pose_weights详细: {pose_weights.flatten()}")
            #     print("="*50)
            
            # 7. 组合成字典
            targets = {
                'masks': gt_masks_tensor,
                'labels': gt_labels_tensor,
                'bboxes': gt_bboxes_tensor,
                'pose_targets_x': torch.from_numpy(target_x),
                'pose_targets_y': torch.from_numpy(target_y),
                'pose_weights': torch.from_numpy(pose_weights),
                
                'pose_kpts_aug': torch.from_numpy(gt_kpts_xys_np), 
                'img_path': data_info['img_path'] 
            }
            return transformed_image, targets

        except Exception as e:
            print(f"!! [错误] 处理图像 {data_info.get('img_id', 'N/A')} (索引 {idx}) 失败: {e}")
            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)


# ==================================
# 3. Collate Fn (修改)
# ==================================

def mtl_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) \
    -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    自定义 Collate_fn 用于多任务训练（不返回路径）。
    """
    images = []
    targets_list = []
    
    for item in batch:
        if item is None:
            continue # 跳过损坏的数据
        img, targets = item
        images.append(img)
        
        # 移除img_path（训练时不需要）
        if 'img_path' in targets:
            targets.pop('img_path')
            
        targets_list.append(targets)
    
    if not images:
        return torch.tensor([]), []

    images = torch.stack(images, dim=0)
    return images, targets_list


def mtl_collate_fn_with_paths(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) \
    -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], List[str]]:
    """
    自定义 Collate_fn 用于多任务测试/可视化（返回路径）。
    """
    images = []
    targets_list = []
    img_paths = [] 
    
    for item in batch:
        if item is None:
            continue # 跳过损坏的数据
        img, targets = item
        images.append(img)
        img_paths.append(targets.pop('img_path'))
        targets_list.append(targets)
    
    if not images:
        return torch.tensor([]), [], []

    images = torch.stack(images, dim=0)
    return images, targets_list, img_paths


# ==================================
# 4. 可视化函数 (增强前) - (不变)
# ==================================

KPT_COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255)  # Magenta
]

def visualize_and_save_pre_aug(
    image_np: np.ndarray, # (H, W, C) RGB
    data_info: Dict,
    gt_masks: List[np.ndarray], # List[(H, W)]
    gt_bboxes: List[List[float]], # List[[x1,y1,x2,y2]]
    gt_keypoints: List[np.ndarray], # List[(K, 3)]
    save_dir: str
):
    try:
        vis_img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        overlay = np.zeros_like(vis_img_bgr, dtype=np.uint8)
        gt_labels = data_info['gt_labels']
        N_inst = len(gt_labels)
        
        instance_colors = [tuple(np.random.randint(50, 256, 3).astype(int)) for _ in range(N_inst)]
        
        for i in range(N_inst):
            mask = gt_masks[i]
            color = instance_colors[i]
            overlay[mask > 0] = color
            
        vis_img_bgr = cv2.addWeighted(vis_img_bgr, 0.6, overlay, 0.4, 0)

        for i in range(N_inst):
            x1, y1, x2, y2 = np.array(gt_bboxes[i]).astype(int)
            color = (int(instance_colors[i][0]), int(instance_colors[i][1]), int(instance_colors[i][2]))
            cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), color, 2)
            
            inst_kpts = gt_keypoints[i]
            for k_idx in range(inst_kpts.shape[0]):
                x, y, v = inst_kpts[k_idx]
                if v > 0:
                    color = KPT_COLORS[k_idx % len(KPT_COLORS)]
                    cv2.circle(vis_img_bgr, (int(x), int(y)), 5, color, -1)
        
        img_name = os.path.basename(data_info['img_path']).split('.')[0]
        img_id = data_info['img_id']
        save_path = os.path.join(save_dir, f"{img_name}_id{img_id}_pre_aug.png")
        cv2.imwrite(save_path, vis_img_bgr)
    except Exception as e:
        print(f"[可视化错误] 无法保存 {data_info.get('img_path', 'N/A')}: {e}")


# ==================================
# 5. 可视化函数 (增强后) - (不变)
# ==================================

def unnormalize_image(tensor_img: torch.Tensor, 
                      mean=np.array([0.485, 0.456, 0.406]), 
                      std=np.array([0.229, 0.224, 0.225])) -> np.ndarray:
    """反归一化 (适用于 ImageNet 均值/标准差)"""
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std) + mean # 反归一化
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img_np, dtype=np.uint8)

def visualize_and_save_post_aug(
    images_tensor: torch.Tensor, 
    targets_list: List[Dict[str, torch.Tensor]], 
    img_paths: List[str],
    config: Dict, 
    save_dir: str = "debug_visuals",
    batch_idx: int = 0
):
    B = images_tensor.shape[0]
    
    mean = np.array(config['NORM_MEAN'])
    std = np.array(config['NORM_STD'])
    n_kpts = 3
    
    for b_idx in range(B):
        try:
            img_np = unnormalize_image(images_tensor[b_idx], mean, std)
            vis_img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            targets = targets_list[b_idx]
            
            overlay = np.zeros_like(vis_img_bgr, dtype=np.uint8)
            masks = targets['masks'].cpu().numpy()
            N_inst = masks.shape[0]
            if N_inst == 0: continue

            instance_colors = [tuple(np.random.randint(50, 256, 3).astype(int)) for _ in range(N_inst)]
            
            for i in range(N_inst):
                mask = masks[i]
                color = instance_colors[i]
                overlay[mask > 0] = color
                
            vis_img_bgr = cv2.addWeighted(vis_img_bgr, 0.6, overlay, 0.4, 0)

            bboxes = targets['bboxes'].cpu().numpy()
            kpts = targets['pose_kpts_aug'].cpu().numpy()
            
            for i in range(N_inst):
                x1, y1, x2, y2 = bboxes[i].astype(int)
                color = (int(instance_colors[i][0]), int(instance_colors[i][1]), int(instance_colors[i][2]))
                cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), color, 2)
                
                inst_kpts = kpts[i]
                for k_idx in range(inst_kpts.shape[0]):
                    x, y, v = inst_kpts[k_idx]
                    if v > 0:
                        color = KPT_COLORS[k_idx % len(KPT_COLORS)]
                        cv2.circle(vis_img_bgr, (int(x), int(y)), 5, color, -1)
            
            img_name = os.path.basename(img_paths[b_idx]).split('.')[0]
            # 添加batch和sample索引以避免重复
            save_path = os.path.join(save_dir, f"{img_name}_batch{batch_idx}_sample{b_idx}_post_aug.png")
            cv2.imwrite(save_path, vis_img_bgr)
        
        except Exception as e:
            print(f"[可视化错误] 无法保存 post-aug {img_paths[b_idx]}: {e}")
            
    # [!! V9 修复 !!] 检查 DDP rank
    try:
        rank = torch.distributed.get_rank()
    except Exception:
        rank = 0
        
    if rank == 0: 
        print(f"  [可视化] 已保存 {B} 张 post-aug 图像到 {save_dir}")


# ==================================
# 6. 演示/测试 (V9 修复)
# ==================================

if __name__ == "__main__":
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import time
    import json
    import os
    from tqdm import tqdm
    
    # [!! V8 修复 !!] 必须加载 config.json 以获取 NORM_MEAN/STD
    try:
        with open("config.json", 'r') as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        print("错误: 演示需要 'config.json' 文件。")
        # 使用你之前的默认值
        CONFIG = {
            'NORM_MEAN': [0.0, 0.0, 0.0],
            'NORM_STD': [1.0, 1.0, 1.0],
            'IMG_SIZE': [224, 224],
            'DATA_PATHS': {
                'TRAIN_ANN_FILE': "train.json", # 占位符
                'TRAIN_IMG_DIR': "test_images", # 占位符
            },
            'MODEL_PARAMS': {
                'POSE_NUM_KEYPOINTS': 3,
                'POSE_SIMCC_RATIO': 2.0
            }
        }
    
    ANN_FILE = CONFIG['DATA_PATHS']['TRAIN_ANN_FILE']
    IMG_DIR = CONFIG['DATA_PATHS']['TRAIN_IMG_DIR']
    
    if not (os.path.exists(ANN_FILE) and os.path.exists(IMG_DIR)):
        print("="*50); print("!! 警告: 演示路径无效 !!"); print("="*50); exit()

    IMG_H, IMG_W = CONFIG['IMG_SIZE']
    SIMCC_RATIO = CONFIG['MODEL_PARAMS']['POSE_SIMCC_RATIO']
    
    SAVE_DIR = "debug_visuals_mtl"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[可视化] 创建目录: {SAVE_DIR}")
    else:
        print(f"[可视化] 将清空并使用现有目录: {SAVE_DIR}")
        for f in os.listdir(SAVE_DIR):
            try: os.remove(os.path.join(SAVE_DIR, f))
            except Exception: pass

    # ---------------------------------------------
    # 2. 定义 Augmentation (V9 修复)
    # ---------------------------------------------
    # transforms = A.Compose([
    #     A.HorizontalFlip(p=0.3),
    #     A.VerticalFlip(p=0.3),
    #     A.DiagonalFlip(p=0.2),
    #     A.Resize(height=224, width=224, interpolation=cv2.INTER_LINEAR),
    #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    #     A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), 
    #     ToTensorV2(),
    # ], 
        transforms = A.Compose([
        # 1. 核心几何变换（支持关键点和边界框）
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Resize(height=IMG_H, width=IMG_W, interpolation=cv2.INTER_LINEAR),
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3,
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        
        # 2. 颜色和对比度增强（不影响关键点和边界框）
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),

        # 3. 模糊增强（不影响关键点和边界框）
        A.OneOf([
            A.Blur(blur_limit=(1, 3), p=1.0),
            A.MedianBlur(blur_limit=(1, 3), p=1.0),
            A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        ], p=0.2),

        # 4. 噪声增强（不影响关键点和边界框）
        A.OneOf([
            A.GaussNoise(var_limit=(5, 15), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.15), p=1.0),
        ], p=0.25),
        
        # 5. 归一化（必须在最后）
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), 
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

    # ---------------------------------------------
    # 3. 创建 Dataset 和 DataLoader
    # ---------------------------------------------
    print("正在初始化数据集...")
    dataset = CocoMtlDataset(
        ann_file=ANN_FILE,
        img_dir=IMG_DIR,
        img_size=(IMG_H, IMG_W),
        simcc_ratio=SIMCC_RATIO,
        transforms=transforms,
        use_train_subset=10, 
        debug_vis_dir=SAVE_DIR
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4, # 修改为更小的批次
        shuffle=False, # 关闭shuffle以便调试
        num_workers=0, 
        collate_fn=mtl_collate_fn_with_paths  # [!! 修复 !!] 测试时使用带路径的版本
    )

    # ---------------------------------------------
    # 4. 迭代测试
    # ---------------------------------------------
    print("开始迭代测试...")
    start_time = time.time()
    
    try:
        # [!! V8 修复 !!] 捕获 img_paths
        for i, (images, targets_list, img_paths) in enumerate(tqdm(dataloader, desc="测试中")):

            if i >= 1: break # 只测试 1 个 batch (B=2)
            
            if not images.numel():
                print("!! [警告] Dataloader 返回了空批次, 跳过")
                continue

            print(f"\n--- Batch {i+1} ---")
            
            B = images.shape[0]
            print(f"  Batch Size (B): {B}")
            print(f"  Image shape: {images.shape}")
            assert images.shape == (B, 3, IMG_H, IMG_W)
            
            assert isinstance(targets_list, list)
            assert len(targets_list) == B
            
            print(f"  Targets: 是一个长度为 {B} 的 List[Dict]")
            
            # [!! V8 修复 !!]
            visualize_and_save_post_aug(
                images, 
                targets_list, 
                img_paths, 
                config=CONFIG, # 传入 config
                save_dir=SAVE_DIR, 
                batch_idx=i
            )
            
            # (断言检查)
            for b_idx in range(B):
                targets = targets_list[b_idx]
                N_kpts = targets['pose_kpts_aug'].shape[1]
                W_simcc = int(IMG_W * SIMCC_RATIO)
                N_inst = targets['labels'].shape[0]
                assert targets['masks'].shape == (N_inst, IMG_H, IMG_W)
                assert targets['pose_targets_x'].shape == (N_inst, N_kpts, W_simcc)

    except Exception as e:
        print("\n!! 测试失败 !!")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        
    end_time = time.time()
    print(f"\n测试完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"请检查 '{SAVE_DIR}' 文件夹中的可视化结果。")