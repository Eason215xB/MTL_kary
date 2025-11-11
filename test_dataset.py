#!/usr/bin/env python3
"""
测试数据集加载，检查关键点数据处理
"""

import sys
import os
sys.path.insert(0, '/home/yxb/project/MTL_kary')

import json
import torch
import numpy as np
from MTL_datasets import CocoMtlDataset

def test_dataset_loading():
    """测试数据集加载"""
    print("🔍 测试数据集加载...")
    
    # 加载配置
    with open('/home/yxb/project/MTL_kary/config.json', 'r') as f:
        config = json.load(f)
    
    try:
        # 创建数据集（只加载1个样本进行测试）
        dataset = CocoMtlDataset(
            ann_file=config['DATA_PATHS']['TRAIN_ANN_FILE'],
            img_dir=config['DATA_PATHS']['TRAIN_IMG_DIR'],
            img_size=tuple(config['IMG_SIZE']),
            simcc_ratio=config['MODEL_PARAMS']['POSE_SIMCC_RATIO'],
            transforms=None,
            use_train_subset=1  # 只加载1张图像
        )
        
        print(f"✅ 数据集创建成功")
        print(f"  - 图像数量: {len(dataset)}")
        print(f"  - 检测到的关键点数量: {dataset.num_kpts}")
        
        # 尝试加载第一个样本
        print("\n🔍 测试样本加载...")
        img, targets = dataset[0]
        
        print(f"✅ 样本加载成功")
        print(f"  - 图像形状: {img.shape if hasattr(img, 'shape') else type(img)}")
        print(f"  - 目标键: {list(targets.keys())}")
        
        # 检查关键点相关的目标
        if 'pose_targets_x' in targets:
            pose_x_shape = targets['pose_targets_x'].shape
            pose_y_shape = targets['pose_targets_y'].shape
            pose_w_shape = targets['pose_weights'].shape
            
            print(f"  - pose_targets_x 形状: {pose_x_shape}")
            print(f"  - pose_targets_y 形状: {pose_y_shape}")
            print(f"  - pose_weights 形状: {pose_w_shape}")
            
            # 检查形状是否合理
            expected_kpts = config['MODEL_PARAMS']['POSE_NUM_KEYPOINTS']
            if pose_w_shape[1] != expected_kpts:
                print(f"⚠️ 关键点数量不匹配: 数据={pose_w_shape[1]}, 配置={expected_kpts}")
                return False
            else:
                print(f"✅ 关键点数量匹配: {expected_kpts}")
        
        # 检查分割相关的目标
        if 'masks' in targets:
            masks_shape = targets['masks'].shape
            labels_shape = targets['labels'].shape
            print(f"  - masks 形状: {masks_shape}")
            print(f"  - labels 形状: {labels_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试数据集加载...")
    print("=" * 60)
    
    success = test_dataset_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 数据集测试通过！")
        print("\n📝 建议:")
        print("  - 现在可以尝试重新训练")
        print("  - 监控训练过程中是否还有错误")
    else:
        print("❌ 数据集测试失败，需要进一步调试")
    
    return success

if __name__ == "__main__":
    main()
