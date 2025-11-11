# create_dummy_image.py
import numpy as np
import cv2
import os

# 目标图像路径 (与 mtl_dataset.py 测试中的路径一致)
IMG_DIR = "test_images"
IMG_NAME = "test_image.jpg"
IMG_PATH = os.path.join(IMG_DIR, IMG_NAME)

# 图像尺寸 (与 test_annotations.json 中的定义一致)
H, W = 1080, 1920

# 确保目录存在
os.makedirs(IMG_DIR, exist_ok=True)

# 创建一个 (H, W, 3) 的空白 (黑色) 图像
# np.uint8 是 0-255 的图像格式
dummy_image = np.zeros((H, W, 3), dtype=np.uint8)

# 使用 cv2 将这个 numpy 数组保存为 .jpg 文件
try:
    cv2.imwrite(IMG_PATH, dummy_image)
    print(f"成功创建空白测试图像: {IMG_PATH}")
except Exception as e:
    print(f"创建图像失败: {e}")
    print("请确保你已安装 'opencv-python' (pip install opencv-python-headless)")