import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from model import make_model
from config import cfg
from datasets import make_dataloader
from processor import do_inference
from utils.logger import setup_logger
from datetime import datetime
import argparse

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载配置文件
cfg.merge_from_file('/root/code/AMaP/configs/RegDB/vit_transreid_stride.yml')  # 这里可以指定你的配置文件路径
cfg.freeze()

# 数据加载
train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

# 初始化模型
model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
model.to(device).eval()

# 加载模型权重
checkpoint_path = "/root/code/AMaP/logs/regdb_WHAT/90.10_83.88_90.73_84.80/transformer_best_69.pth" # 权重路径
model.load_state_dict(torch.load(checkpoint_path))

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 128), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 选择一张图像
image_path = "/root/code/IDKL-main/IDKL/configs/dataset/RegDB/Thermal/1/male_front_t_00007_1.bmp"
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# 检查输入张量的形状
print(f"Original Input shape: {input_tensor.shape}")  # 输出输入的形状

# 确保输入张量的形状为四维 (B, C, H, W)
B, C, H, W = input_tensor.shape
if len(input_tensor.shape) == 3:  # 如果是三维 (B, N, C)，需要调整为四维
    input_tensor = input_tensor.unsqueeze(0)  # 添加一个批量维度，使其变为 (B, C, H, W)
    print(f"Adjusted Input shape: {input_tensor.shape}")  # 输出调整后的形状

# Grad-CAM 设置：选择 Transformer 的最后一层作为目标层
target_layers = [model.base.blocks[-1]]  # 选择最后一层 Transformer block

# 实例化 Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# 设置目标类别（选择正确的类别 ID，例如 1）
target_id = 1
targets = [ClassifierOutputTarget(target_id)]

# 获取 Grad-CAM 热图
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# 显示热图叠加效果
rgb_img = np.array(img.resize((256, 128))) / 255.0  # 将图像调整到 128x384，确保与网络输入一致
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 保存结果
save_path = "/root/code/AMaP/figs.jpg"
Image.fromarray(visualization).save(save_path)
print(f"Grad-CAM 图像已保存至：{save_path}")
