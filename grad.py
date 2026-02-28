import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# 加载预训练的模型（ResNet50 作为例子）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # 使用 ResNet50
model.to(device).eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 128), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 用于预训练模型的标准化
])

# 加载图片
image_path = "/root/code/IDKL-main/IDKL/configs/dataset/SYSU-MM01/cam1/0001/0001.jpg"  # 这里是你上传的图片路径
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# Grad-CAM 设置：选择模型的目标层（例如最后一个卷积层）
target_layers = [model.layer4[2].conv3]  # ResNet50 的最后一个卷积层

# 实例化 Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# 设置目标类别（假设我们感兴趣的类别是 1）
target_id = 1
targets = [ClassifierOutputTarget(target_id)]

# 获取 Grad-CAM 热图
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# 选择图像大小并确保与网络输入一致
rgb_img = np.array(img.resize((256, 128))) / 255.0  # 归一化图像

# 确保 Grad-CAM 热图与原始图像的大小一致
grayscale_cam_resized = np.array(grayscale_cam)
grayscale_cam_resized = np.resize(grayscale_cam_resized, (rgb_img.shape[0], rgb_img.shape[1]))

# 显示热图叠加效果
visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(visualization)
plt.axis('off')  # 关闭坐标轴
plt.show()

# 保存结果
save_path = "/root/code/AMaP/figs/grad_cam_visualization1.jpg"
Image.fromarray(visualization).save(save_path)
print(f"Grad-CAM 图像已保存至：{save_path}")
