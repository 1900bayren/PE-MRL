# gradcam_utils.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import types

# -----------------------------
# 工具：把 numpy heatmap 叠到 PIL 图上
# -----------------------------
def visualize_cam_overlay(img_pil, mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    img_pil: PIL.Image (RGB)
    mask: HxW numpy float [0..1]
    returns: PIL.Image overlay
    """
    img = np.array(img_pil).astype(np.uint8)
    h, w = mask.shape
    if (img.shape[0] != h) or (img.shape[1] != w):
        mask = cv2.resize((mask*255).astype(np.uint8), (img.shape[1], img.shape[0]))
        mask = mask.astype(np.float32)/255.0
    else:
        mask = (mask*255).astype(np.uint8)
        mask = mask.astype(np.float32)/255.0

    heatmap = cv2.applyColorMap((mask*255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)

# -----------------------------
# Grad-CAM for CNN-style feature map (4D tensor)
# -----------------------------
class _Hook():
    def __init__(self):
        self.activations = None
        self.gradients = None
    def forward_hook(self, module, inp, out):
        self.activations = out.detach()
    def backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple (grad,)
        self.gradients = grad_out[0].detach()

def grad_cam_cnn(model, input_tensor, target_class=None, target_layer=None, device='cuda'):
    """
    model: 模型 (eval 模式)
    input_tensor: (1,C,H,W) 归一化的 tensor in device
    target_class: int (要观测的类别). 如果 None，取模型预测的 top1
    target_layer: 要挂 hook 的卷积层模块（module），例如 model.base.layer4[-1].conv3
                  （你可以传 module 对象；如果传 None，会尝试自动查找最后一个 Conv2d）
    返回: heatmap numpy float [0..1] (H, W)
    """
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    # 自动选择 target_layer（寻找最后一个 nn.Conv2d）
    if target_layer is None:
        target_layer = None
        for m in reversed(list(model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break
        if target_layer is None:
            raise RuntimeError("没找到 Conv2d 层，必须传入 target_layer 参数。")

    hook = _Hook()
    fh = target_layer.register_forward_hook(hook.forward_hook)
    bh = target_layer.register_full_backward_hook(hook.backward_hook)

    # forward
    out = model(input_tensor)  # 注意：你的 Backbones 有时返回 tuple (cls_score, feat)
    # 兼容几种输出形式
    if isinstance(out, tuple) or isinstance(out, list):
        # 如果是训练模式的返回（cls_score, global_feat）
        # 我们取第一个元素作为分类分数(可能是 tensor 或 list)
        # 尝试解包直至得到 tensor
        scores = out[0]
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
    else:
        scores = out

    if isinstance(scores, torch.Tensor) and scores.dim() == 2:
        probs = F.softmax(scores, dim=1)
        pred_cls = probs.argmax(dim=1).item()
    elif isinstance(scores, torch.Tensor) and scores.dim() == 1:
        # 单样本单分数向量
        pred_cls = int(scores.argmax().item())
    else:
        # 无法识别输出格式
        pred_cls = 0

    if target_class is None:
        target_class = pred_cls

    # 如果 scores 是二维 batch x C
    if isinstance(scores, torch.Tensor) and scores.dim() == 2:
        score_scalar = scores[0, target_class]
    else:
        # 若模型没有输出分类分数（例如只返回特征），可以用某个通道的 L2-norm 作目标
        score_scalar = (hook.activations.mean())

    model.zero_grad()
    score_scalar.backward(retain_graph=True)

    activations = hook.activations  # (B, C, H, W)
    gradients = hook.gradients      # (B, C, H, W)

    # global-average pool gradients over spatial dims
    weights = gradients.mean(dim=(2,3), keepdim=True)  # (B, C, 1, 1)
    gcam = (weights * activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
    gcam = F.relu(gcam)
    gcam = F.interpolate(gcam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    gcam = gcam.squeeze().cpu().numpy()
    gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min() + 1e-8)

    fh.remove(); bh.remove()
    return gcam

# -----------------------------
# Grad-CAM for ViT-like token sequence (B, N, D)
# 说明：此处假设 transformer.backbone 返回 token 序列，
#       cls token 在 index 0， patch tokens 在 1..N-1。
#       我们计算 patch token 的梯度重要性并 reshape 成 P_H x P_W
# -----------------------------
def grad_cam_vit(model, input_tensor, target_class=None, target_module=None,
                 patch_size=16, img_size=224, device='cuda'):
    """
    model: the whole model in eval mode, which returns classification scores or features.
    input_tensor: (1,C,H,W) normalized, on device
    target_module: the module that outputs the token sequence (eg model.base or specific block)
                   Should be the module whose forward returns (B, N, D) features.
                   If None, we try to find the first module that produces 3D/2D token outputs via a forward hook.
    patch_size: patch size of ViT (default 16). 如果你使用 patch32，请改为32。
    img_size: 输入图像分辨率（默认224）
    返回: heatmap numpy float [0..1] (H, W)
    """
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    # 安装 hook 捕获 token activations + gradients
    activations = {'feat': None}
    gradients = {'grad': None}

    def forward_hook(module, inp, out):
        # out 期望为 (B, N, D)
        activations['feat'] = out.detach()
    def backward_hook(module, grad_in, grad_out):
        gradients['grad'] = grad_out[0].detach()

    # 如果用户没传 target_module，尝试自动找到一个返回 (B,N,D) 的模块
    if target_module is None:
        candidate = None
        for m in reversed(list(model.modules())):
            # 用一个试探的 forward: 但我们不能在此执行前向。改为用户传入最可靠。
            # 所以默认尝试寻找名含 'blocks' 或 'encoder' 的模块
            name = m.__class__.__name__.lower()
            if 'block' in name or 'encoder' in name or 'transformer' in name or 'vit' in name:
                candidate = m
                break
        if candidate is None:
            # 退而求其次：使用 model.base（如果存在）
            target_module = getattr(model, 'base', None)
        else:
            target_module = candidate

    if target_module is None:
        raise RuntimeError("找不到合适的 transformer module，请传入 target_module（例如 model.base 或 model.base.blocks[-1]）")

    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)

    # forward
    out = model(input_tensor)
    # 兼容返回 (cls_score, feat) 形式
    if isinstance(out, (list, tuple)):
        scores = out[0]
        # if list of cls scores, get the first
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
    else:
        scores = out

    if isinstance(scores, torch.Tensor) and scores.dim() == 2:
        probs = F.softmax(scores, dim=1)
        pred_cls = probs.argmax(dim=1).item()
    elif isinstance(scores, torch.Tensor) and scores.dim() == 1:
        pred_cls = int(scores.argmax().item())
    else:
        pred_cls = 0

    if target_class is None:
        target_class = pred_cls

    if isinstance(scores, torch.Tensor) and scores.dim() == 2:
        score_scalar = scores[0, target_class]
    else:
        # 如果模型没有分类头，取 token norm 的均值
        score_scalar = activations['feat'].mean()

    model.zero_grad()
    score_scalar.backward(retain_graph=True)

    token_feats = activations['feat']  # (B, N, D)
    token_grads = gradients['grad']    # (B, N, D)

    # 排除 cls token（index 0），我们关注 patch tokens 1..N-1
    patch_feats = token_feats[:, 1:, :]   # (B, P, D)
    patch_grads = token_grads[:, 1:, :]   # (B, P, D)

    # 对每个 patch 计算重要性权重：例如对 feature dim 做全局平均
    weights = patch_grads.mean(dim=2)  # (B, P)
    # 或者可用 L2 norm: weights = patch_grads.norm(dim=2)

    # 权重与 patch feature 做点积（按 patch）
    cam_per_patch = (weights * patch_feats.abs().mean(dim=2)).squeeze(0).cpu().numpy()  # (P,)

    # 归一化
    cam_per_patch = cam_per_patch - cam_per_patch.min()
    cam_per_patch = cam_per_patch / (cam_per_patch.max() + 1e-8)

    # reshape 到 patch 网格
    num_patches = cam_per_patch.shape[0]
    # 计算 patch 网格尺寸 (假设正方形)
    pside = int(np.sqrt(num_patches))
    if pside * pside != num_patches:
        # fallback: try to infer from img_size and patch_size
        pside = img_size // patch_size
    cam_grid = cam_per_patch.reshape(pside, pside)

    # 上采样到原图大小
    cam_grid = cv2.resize(cam_grid.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    cam_grid = (cam_grid - cam_grid.min()) / (cam_grid.max() - cam_grid.min() + 1e-8)

    fh.remove(); bh.remove()
    return cam_grid

# -----------------------------
# 预处理 helper（与你训练时用的 normalize 保持一致）
# -----------------------------
def make_preprocess(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

# -----------------------------
# 示例：完整运行流程
# -----------------------------
def example_run_cnn(model, image_path, device='cuda', target_layer=None, img_size=224):
    """
    model: your ResNet-based model (Backbone)，eval 模式
    """
    preprocess = make_preprocess(img_size=img_size)
    img = Image.open(image_path).convert('RGB')
    inp = preprocess(img).unsqueeze(0)
    gcam = grad_cam_cnn(model, inp, target_class=None, target_layer=target_layer, device=device)
    overlay = visualize_cam_overlay(img, gcam)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img); plt.axis('off'); plt.title('input')
    plt.subplot(1,2,2); plt.imshow(overlay); plt.axis('off'); plt.title('Grad-CAM')
    plt.show()
    return gcam, overlay

def example_run_vit(model, image_path, device='cuda', target_module=None, patch_size=16, img_size=224):
    preprocess = make_preprocess(img_size=img_size)
    img = Image.open(image_path).convert('RGB')
    inp = preprocess(img).unsqueeze(0)
    gcam = grad_cam_vit(model, inp, target_class=None, target_module=target_module,
                       patch_size=patch_size, img_size=img_size, device=device)
    overlay = visualize_cam_overlay(img, gcam)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img); plt.axis('off'); plt.title('input')
    plt.subplot(1,2,2); plt.imshow(overlay); plt.axis('off'); plt.title('ViT Grad-CAM (patch)')
    plt.savefig('/root/code/AMaP/figs/cam_exp1')
    return gcam, overlay
