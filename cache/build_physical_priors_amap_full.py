#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMaP风格的物理先验离线生成器（.npy），镜像原图目录结构，文件名只改后缀为 .npy

IR（红外）包含：
  - ir.lowfreq_hist (16)     低频亮度/热分布径向能量（FFT环带）
  - ir.lowfreq_stats (3)     低频亮度 mean/var/energy（高斯平滑）
  - ir.edge_mag_hist (8)     Sobel梯度强度直方图
  - ir.edge_ori_hist (8)     边缘方向直方图（-pi..pi）
  - ir.edge_density (1)      边缘密度（强度 > 阈值 的占比）
  - ir.saliency_hist (16)    显著性图直方图（谱残差）
  - ir.saliency_energy (1)   显著性能量

VIS（可见光）包含：
  - vis.hsv_meanvar (6)      HSV三通道 mean/var
  - vis.h_hist (16)          H通道直方图（环形）
  - vis.grayworld_dev (1)    灰世界偏差
  - vis.gabor_energy (12)    Gabor能量（4方向×3尺度）
  - vis.saliency_hist (16)   显著性图直方图（谱残差）
  - vis.saliency_energy (1)  显著性能量

依赖：numpy、PIL（不依赖scipy/opencv，速度够用；可多进程）
"""
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_start_method, set_start_method
import numpy as np
from PIL import Image

# ----------------------------
# 基础工具
# ----------------------------
def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

def to_gray01(img_pil: Image.Image) -> np.ndarray:
    # [H,W], float32 in [0,1]
    return np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0

def box_filter2d(x: np.ndarray, k: int = 3) -> np.ndarray:
    assert k % 2 == 1
    pad = k // 2
    xpad = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(x)
    # 朴素的均值滤波
    for i in range(k):
        for j in range(k):
            out += xpad[i:i+x.shape[0], j:j+x.shape[1]]
    out /= (k*k)
    return out

def gaussian_kernel(size=21, sigma=5.0):
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx**2 + yy**2) / (2.0*sigma*sigma))
    ker /= ker.sum()
    return ker.astype(np.float32)

def fft_conv2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    # x,k: [H,W], real
    H, W = x.shape
    Kh, Kw = k.shape
    pad_h, pad_w = H + Kh - 1, W + Kw - 1
    X = np.fft.rfft2(x, s=(pad_h, pad_w))
    K = np.fft.rfft2(k, s=(pad_h, pad_w))
    Y = X * K
    y = np.fft.irfft2(Y, s=(pad_h, pad_w))
    # center crop
    sh, sw = (Kh - 1)//2, (Kw - 1)//2
    return y[sh:sh+H, sw:sw+W].astype(np.float32)

def hist_norm(vals: np.ndarray, bins: int, vmin: float, vmax: float) -> np.ndarray:
    h, _ = np.histogram(vals, bins=bins, range=(vmin, vmax))
    h = h.astype(np.float32)
    s = h.sum() + 1e-8
    return h / s

# ----------------------------
# 先验：梯度/边缘/方向/密度
# ----------------------------
def sobel_mag_ang(g: np.ndarray):
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = np.arctan2(gy, gx)  # [-pi, pi]
    return mag, ang

def edge_density(mag: np.ndarray, mode="p95"):
    if mode == "p95":
        thr = np.percentile(mag, 95.0)
    else:
        thr = mag.mean() + 0.5 * mag.std()
    return float((mag > thr).mean())

# ----------------------------
# 先验：低频亮度/热分布（IR偏）
# ----------------------------
def radial_energy_fft(g: np.ndarray, bins: int = 16) -> np.ndarray:
    F = np.fft.fft2(g)
    S = (F.real**2 + F.imag**2).astype(np.float32)
    H, W = S.shape
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    hist = np.zeros(bins, dtype=np.float32)
    for b in range(bins):
        r0 = b * (rmax / bins)
        r1 = (b + 1) * (rmax / bins)
        mask = (r >= r0) & (r < r1)
        if mask.any():
            hist[b] = S[mask].mean()
    hist /= (hist.sum() + 1e-8)
    return hist

def lowfreq_stats(g: np.ndarray, ksize=21, sigma=5.0):
    ker = gaussian_kernel(ksize, sigma)
    low = fft_conv2d(g, ker)
    mu, var = float(low.mean()), float(low.var())
    energy = float((low*low).mean())
    return np.array([mu, var, energy], np.float32)

# ----------------------------
# 先验：显著性（谱残差）
# ----------------------------
def spectral_residual_saliency(g: np.ndarray, avg_k=3, post_blur=5):
    # Ref: Hou & Zhang, CVPR 2007
    G = np.fft.fft2(g)
    A = np.abs(G) + 1e-8
    L = np.log(A)
    L_avg = box_filter2d(L, k=avg_k)
    R = L - L_avg
    S = np.exp(R) * np.exp(1j * np.angle(G))
    sal = np.abs(np.fft.ifft2(S))**2
    if post_blur > 1:
        ker = gaussian_kernel(size=post_blur, sigma=post_blur/3)
        sal = fft_conv2d(sal, ker)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.astype(np.float32)

# ----------------------------
# 先验：可见光颜色统计（HSV + 灰世界）
# ----------------------------
def rgb_to_hsv_np(img_rgb01: np.ndarray) -> np.ndarray:
    # img: [H,W,3], [0,1]
    r, g, b = img_rgb01[...,0], img_rgb01[...,1], img_rgb01[...,2]
    cmax = np.max(img_rgb01, axis=-1)
    cmin = np.min(img_rgb01, axis=-1)
    delta = cmax - cmin + 1e-8

    # Hue
    h = np.zeros_like(cmax)
    mask = (cmax == r)
    h[mask] = ((g - b)/delta)[mask] % 6
    mask = (cmax == g)
    h[mask] = ((b - r)/delta)[mask] + 2
    mask = (cmax == b)
    h[mask] = ((r - g)/delta)[mask] + 4
    h = (h / 6.0)  # [0,1)

    # Saturation & Value
    s = delta / (cmax + 1e-8)
    v = cmax

    return np.stack([h, s, v], axis=-1).astype(np.float32)

def color_stats(img_rgb01: np.ndarray):
    hsv = rgb_to_hsv_np(img_rgb01)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    hsv_meanvar = np.array([h.mean(), h.var(), s.mean(), s.var(), v.mean(), v.var()], np.float32)
    # Hue直方图（环）
    h_hist = hist_norm(h.reshape(-1), bins=16, vmin=0.0, vmax=1.0)
    # 灰世界偏差（期望R≈G≈B；用通道均值的方差近似）
    mr, mg, mb = img_rgb01[...,0].mean(), img_rgb01[...,1].mean(), img_rgb01[...,2].mean()
    grayworld_dev = float(np.var([mr, mg, mb]))
    return hsv_meanvar, h_hist, grayworld_dev

# ----------------------------
# 先验：纹理响应（Gabor能量）
# ----------------------------
def gabor_kernel(size, sigma, theta, lambd, gamma=0.5, psi=0.0):
    y, x = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-(x_t**2 + (gamma**2) * y_t**2) / (2*sigma**2)) * np.cos(2*np.pi * x_t / lambd + psi)
    gb -= gb.mean()
    return (gb / (np.sqrt((gb*gb).sum()) + 1e-8)).astype(np.float32)

def gabor_energy(g: np.ndarray, thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4), lambds=(4,8,16)):
    # 4方向 × 3尺度 = 12 维
    es = []
    for th in thetas:
        for lb in lambds:
            size = max(9, int(6*lb))
            ker = gabor_kernel(size=size, sigma=lb/2.5, theta=th, lambd=lb, gamma=0.5, psi=0.0)
            resp = fft_conv2d(g, ker)
            es.append(float((resp*resp).mean()))
    es = np.array(es, dtype=np.float32)
    # 归一化到能量比例
    es /= (es.sum() + 1e-8)
    return es  # (12,)

# ----------------------------
# 主流程：按镜像路径保存 .npy
# ----------------------------
def build_one(args):
    img_path, data_root, priors_root, modality = args
    ip = Path(img_path)
    rel = ip.relative_to(data_root)
    out = (Path(priors_root) / rel).with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(ip).convert("RGB")
        g = to_gray01(img)

        if modality == "ir":
            # ---------------- IR 先验 ----------------
            ir_low_hist  = radial_energy_fft(g, bins=16)
            ir_low_stats = lowfreq_stats(g, ksize=21, sigma=5.0)
            mag, ang = sobel_mag_ang(g)
            ir_edge_mag_hist = hist_norm(mag.reshape(-1), 8, 0.0, max(1e-3, mag.max()))
            ir_edge_ori_hist = hist_norm(ang.reshape(-1), 8, -np.pi, np.pi)
            ir_edge_density  = np.array([edge_density(mag, "p95")], np.float32)
            sal = spectral_residual_saliency(g, avg_k=3, post_blur=5)
            ir_sal_hist   = hist_norm(sal.reshape(-1), 16, 0.0, 1.0)
            ir_sal_energy = np.array([float((sal*sal).mean())], np.float32)

            np.save(out, {
                "modality": "ir",
                "ir": {
                    "lowfreq_hist": ir_low_hist,
                    "lowfreq_stats": ir_low_stats,
                    "edge_mag_hist": ir_edge_mag_hist,
                    "edge_ori_hist": ir_edge_ori_hist,
                    "edge_density": ir_edge_density,
                    "saliency_hist": ir_sal_hist,
                    "saliency_energy": ir_sal_energy,
                }
            }, allow_pickle=True)

        elif modality == "rgb":
            # ---------------- RGB 先验 ----------------
            rgb01 = np.asarray(img, dtype=np.float32) / 255.0
            hsv_meanvar, h_hist, gw_dev = color_stats(rgb01)
            vis_gabor = gabor_energy(g, thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4), lambds=(4,8,16))
            sal = spectral_residual_saliency(g, avg_k=3, post_blur=5)
            vis_sal_hist   = hist_norm(sal.reshape(-1), 16, 0.0, 1.0)
            vis_sal_energy = np.array([float((sal*sal).mean())], np.float32)

            np.save(out, {
                "modality": "rgb",
                "vis": {
                    "hsv_meanvar": hsv_meanvar,
                    "h_hist": h_hist,
                    "grayworld_dev": np.array([gw_dev], np.float32),
                    "gabor_energy": vis_gabor,
                    "saliency_hist": vis_sal_hist,
                    "saliency_energy": vis_sal_energy,
                }
            }, allow_pickle=True)

        return True, str(out)

    except Exception as e:
        return False, f"{ip}: {e}"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="原始图像根目录")
    ap.add_argument("--priors-root", required=True, help="先验保存根目录（镜像结构）")
    ap.add_argument("--modality", required=True, choices=["ir", "rgb"], help="指定当前目录图像模态：ir 或 rgb")
    ap.add_argument("--num-workers", type=int, default=max(1, cpu_count()//2))
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    priors_root = Path(args.priors_root).resolve()
    paths = [str(p) for p in data_root.rglob("*") if p.is_file() and is_image(p)]
    if len(paths) == 0:
        print("[WARN] No images under:", data_root)
        return

    try:
        if get_start_method(allow_none=True) is None:
            set_start_method("spawn")
    except RuntimeError:
        pass

    tasks = [(p, data_root, priors_root, args.modality) for p in paths]
    with Pool(args.num_workers) as pool:
        for ok, msg in pool.imap_unordered(build_one, tasks):
            if not ok:
                print("[ERR]", msg)
    print("[DONE] priors saved under:", priors_root)


if __name__ == "__main__":
    main()
