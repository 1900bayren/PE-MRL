# data/priors.py
from pathlib import Path
import numpy as np
import torch

_EPS = 1e-8

def _l1_norm_hist(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    s = x.sum() + _EPS
    return x / s

def _robust_z(x: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    # IQR->近似std: iqr/1.349；下限避免除零
    scale = np.maximum(iqr / 1.349, 1e-6)
    return (x - med) / scale

def _group_l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + _EPS
    return x / n

class PriorLoaderDual:
    """
    与原版接口一致：
      - load(img_path, modality) 返回 shape=(padded_dim,) 的 float32 Tensor
      - modality: 0=IR, 1=VIS
    仅在内部对每一组特征做“可比尺度”的稳健预处理；维度/顺序/返回类型不变
    """
    def __init__(self, data_root_ir: str, priors_root_ir: str,
                 data_root_vis: str, priors_root_vis: str,
                 prior_dim_ir: int = 53, prior_dim_vis: int = 52, strict: bool = False):
        self.data_root_ir  = Path(data_root_ir).resolve()
        self.priors_root_ir = Path(priors_root_ir).resolve()
        self.data_root_vis = Path(data_root_vis).resolve()
        self.priors_root_vis = Path(priors_root_vis).resolve()
        self.prior_dim_ir  = int(prior_dim_ir)
        self.prior_dim_vis = int(prior_dim_vis)
        self.padded_dim = max(self.prior_dim_ir, self.prior_dim_vis)
        self.strict = strict

        # 可选：若目录下提供全局统计文件，则做 per-dim z-score（稳态最好）
        self.stats_ir  = self._try_load_stats(self.priors_root_ir, "stats_ir.npz")
        self.stats_vis = self._try_load_stats(self.priors_root_vis, "stats_vis.npz")

    def _try_load_stats(self, root: Path, fname: str):
        p = root / fname
        if p.exists():
            z = np.load(p)
            # 期望含有 'median' 与 'iqr' 两个键，shape 与维度一致
            return {"median": z["median"].astype(np.float32),
                    "iqr":    z["iqr"].astype(np.float32)}
        return None

    def _prior_path(self, img_path: str, is_ir: bool) -> Path:
        ip = Path(img_path).resolve()
        if is_ir:
            rel = ip.relative_to(self.data_root_ir)
            return (self.priors_root_ir / rel).with_suffix(".npy")
        else:
            rel = ip.relative_to(self.data_root_vis)
            return (self.priors_root_vis / rel).with_suffix(".npy")

    # ----------------- 组装 + 预处理：IR（保留：lowfreq_* + edge_*） -----------------
    # def _pack_ir(self, d: dict) -> np.ndarray:
    #     lowfreq_hist = np.asarray(d["lowfreq_hist"], np.float32)  # 16
    #     lowfreq_stats = np.asarray(d["lowfreq_stats"], np.float32)  # 3
    #     edge_mag_hist = np.asarray(d["edge_mag_hist"], np.float32)  # 8
    #     edge_ori_hist = np.asarray(d["edge_ori_hist"], np.float32)  # 8
    #     edge_density = np.asarray(d["edge_density"], np.float32)  # 1
    #     # —— 不再使用：saliency_hist(16), saliency_energy(1)
    #
    #     # 预处理（与你现有一致）
    #     lowfreq_hist = _group_l2(_l1_norm_hist(lowfreq_hist))
    #     edge_mag_hist = _group_l2(_l1_norm_hist(edge_mag_hist))
    #     edge_ori_hist = _group_l2(_l1_norm_hist(edge_ori_hist))
    #     edge_density = np.log1p(np.maximum(edge_density, 0.0))
    #     lowfreq_stats = (lowfreq_stats - lowfreq_stats.mean()) / (lowfreq_stats.std() + 1e-6)
    #
    #     # 仅拼接保留的 5 组 → 16+3+8+8+1 = 36
    #     v = np.concatenate([
    #         lowfreq_hist, lowfreq_stats,
    #         edge_mag_hist, edge_ori_hist, edge_density.reshape(1),
    #     ]).astype(np.float32)
    #
    #     if self.stats_ir is not None:
    #         if self.stats_ir["median"].shape[0] == v.shape[0]:
    #             v = _robust_z(v, self.stats_ir["median"], self.stats_ir["iqr"]).astype(np.float32)
    #
    #     assert v.shape[0] == self.prior_dim_ir  # 现在应为 36
    #     return v
    #
    # # ----------------- 组装 + 预处理：VIS（保留：hsv_meanvar + gabor_energy + saliency_*） -----------------
    # def _pack_vis(self, d: dict) -> np.ndarray:
    #     hsv_meanvar = np.asarray(d["hsv_meanvar"], np.float32)  # 6
    #     gabor_energy = np.asarray(d["gabor_energy"], np.float32)  # 12
    #     saliency_hist = np.asarray(d["saliency_hist"], np.float32)  # 16
    #     saliency_energy = np.asarray(d["saliency_energy"], np.float32)  # 1
    #     # —— 不再使用：h_hist(16), grayworld_dev(1)
    #
    #     saliency_hist = _group_l2(_l1_norm_hist(saliency_hist))
    #     hsv_meanvar = (hsv_meanvar - hsv_meanvar.mean()) / (hsv_meanvar.std() + 1e-6)
    #     gabor_energy = _group_l2(np.log1p(np.maximum(gabor_energy, 0.0)))
    #
    #     # 仅拼接保留的 4 组 → 6+12+16+1 = 35
    #     v = np.concatenate([
    #         hsv_meanvar, gabor_energy, saliency_hist, saliency_energy.reshape(1)
    #     ]).astype(np.float32)
    #
    #     if self.stats_vis is not None:
    #         if self.stats_vis["median"].shape[0] == v.shape[0]:
    #             v = _robust_z(v, self.stats_vis["median"], self.stats_vis["iqr"]).astype(np.float32)
    #
    #     assert v.shape[0] == self.prior_dim_vis  # 现在应为 35
    #     return v

    # # ----------------- 组装 + 预处理：IR -----------------
    def _pack_ir(self, d: dict) -> np.ndarray:
        # 原始分组（保持顺序与维度不变）
        lowfreq_hist   = np.asarray(d["lowfreq_hist"],   np.float32)  # 16
        lowfreq_stats  = np.asarray(d["lowfreq_stats"],  np.float32)  # 3  (均值/方差/…)
        edge_mag_hist  = np.asarray(d["edge_mag_hist"],  np.float32)  # 8
        edge_ori_hist  = np.asarray(d["edge_ori_hist"],  np.float32)  # 8
        edge_density   = np.asarray(d["edge_density"],   np.float32)  # 1 (密度/能量类)
        saliency_hist  = np.asarray(d["saliency_hist"],  np.float32)  # 16
        saliency_energy= np.asarray(d["saliency_energy"],np.float32)  # 1 (能量类)

        # 组内归一化 / 稳健缩放（无全局统计时的本地方案）
        lowfreq_hist   = _l1_norm_hist(lowfreq_hist)
        edge_mag_hist  = _l1_norm_hist(edge_mag_hist)
        edge_ori_hist  = _l1_norm_hist(edge_ori_hist)
        saliency_hist  = _l1_norm_hist(saliency_hist)

        # 能量/密度：log1p 压缩 + 再做 group-L2，避免一维主导
        edge_density    = np.log1p(np.maximum(edge_density, 0.0))
        saliency_energy = np.log1p(np.maximum(saliency_energy, 0.0))

        # 均值/方差类：稳健中心化（若无全局统计，就用自身的均值/方差近似）
        lowfreq_stats = (lowfreq_stats - lowfreq_stats.mean()) / (lowfreq_stats.std() + 1e-6)

        # 组间均衡：每组再做一次 L2
        lowfreq_hist   = _group_l2(lowfreq_hist)
        edge_mag_hist  = _group_l2(edge_mag_hist)
        edge_ori_hist  = _group_l2(edge_ori_hist)
        saliency_hist  = _group_l2(saliency_hist)

        v = np.concatenate([
            lowfreq_hist, lowfreq_stats,
            edge_mag_hist, edge_ori_hist, edge_density.reshape(1),
            saliency_hist, saliency_energy.reshape(1),
        ]).astype(np.float32)  # 16+3+8+8+1+16+1 = 53

        # 若提供了全局统计（推荐离线预计算），做 per-dim robust z
        if self.stats_ir is not None and self.stats_ir["median"].shape[0] == v.shape[0]:
            v = _robust_z(v, self.stats_ir["median"], self.stats_ir["iqr"]).astype(np.float32)

        assert v.shape[0] == self.prior_dim_ir
        return v

    # ----------------- 组装 + 预处理：VIS -----------------
    def _pack_vis(self, d: dict) -> np.ndarray:
        hsv_meanvar   = np.asarray(d["hsv_meanvar"],   np.float32)  # 6 (均值/方差)
        h_hist        = np.asarray(d["h_hist"],        np.float32)  # 16
        grayworld_dev = np.asarray(d["grayworld_dev"], np.float32)  # 1
        gabor_energy  = np.asarray(d["gabor_energy"],  np.float32)  # 12 (能量)
        saliency_hist = np.asarray(d["saliency_hist"], np.float32)  # 16
        saliency_energy = np.asarray(d["saliency_energy"], np.float32)  # 1

        h_hist        = _l1_norm_hist(h_hist)
        saliency_hist = _l1_norm_hist(saliency_hist)

        # 能量/偏差：log1p 压缩再标准化
        gabor_energy  = np.log1p(np.maximum(gabor_energy, 0.0))
        grayworld_dev = np.log1p(np.maximum(grayworld_dev, 0.0))

        # 均值/方差：稳健标准化（本地近似）
        hsv_meanvar = (hsv_meanvar - hsv_meanvar.mean()) / (hsv_meanvar.std() + 1e-6)

        # 组间均衡
        h_hist        = _group_l2(h_hist)
        saliency_hist = _group_l2(saliency_hist)
        gabor_energy  = _group_l2(gabor_energy)

        v = np.concatenate([
            hsv_meanvar, h_hist, grayworld_dev.reshape(1),
            gabor_energy, saliency_hist, saliency_energy.reshape(1),
        ]).astype(np.float32)  # 6+16+1+12+16+1 = 52

        if self.stats_vis is not None and self.stats_vis["median"].shape[0] == v.shape[0]:
            v = _robust_z(v, self.stats_vis["median"], self.stats_vis["iqr"]).astype(np.float32)

        assert v.shape[0] == self.prior_dim_vis
        return v

    # ----------------- 外部接口：不变 -----------------
    def load(self, img_path: str, modality: int):
        """
        modality: 0=IR, 1=VIS
        返回 shape=(padded_dim,) 的 float32 Tensor；缺失则全 0（与你原逻辑一致）
        """
        is_ir = (modality == 0)
        p = self._prior_path(img_path, is_ir)

        if not p.exists():
            if self.strict:
                raise FileNotFoundError(str(p))
            pad = np.zeros(self.padded_dim, np.float32)
            return torch.from_numpy(pad)

        arr = np.load(p, allow_pickle=True).item()

        if is_ir:
            v = self._pack_ir(arr["ir"])
            pad = np.zeros(self.padded_dim, np.float32)
            pad[:self.prior_dim_ir] = v
            return torch.from_numpy(pad)
        else:
            v = self._pack_vis(arr["vis"])
            pad = np.zeros(self.padded_dim, np.float32)
            pad[:self.prior_dim_vis] = v
            return torch.from_numpy(pad)
