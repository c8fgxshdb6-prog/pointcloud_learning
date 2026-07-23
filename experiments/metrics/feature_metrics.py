"""
特征失真评估指标集
所有函数接受 numpy 数组，形状为 (B, C, N) 或 (B, N, C)
但为了统一，约定输入为 (B, C, N) 以匹配 PointNet++ 输出格式。
若需要处理 (B, N, C)，请先转换。
"""

import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import warnings

def compute_mse(clean, noisy):
    """均方误差，逐样本平均"""
    return np.mean((clean - noisy) ** 2, axis=(1, 2))

def compute_psnr(clean, noisy, max_val=None):
    """峰值信噪比 (dB)"""
    mse = compute_mse(clean, noisy)
    if max_val is None:
        max_val = np.max(clean, axis=(1, 2), keepdims=True)
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-8))
    return psnr

def compute_cosine_similarity(clean, noisy):
    """余弦相似度，沿所有元素计算"""
    B = clean.shape[0]
    clean_flat = clean.reshape(B, -1)
    noisy_flat = noisy.reshape(B, -1)
    dot = np.sum(clean_flat * noisy_flat, axis=1)
    norm_clean = np.linalg.norm(clean_flat, axis=1)
    norm_noisy = np.linalg.norm(noisy_flat, axis=1)
    sim = dot / (norm_clean * norm_noisy + 1e-8)
    return sim

def compute_ssim(clean, noisy, data_range=None):
    """
    结构相似性 (SSIM)，仅当空间维度 >= 7 时可用。
    将特征视为单通道图像 (H, W)，其中 H = N (点数)，W = 1 或 C？
    注意：对于 SA1/SA2，N 可能很大，通常作为高度，宽度设为 C？这里简化为将特征 reshape 为 (H, W) 其中 H=N, W=C。
    但 SSIM 通常用于 2D 图像，需要空间结构。如果您的特征空间维度是点数，可能没有局部结构，因此 SSIM 可能不适用。
    此处提供一种实现，但使用时请根据实际情况判断。
    """
    B, C, N = clean.shape
    # 将特征视为 (B, 1, N, C) 的单通道图像，计算 SSIM 时取平均
    ssim_vals = []
    for i in range(B):
        # 将 (C, N) 转为 (N, C) 作为图像
        img_clean = clean[i].T  # (N, C)
        img_noisy = noisy[i].T
        # 如果 N 或 C 太小，可能无法计算，返回 NaN
        if img_clean.shape[0] < 7 or img_clean.shape[1] < 7:
            ssim_vals.append(np.nan)
            continue
        try:
            # 使用多通道 SSIM (将每个通道视为独立，取平均)
            s = ssim(img_clean, img_noisy, data_range=data_range, channel_axis=-1, win_size=min(7, img_clean.shape[0], img_clean.shape[1]))
            ssim_vals.append(s)
        except Exception as e:
            warnings.warn(f"SSIM 计算失败: {e}")
            ssim_vals.append(np.nan)
    return np.array(ssim_vals)

def compute_stats_shift(clean, noisy):
    """均值偏移、方差变化、信息熵变化"""
    B = clean.shape[0]
    # 均值
    mean_clean = np.mean(clean, axis=(1, 2))
    mean_noisy = np.mean(noisy, axis=(1, 2))
    mean_shift = np.abs(mean_clean - mean_noisy)
    # 方差
    var_clean = np.var(clean, axis=(1, 2))
    var_noisy = np.var(noisy, axis=(1, 2))
    var_change = np.abs(var_clean - var_noisy)
    # 信息熵（需量化特征值）
    ent_clean = []
    ent_noisy = []
    bins = 20  # 可调整
    for i in range(B):
        hist_c, _ = np.histogram(clean[i].flatten(), bins=bins)
        hist_n, _ = np.histogram(noisy[i].flatten(), bins=bins)
        ent_clean.append(entropy(hist_c + 1e-8))
        ent_noisy.append(entropy(hist_n + 1e-8))
    ent_shift = np.array(ent_clean) - np.array(ent_noisy)
    return mean_shift, var_change, ent_shift

def compute_bandwidth(features):
    """估计特征占用的带宽 (KB)，假设 float32"""
    bytes_per_element = 4
    total_bytes = features.size * bytes_per_element
    return total_bytes / 1024.0

def compute_distortion_per_band(mse, bandwidth):
    """失真-带宽比：MSE / bandwidth"""
    return mse / bandwidth

# 如果需要同时计算多个指标，可以封装一个函数
def compute_all_metrics(clean, noisy):
    """计算所有指标，返回字典"""
    metrics = {}
    metrics['mse'] = compute_mse(clean, noisy)
    metrics['psnr'] = compute_psnr(clean, noisy)
    metrics['cos_sim'] = compute_cosine_similarity(clean, noisy)
    metrics['ssim'] = compute_ssim(clean, noisy)
    mean_shift, var_change, ent_shift = compute_stats_shift(clean, noisy)
    metrics['mean_shift'] = mean_shift
    metrics['var_change'] = var_change
    metrics['ent_shift'] = ent_shift
    metrics['bandwidth'] = compute_bandwidth(clean)
    metrics['dist_per_band'] = metrics['mse'] / metrics['bandwidth']
    return metrics