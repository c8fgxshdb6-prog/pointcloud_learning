# test_all.py
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from experiments.adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from experiments.adapters.swin_adaptive_modules import ChannelModNet, RateModNet
from experiments.metrics.feature_metrics import compute_all_metrics

print("="*50)
print("开始测试三个模块的集成...")
print("="*50)

# 1. 模拟 PointNet++ 输出特征 (B, C, N)
B, C, N = 4, 640, 128
x_pointnet = torch.randn(B, C, N)

# 2. 转换为自适应格式
x_adaptive = pointnet_feature_to_adaptive(x_pointnet)  # (B, N, C)

# 3. 实例化自适应模块
sa_net = ChannelModNet(feat_dim=C)
ra_net = RateModNet(feat_dim=C)

# 4. 应用自适应
snr = 10.0
rate = 320
x_modulated = sa_net(x_adaptive, snr)
x_modulated, mask = ra_net(x_modulated, rate)

# 5. 转回 PointNet++ 格式
x_modulated_pc = adaptive_feature_to_pointnet(x_modulated)  # (B, C, N)

# 6. 加噪声模拟信道（简单 AWGN）
noise = 0.1 * torch.randn_like(x_modulated_pc)
x_noisy = x_modulated_pc + noise

# 7. 转换为 numpy 并计算指标（使用 detach 避免梯度问题）
clean_np = x_pointnet.detach().cpu().numpy()
noisy_np = x_noisy.detach().cpu().numpy()
metrics = compute_all_metrics(clean_np, noisy_np)

print("集成流程完成，指标计算结果：")
for k, v in metrics.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: shape {v.shape}, mean {v.mean():.4f}")
    else:
        print(f"{k}: {v}")

print("="*50)
print("✅ 所有模块集成测试通过！")