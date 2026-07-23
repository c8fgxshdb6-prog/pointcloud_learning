# run_no_adapt_exp.py
import sys
import os
# 将项目根目录加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))          # experiments/scripts
project_root = os.path.dirname(os.path.dirname(current_dir))      # 项目根目录
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from experiments.metrics.feature_metrics import compute_all_metrics

# ... 其余代码保持不变

def awgn_channel(features, snr_db):
    signal_power = torch.mean(features ** 2, dim=(1,2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_layers = ['sa1', 'sa2', 'sa3']
    snr_list = [0, 2,4,6,8, 10,12,14, 16,18, 20]
    rate_ratio_list = [0.2, 0.5, 0.8, 1.0]
    n_repeats = 10

    results = []

    for layer in feature_layers:
        clean = np.load(f'results/clean_features_{layer}.npy')
        clean = clean[:1000]   # 只取前200个样本
        B, C, N = clean.shape
        print(f"处理 {layer}: 形状 {clean.shape}")
        clean_t = torch.from_numpy(clean).float().to(device)

        for snr in tqdm(snr_list, desc=f'SNR for {layer}'):
            for rate_ratio in rate_ratio_list:
                rate = int(C * rate_ratio)
                for rep in range(n_repeats):
                    x_adaptive = pointnet_feature_to_adaptive(clean_t)
                    x_modulated = adaptive_feature_to_pointnet(x_adaptive)
                    x_noisy = awgn_channel(x_modulated, snr)

                    clean_np = clean_t.cpu().numpy()
                    noisy_np = x_noisy.cpu().numpy()
                    metrics = compute_all_metrics(clean_np, noisy_np)

                    result = {
                        'layer': layer,
                        'snr': snr,
                        'rate_ratio': rate_ratio,
                        'rate': rate,
                        'repeat': rep,
                        'bandwidth': metrics['bandwidth'],
                    }
                    for key in ['mse', 'psnr', 'cos_sim', 'ssim', 'mean_shift', 'var_change', 'ent_shift', 'dist_per_band']:
                        val = metrics[key]
                        result[key] = val.mean() if isinstance(val, np.ndarray) else val
                    results.append(result)

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/no_adapt_results.csv', index=False)
    print("无自适应基线实验完成，结果保存至 results/no_adapt_results.csv")

if __name__ == '__main__':
    main()