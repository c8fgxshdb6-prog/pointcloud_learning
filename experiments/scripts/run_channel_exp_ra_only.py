# run_channel_exp_ra_only.py
import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from adapters.swin_adaptive_modules import RateModNet   # 只导入 RA
from metrics.feature_metrics import compute_all_metrics

def awgn_channel(features, snr_db):
    signal_power = torch.mean(features ** 2, dim=(1,2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_layers = ['sa1', 'sa2', 'sa3']
    snr_list = [0, 2,4,6,8,10, 12, 14,16,18 ,20]
    rate_ratio_list = [0.2, 0.5, 0.8, 1.0]   # 不同速率比例
    n_repeats = 10

    results = []

    for layer in feature_layers:
        clean = np.load(f'results/clean_features_{layer}.npy')
        clean = clean[:200]   # 可调整为更大样本
        B, C, N = clean.shape
        print(f"处理 {layer}: 形状 {clean.shape}, 通道数 {C}, 点数 {N}")
        clean_t = torch.from_numpy(clean).float().to(device)

        # 仅实例化 RA 模块
        ra_net = RateModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(device)

        for snr in tqdm(snr_list, desc=f'SNR for {layer}'):
            for rate_ratio in rate_ratio_list:
                rate = int(C * rate_ratio)
                if rate == 0:
                    continue
                for rep in range(n_repeats):
                    x_adaptive = pointnet_feature_to_adaptive(clean_t)   # (B, N, C)

                    with torch.no_grad():
                        # SA 部分：恒等映射（不做任何处理）
                        x_sa = x_adaptive
                        # RA 应用
                        x_ra, mask = ra_net(x_sa, rate)

                    x_modulated = adaptive_feature_to_pointnet(x_ra)   # (B, C, N)
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
    df.to_csv('results/channel_exp_results_ra_only.csv', index=False)
    print("实验完成，结果保存至 results/channel_exp_results_ra_only.csv")

if __name__ == '__main__':
    main()