# run_no_adapt_exp_rayleigh.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# 添加项目根目录和 SwinJSCC 路径
project_root = r'D:\Users\yxf\Desktop\pointcloud_learning'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'SwinJSCC'))

from experiments.adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from experiments.metrics.feature_metrics import compute_all_metrics
from net.channel import Channel

class DummyArgs:
    def __init__(self, channel_type):
        self.channel_type = channel_type
        self.multiple_snr = '10'

class DummyConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CUDA = True
        self.logger = None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channel_type = 'rayleigh'
    args = DummyArgs(channel_type)
    config = DummyConfig()
    channel = Channel(args, config)

    feature_layers = ['sa1', 'sa2', 'sa3']
    snr_list = [0,2,4,6 ,8,10,12, 14,16,18, 20]          # 可根据需要调整
    rate_ratio_list = [0.5, 1.0]           # 仅用于带宽记录，实际不改变特征
    n_repeats = 10                          # 重复次数，Rayleigh 建议增加到 5

    results = []

    for layer in feature_layers:
        clean = np.load(f'results/clean_features_{layer}.npy')
        # 限制样本数量以避免显存溢出（请根据实际情况调整，例如 200 或 100）
        clean = clean[:1000]                # <--- 添加这一行
        B, C, N = clean.shape
        print(f"处理 {layer}: 形状 {clean.shape}")
        clean_t = torch.from_numpy(clean).float().to(device)

        for snr in tqdm(snr_list, desc=f'SNR for {layer}'):
            for rate_ratio in rate_ratio_list:
                rate = int(C * rate_ratio)
                for rep in range(n_repeats):
                    # 无自适应：直接加噪声
                    x_adaptive = pointnet_feature_to_adaptive(clean_t)
                    x_modulated = adaptive_feature_to_pointnet(x_adaptive)
                    x_noisy = channel.forward(x_modulated, chan_param=snr, avg_pwr=False)

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
    df.to_csv('results/no_adapt_results_rayleigh.csv', index=False)
    print("Rayleigh 信道无自适应实验完成，结果保存至 results/no_adapt_results_rayleigh.csv")

if __name__ == '__main__':
    main()