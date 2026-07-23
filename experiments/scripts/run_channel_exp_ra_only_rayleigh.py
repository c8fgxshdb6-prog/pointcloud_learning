# run_channel_exp_ra_only_rayleigh.py
import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = r'D:\Users\yxf\Desktop\pointcloud_learning'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'SwinJSCC'))

from experiments.adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from experiments.adapters.swin_adaptive_modules import RateModNet
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
    snr_list = [0, 2, 4,6,8,10,12,14,16, 18, 20]
    rate_ratio_list = [0.2, 0.5, 0.8, 1.0]
    n_repeats = 10

    results = []

    for layer in feature_layers:
        clean = np.load(f'results/clean_features_{layer}.npy')
        clean = clean[:200]
        B, C, N = clean.shape
        print(f"处理 {layer}: 形状 {clean.shape}, 通道数 {C}, 点数 {N}")
        clean_t = torch.from_numpy(clean).float().to(device)

        ra_net = RateModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(device)

        for snr in tqdm(snr_list, desc=f'SNR for {layer}'):
            for rate_ratio in rate_ratio_list:
                rate = int(C * rate_ratio)
                if rate == 0:
                    continue
                for rep in range(n_repeats):
                    x_adaptive = pointnet_feature_to_adaptive(clean_t)
                    with torch.no_grad():
                        x_ra, mask = ra_net(x_adaptive, rate)
                    x_modulated = adaptive_feature_to_pointnet(x_ra)
                    x_noisy = channel.forward(x_modulated, chan_param=snr, avg_pwr=False)

                    clean_np = clean_t.detach().cpu().numpy()
                    noisy_np = x_noisy.detach().cpu().numpy()
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
    df.to_csv('results/channel_exp_results_ra_only_rayleigh.csv', index=False)
    print("Rayleigh 信道 RA only 实验完成，结果保存至 results/channel_exp_results_ra_only_rayleigh.csv")

if __name__ == '__main__':
    main()