# run_channel_exp.py (训练后 SA + 解码器，固定 RA=1.0)
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from adapters.swin_adaptive_modules import ChannelModNet, RateModNet
from metrics.feature_metrics import compute_all_metrics

class Decoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feat_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)

def awgn_channel(features, snr_db):
    signal_power = torch.mean(features ** 2, dim=(1,2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_layers = ['sa1', 'sa2', 'sa3']
    snr_list = [0,2,4,6,8,10,12,14,16,18,20]
    rate_ratio_list = [1.0]
    n_repeats = 10

    results = []

    for layer in feature_layers:
        clean = np.load(f'results/clean_features_{layer}.npy')
        clean = clean[:1000]
        B, C, N = clean.shape
        print(f"处理 {layer}: 形状 {clean.shape}, 通道数 {C}, 点数 {N}")
        clean_t = torch.from_numpy(clean).float().to(device)

        sa_net = ChannelModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(device)
        trained_sa_path = f'pretrained/sa_net_{layer}_trained.pth'
        if os.path.exists(trained_sa_path):
            sa_net.load_state_dict(torch.load(trained_sa_path, map_location=device))
            print(f"✅ 加载 {layer} 的预训练 SA 权重")
        else:
            print(f"⚠️ 未找到 {layer} 的预训练 SA 权重，使用随机初始化")

        decoder = Decoder(feat_dim=C).to(device)
        decoder_path = f'pretrained/decoder_{layer}.pth'
        if os.path.exists(decoder_path):
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            print(f"✅ 加载 {layer} 的解码器权重")
        else:
            print(f"⚠️ 未找到 {layer} 的解码器权重，将不使用解码器")
            decoder = None

        ra_net = RateModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(device)

        for snr in tqdm(snr_list, desc=f'SNR for {layer}'):
            for rate_ratio in rate_ratio_list:
                rate = int(C * rate_ratio)
                for rep in range(n_repeats):
                    # 整个前向过程不记录梯度
                    with torch.no_grad():
                        x_adaptive = pointnet_feature_to_adaptive(clean_t)
                        x_sa = sa_net(x_adaptive, snr)
                        x_ra, mask = ra_net(x_sa, rate)
                        x_modulated = adaptive_feature_to_pointnet(x_ra)
                        x_noisy = awgn_channel(x_modulated, snr)
                        if decoder is not None:
                            x_recon = decoder(x_noisy)
                        else:
                            x_recon = x_noisy

                    clean_np = clean_t.detach().cpu().numpy()
                    recon_np = x_recon.detach().cpu().numpy()
                    metrics = compute_all_metrics(clean_np, recon_np)

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
    df.to_csv('results/channel_exp_results_trained_with_decoder.csv', index=False)
    print("实验完成，结果已保存至 results/channel_exp_results_trained_with_decoder.csv")

if __name__ == '__main__':
    main()