# run_channel_exp_rayleigh.py (Rayleigh信道，固定RA=1.0，使用训练后SA+解码器)
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录和 SwinJSCC 路径
project_root = r'D:\Users\yxf\Desktop\pointcloud_learning'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'SwinJSCC'))

from experiments.adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from experiments.adapters.swin_adaptive_modules import ChannelModNet, RateModNet
from experiments.metrics.feature_metrics import compute_all_metrics
from net.channel import Channel

# 定义解码器（与训练时一致）
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

# 模拟 SwinJSCC Channel 所需的参数
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
    snr_list = [0, 2,4,6,8, 10,12,14,16, 18, 20]
    rate_ratio_list = [1.0]          # 固定 RA=1.0
    n_repeats =10

    results = []

    for layer in feature_layers:
        clean = np.load(f'results/clean_features_{layer}.npy')
        clean = clean[:1000]           # 取前200个样本
        B, C, N = clean.shape
        print(f"处理 {layer}: 形状 {clean.shape}, 通道数 {C}, 点数 {N}")
        clean_t = torch.from_numpy(clean).float().to(device)

        # 加载训练好的 SA 权重（来自 AWGN 训练）
        sa_net = ChannelModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(device)
        trained_sa_path = f'pretrained/sa_net_{layer}_trained.pth'
        if os.path.exists(trained_sa_path):
            sa_net.load_state_dict(torch.load(trained_sa_path, map_location=device))
            print(f"✅ 加载 {layer} 的预训练 SA 权重")
        else:
            print(f"⚠️ 未找到 {layer} 的预训练 SA 权重，使用随机初始化")

        # 加载训练好的解码器权重
        decoder = Decoder(feat_dim=C).to(device)
        decoder_path = f'pretrained/decoder_{layer}.pth'
        if os.path.exists(decoder_path):
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            print(f"✅ 加载 {layer} 的解码器权重")
        else:
            print(f"⚠️ 未找到 {layer} 的解码器权重，将不使用解码器")
            decoder = None

        # RA 模块（rate_ratio=1.0 时 mask 全为1，不影响）
        ra_net = RateModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(device)

        for snr in tqdm(snr_list, desc=f'SNR for {layer}'):
            for rate_ratio in rate_ratio_list:
                rate = int(C * rate_ratio)
                for rep in range(n_repeats):
                    with torch.no_grad():
                        x_adaptive = pointnet_feature_to_adaptive(clean_t)
                        x_sa = sa_net(x_adaptive, snr)
                        x_ra, mask = ra_net(x_sa, rate)
                        x_modulated = adaptive_feature_to_pointnet(x_ra)
                        # 使用 Rayleigh 信道
                        x_noisy = channel.forward(x_modulated, chan_param=snr, avg_pwr=False)
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
    df.to_csv('results/channel_exp_results_rayleigh_trained_with_decoder.csv', index=False)
    print("Rayleigh 信道实验完成，结果保存至 results/channel_exp_results_rayleigh_trained_with_decoder.csv")

if __name__ == '__main__':
    main()