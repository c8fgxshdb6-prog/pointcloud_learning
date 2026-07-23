# run_ra_trained_sa1_sa3.py
# 训练后 RA + 解码器实验（AWGN），SA1 小批次处理避免 OOM
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.pointnet_adapter import pointnet_feature_to_adaptive, adaptive_feature_to_pointnet
from adapters.swin_adaptive_modules import RateModNet
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
    signal_power = torch.mean(features ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snr_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    rate_ratio_list = [0.2, 0.5, 0.8, 1.0]
    n_repeats = 10

    layer_configs = [
        {'layer': 'sa1', 'n_total': 500, 'micro_bs': 50, 'metric_n': 200},
        {'layer': 'sa3', 'n_total': 500, 'micro_bs': 200, 'metric_n': 200},
    ]

    results = []

    for cfg in layer_configs:
        layer = cfg['layer']
        micro_bs = cfg['micro_bs']
        n_total = cfg['n_total']
        n_metric = cfg['metric_n']

        clean = np.load(f'results/clean_features_{layer}.npy')
        clean = clean[:n_total]
        B, C, N = clean.shape
        print(f"\nLayer {layer}: ({B},{C},{N}), C={C}, N={N}, micro_batch={micro_bs}")

        # Load models
        ra_net = RateModNet(feat_dim=C, hidden_dim=int(C * 1.5), num_layers=7).to(device)
        ra_net.load_state_dict(torch.load(f'pretrained/ra_net_{layer}_trained.pth', map_location=device))
        print(f"[OK] Loaded ra_net_{layer}_trained.pth")

        decoder = Decoder(feat_dim=C).to(device)
        decoder.load_state_dict(torch.load(f'pretrained/ra_decoder_{layer}.pth', map_location=device))
        print(f"[OK] Loaded ra_decoder_{layer}.pth")

        # 预生成所有 (snr, rate_ratio, rep) 的组合参数
        combos = [(s, r, rep)
                  for s in snr_list
                  for r in rate_ratio_list
                  for rep in range(n_repeats)]

        all_mse, all_psnr, all_cos = {c: [] for c in combos}, {c: [] for c in combos}, {c: [] for c in combos}

        # 逐 micro-batch 处理，每次只加载一小部分数据到 GPU
        for start in range(0, B, micro_bs):
            end = min(start + micro_bs, B)
            micro_clean = torch.from_numpy(clean[start:end]).float().to(device)
            micro_B = micro_clean.shape[0]

            for (snr, rate_ratio, rep) in tqdm(combos, desc=f'  sa1[{start}:{end}]' if layer == 'sa1' else f'  sa3[{start}:{end}]', leave=False):
                rate = max(1, int(C * rate_ratio))
                with torch.no_grad():
                    x = pointnet_feature_to_adaptive(micro_clean)
                    x, mask = ra_net(x, rate)
                    x = adaptive_feature_to_pointnet(x)
                    x = awgn_channel(x, snr)
                    x = decoder(x)

                # 移回 CPU 逐样本计算指标
                c_np = micro_clean.cpu().numpy()
                r_np = x.cpu().numpy()
                for i in range(micro_B):
                    m = compute_all_metrics(c_np[i:i+1], r_np[i:i+1])
                    all_mse[(snr, rate_ratio, rep)].append(m['mse'].mean())
                    all_psnr[(snr, rate_ratio, rep)].append(m['psnr'].mean())
                    all_cos[(snr, rate_ratio, rep)].append(m['cos_sim'].mean())

            del micro_clean, x, c_np, r_np
            torch.cuda.empty_cache()

        # 汇总结果
        channel_count = max(1, int(C * rate_ratio_list[-1]))
        bw = C * N * 4 / 1024  # KB per sample (approx, varies by rate_ratio)
        for (snr, rate_ratio, rep) in combos:
            results.append({
                'layer': layer, 'snr': snr, 'rate_ratio': rate_ratio,
                'rate': max(1, int(C * rate_ratio)), 'repeat': rep,
                'bandwidth': bw * rate_ratio,
                'mse': np.mean(all_mse[(snr, rate_ratio, rep)]),
                'psnr': np.mean(all_psnr[(snr, rate_ratio, rep)]),
                'cos_sim': np.mean(all_cos[(snr, rate_ratio, rep)]),
            })

        del ra_net, decoder
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/channel_exp_results_ra_trained_sa1_sa3.csv', index=False)
    print("\nDone. Saved to results/channel_exp_results_ra_trained_sa1_sa3.csv")


if __name__ == '__main__':
    main()
