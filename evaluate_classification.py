# evaluate_classification.py
# SA3 任务驱动分类精度评估
# 对比 4 种传输方案在 AWGN 信道下的分类精度保持能力
#
# 评估栈:
#   PointNet++ 前向 → SA3干净特征 → [4选1调制+解码] → 分类头 → 精度对比
#
# 4种方法:
#   NoAdapt : SA3 → AWGN → cls_head (无解码器)
#   SA-only : SA3 → sa_net(snr) → AWGN → sa_decoder → cls_head
#   RA-only : SA3 → ra_net(rate) → AWGN → ra_decoder → cls_head
#   SA+RA   : SA3 → sara_sa(snr) → sara_ra(rate) → AWGN → sara_decoder → cls_head

import sys
import os
sys.path.append('./Pointnet_Pointnet2_pytorch')
sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from collections import defaultdict

from swin_adaptive_modules import ChannelModNet, RateModNet

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'D:/Users/yxf/Desktop/pointcloud_learning/data/modelnet40_normal_resampled/test'
CHECKPOINT_PATH = 'D:/Users/yxf/Desktop/pointcloud_learning/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'
BATCH_SIZE = 32
NUM_POINTS = 1024
SNRS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]
N_REPEATS = 10
SA3_C = 1024  # SA3 通道数

print(f"Device: {DEVICE}")
print(f"SNRs: {SNRS}")
print(f"Rate Ratios: {RATE_RATIOS} -> channels: {[max(1,int(SA3_C*r)) for r in RATE_RATIOS]}")
print(f"Repeats per config: {N_REPEATS}")

# ============================================================
# 数据集 (从 extract_features.py 复现)
# ============================================================
class ModelNet40PLY(Dataset):
    def __init__(self, root, num_points=1024, use_normals=True):
        self.root = root
        self.num_points = num_points
        self.use_normals = use_normals
        self.classes = sorted([d for d in os.listdir(self.root)
                               if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.points = []
        self.labels = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            files = glob.glob(os.path.join(cls_path, '*.txt'))
            for f in files:
                self.points.append(f)
                self.labels.append(self.class_to_idx[cls])
        print(f"Loaded {len(self.points)} test samples from {root}")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        filepath = self.points[idx]
        label = self.labels[idx]
        data = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
        if data.shape[0] < self.num_points:
            choice = np.random.choice(data.shape[0], self.num_points, replace=True)
        else:
            choice = np.random.choice(data.shape[0], self.num_points, replace=False)
        data = data[choice, :]
        if not self.use_normals:
            data = data[:, :3]
        return data, label

# ============================================================
# 解码器定义 (与训练完全一致)
# ============================================================

class Decoder2Layer(nn.Module):
    """SA / RA 各自训练的 2 层解码器"""
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feat_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (B, C, N) -> (B, N, C) -> (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)

class DeeperDecoder(nn.Module):
    """SA+RA 联合训练的 3 层解码器"""
    def __init__(self, feat_dim, hidden1=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, feat_dim),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)

# ============================================================
# AWGN 信道
# ============================================================
def awgn_channel(features, snr_db):
    if isinstance(snr_db, (int, float)):
        snr_tensor = torch.full((features.shape[0], 1), snr_db, device=features.device)
    else:
        snr_tensor = snr_db.view(-1, 1).float()
    signal_power = torch.mean(features ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_tensor / 10.0)
    noise_power = signal_power / snr_linear.unsqueeze(-1)
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise

# ============================================================
# 分类头前向 (复用 PointNet++ 的 fc 层)
# ============================================================
def classify_features(model, features):
    """
    features: (B, 1024, 1) — 解码器输出 (或含噪特征)
    返回: (B,) 预测标签
    """
    # squeeze → (B, 1024) → 经过训练好的 FC 层
    x = features.squeeze(-1)  # (B, 1024)
    # 分类头: fc1 → bn1 → relu → drop1 → fc2 → bn2 → relu → drop2 → fc3 → log_softmax
    # model.eval() 时 dropout 是 identity
    x = model.drop1(F.relu(model.bn1(model.fc1(x))))
    x = model.drop2(F.relu(model.bn2(model.fc2(x))))
    x = model.fc3(x)
    x = F.log_softmax(x, -1)
    return x.argmax(dim=-1)

# ============================================================
# 主流程
# ============================================================
def main():
    # ---- 1. 加载数据 & 模型 ----
    dataset = ModelNet40PLY(root=DATA_ROOT, num_points=NUM_POINTS, use_normals=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    from pointnet2_cls_msg import get_model
    model = get_model(num_class=40, normal_channel=True).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("[OK] PointNet++ model loaded")

    # ---- 2. 加载各方法的调制+解码器权重 ----
    weights = {}

    # SA-only
    weights['SA'] = {}
    sa_net = ChannelModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
    sa_net.load_state_dict(torch.load('pretrained/sa_net_sa3_trained.pth', map_location=DEVICE))
    sa_net.eval()
    weights['SA']['mod_net'] = sa_net
    sa_dec = Decoder2Layer(SA3_C).to(DEVICE)
    sa_dec.load_state_dict(torch.load('pretrained/decoder_sa3.pth', map_location=DEVICE))
    sa_dec.eval()
    weights['SA']['decoder'] = sa_dec
    print("[OK] SA weights loaded")

    # RA-only
    weights['RA'] = {}
    ra_net = RateModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
    ra_net.load_state_dict(torch.load('pretrained/ra_net_sa3_trained.pth', map_location=DEVICE))
    ra_net.eval()
    weights['RA']['mod_net'] = ra_net
    ra_dec = Decoder2Layer(SA3_C).to(DEVICE)
    ra_dec.load_state_dict(torch.load('pretrained/ra_decoder_sa3.pth', map_location=DEVICE))
    ra_dec.eval()
    weights['RA']['decoder'] = ra_dec
    print("[OK] RA weights loaded")

    # SA+RA joint
    weights['SA+RA'] = {}
    sara_sa = ChannelModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
    sara_sa.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth', map_location=DEVICE))
    sara_sa.eval()
    weights['SA+RA']['sa_net'] = sara_sa
    sara_ra = RateModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
    sara_ra.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth', map_location=DEVICE))
    sara_ra.eval()
    weights['SA+RA']['ra_net'] = sara_ra
    sara_dec = DeeperDecoder(SA3_C).to(DEVICE)
    sara_dec.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth', map_location=DEVICE))
    sara_dec.eval()
    weights['SA+RA']['decoder'] = sara_dec
    print("[OK] SA+RA weights loaded")

    # ---- 3. 第一批: 收集所有样本的 SA3 特征和标签 ----
    all_features = []
    all_labels = []
    all_clean_preds = []

    print("\nExtracting SA3 features and clean predictions...")
    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc='PointNet++ forward'):
            points = points.transpose(2, 1).to(DEVICE)  # (B, 6, N) → 模型要求的输入
            cls_logits, _, _, l3 = model(points)         # l3: (B, 1024, 1)
            all_features.append(l3.cpu())
            all_labels.append(labels)
            all_clean_preds.append(cls_logits.argmax(dim=-1).cpu())

    # 拼接
    features_all = torch.cat(all_features, dim=0)  # (N_total, 1024, 1)
    labels_all = torch.cat(all_labels, dim=0)
    clean_preds_all = torch.cat(all_clean_preds, dim=0)
    N_total = features_all.shape[0]

    # 干净分类准确率
    acc_clean = (clean_preds_all == labels_all).float().mean().item()
    print(f"Total samples: {N_total}")
    print(f"Clean accuracy (upper bound): {acc_clean:.4f} ({acc_clean*100:.2f}%)")

    # ---- 4. 评估循环 ----
    results = []
    methods = ['NoAdapt', 'SA', 'RA', 'SA+RA']

    # 为加速, 预加载所有 SA3 特征到 GPU
    feats_gpu = features_all.to(DEVICE)  # (N, 1024, 1)
    # SA/RM 模块需要的输入格式是 (B, N_pts, C), SA3 的 N_pts=1
    feats_mod_format = feats_gpu.transpose(1, 2)  # (N, 1, 1024)

    total_configs = len(SNRS) * len(RATE_RATIOS) * N_REPEATS * len(methods)
    print(f"\nEvaluating {total_configs} configurations...")

    for method in methods:
        if method == 'NoAdapt':
            # 无自适应: 干净特征 → AWGN → 分类头
            for snr in tqdm(SNRS, desc=f'{method}'):
                for rate_ratio in RATE_RATIOS:
                    rate = max(1, int(SA3_C * rate_ratio))
                    for rep in range(N_REPEATS):
                        with torch.no_grad():
                            x_noisy = awgn_channel(feats_gpu, snr)  # (N, 1024, 1)
                            preds = classify_features(model, x_noisy)
                        acc = (preds == labels_all.to(DEVICE)).float().mean().item()
                        results.append({
                            'method': method, 'snr': snr, 'rate_ratio': rate_ratio,
                            'rate': rate, 'repeat': rep,
                            'accuracy_clean': acc_clean, 'accuracy_noisy': acc,
                            'accuracy_gap': acc_clean - acc,
                        })

        elif method == 'SA':
            sa_net = weights['SA']['mod_net']
            decoder = weights['SA']['decoder']
            # 预计算每个 SNR 的 SA 调制输出（不依赖 rate/repeat）
            sa_cache = {}
            for snr in tqdm(SNRS, desc=f'{method} SA cache', leave=False):
                with torch.no_grad():
                    sa_cache[snr] = sa_net(feats_mod_format, snr)  # (N, 1, 1024)

            for snr in tqdm(SNRS, desc=f'{method}'):
                for rate_ratio in RATE_RATIOS:
                    rate = max(1, int(SA3_C * rate_ratio))
                    for rep in range(N_REPEATS):
                        with torch.no_grad():
                            x_t = sa_cache[snr].transpose(1, 2)  # (N, 1024, 1)
                            x_noisy = awgn_channel(x_t, snr)
                            x_recon = decoder(x_noisy)             # (N, 1024, 1)
                            preds = classify_features(model, x_recon)
                        acc = (preds == labels_all.to(DEVICE)).float().mean().item()
                        results.append({
                            'method': method, 'snr': snr, 'rate_ratio': rate_ratio,
                            'rate': rate, 'repeat': rep,
                            'accuracy_clean': acc_clean, 'accuracy_noisy': acc,
                            'accuracy_gap': acc_clean - acc,
                        })

        elif method == 'RA':
            ra_net = weights['RA']['mod_net']
            decoder = weights['RA']['decoder']
            # 预计算每个 rate 的 RA 调制输出（RA 不依赖 SNR）
            ra_cache = {}
            for rr in tqdm(RATE_RATIOS, desc=f'{method} RA cache', leave=False):
                rate = max(1, int(SA3_C * rr))
                with torch.no_grad():
                    ra_cache[rr] = ra_net(feats_mod_format, rate)  # (x, mask)

            for snr in tqdm(SNRS, desc=f'{method}'):
                for rate_ratio in RATE_RATIOS:
                    rate = max(1, int(SA3_C * rate_ratio))
                    x_ra, mask = ra_cache[rate_ratio]
                    for rep in range(N_REPEATS):
                        with torch.no_grad():
                            x_t = x_ra.transpose(1, 2)          # (N, 1024, 1)
                            x_noisy = awgn_channel(x_t, snr)
                            x_recon = decoder(x_noisy)
                            preds = classify_features(model, x_recon)
                        acc = (preds == labels_all.to(DEVICE)).float().mean().item()
                        results.append({
                            'method': method, 'snr': snr, 'rate_ratio': rate_ratio,
                            'rate': rate, 'repeat': rep,
                            'accuracy_clean': acc_clean, 'accuracy_noisy': acc,
                            'accuracy_gap': acc_clean - acc,
                        })

        elif method == 'SA+RA':
            sa_net = weights['SA+RA']['sa_net']
            ra_net = weights['SA+RA']['ra_net']
            decoder = weights['SA+RA']['decoder']

            # 预计算 SA(每个SNR)
            sa_cache = {}
            for snr in tqdm(SNRS, desc=f'{method} SA cache', leave=False):
                with torch.no_grad():
                    sa_cache[snr] = sa_net(feats_mod_format, snr)

            # 预计算 RA(每个SNR,rate) 在 SA 输出上
            ra_cache = {}
            for snr in tqdm(SNRS, desc=f'{method} RA cache', leave=False):
                for rr in RATE_RATIOS:
                    rate = max(1, int(SA3_C * rr))
                    with torch.no_grad():
                        ra_cache[(snr, rr)] = ra_net(sa_cache[snr], rate)

            for snr in tqdm(SNRS, desc=f'{method}'):
                for rate_ratio in RATE_RATIOS:
                    rate = max(1, int(SA3_C * rate_ratio))
                    x_sara, mask = ra_cache[(snr, rate_ratio)]
                    for rep in range(N_REPEATS):
                        with torch.no_grad():
                            x_t = x_sara.transpose(1, 2)
                            x_noisy = awgn_channel(x_t, snr)
                            x_recon = decoder(x_noisy)
                            preds = classify_features(model, x_recon)
                        acc = (preds == labels_all.to(DEVICE)).float().mean().item()
                        results.append({
                            'method': method, 'snr': snr, 'rate_ratio': rate_ratio,
                            'rate': rate, 'repeat': rep,
                            'accuracy_clean': acc_clean, 'accuracy_noisy': acc,
                            'accuracy_gap': acc_clean - acc,
                        })

    # ---- 5. 保存 ----
    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/classification_accuracy.csv', index=False)
    print(f"\nSaved: results/classification_accuracy.csv ({len(df)} rows)")

    # ---- 6. 快速摘要 ----
    print(f"\n{'='*70}")
    print("Quick Summary — SA3 Classification Accuracy")
    print(f"Clean accuracy (upper bound): {acc_clean:.4f}")

    for method in methods:
        sub = df[df['method'] == method]
        for rr in [0.5, 1.0]:
            for snr in [0, 10, 20]:
                acc = sub[(sub['snr']==snr)&(sub['rate_ratio']==rr)]['accuracy_noisy'].mean()
                print(f"  {method:<8} SNR={snr:2d}dB rate={rr}: acc={acc:.4f}")

    print(f"\nDone. Next: run experiments/analysis/plot_classification.py to generate figures.")


if __name__ == '__main__':
    main()
