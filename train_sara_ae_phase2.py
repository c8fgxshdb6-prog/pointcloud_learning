# train_sara_ae_phase2.py
# SA+RA 联合训练 阶段2：联合微调
# 加载阶段1的所有权重，小学习率联合微调全部参数
# 先在 SA2 层验证

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append('./experiments/adapters')
from swin_adaptive_modules import ChannelModNet, RateModNet

# ========== 配置 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_LAYER = 'sa2'
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4               # 比阶段1低10倍，避免破坏已有能力
SNR_MIN = 0
SNR_MAX = 20
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]

# ========== 加载数据 ==========
clean = np.load(f'results/clean_features_{FEAT_LAYER}.npy')
clean = clean[:2000]
N_pts = clean.shape[2]
C = clean.shape[1]
x_np = clean.transpose(0, 2, 1)
x_tensor = torch.from_numpy(x_np).float().to(DEVICE)
dataset = TensorDataset(x_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Device: {DEVICE}")
print(f"Layer: {FEAT_LAYER}, C={C}, N={N_pts}")
print(f"SNR: [{SNR_MIN},{SNR_MAX}], Rates: {RATE_RATIOS}")
print(f"Samples: {len(dataset)}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")

# ========== 加载阶段1的 SA 权重（可训练） ==========
sa_net = ChannelModNet(feat_dim=C, hidden_dim=int(C * 1.5), num_layers=7).to(DEVICE)
sa_path = f'pretrained/sa_net_{FEAT_LAYER}_trained.pth'
if os.path.exists(sa_path):
    sa_net.load_state_dict(torch.load(sa_path, map_location=DEVICE))
    print(f"[OK] Loaded SA: {sa_path}")
else:
    print(f"[FAIL] {sa_path} not found"); exit(1)

# ========== 加载阶段1的 RA 权重（可训练） ==========
ra_net = RateModNet(feat_dim=C, hidden_dim=int(C * 1.5), num_layers=7).to(DEVICE)
ra_path = f'pretrained/ra_net_{FEAT_LAYER}_trained.pth'
if os.path.exists(ra_path):
    ra_net.load_state_dict(torch.load(ra_path, map_location=DEVICE))
    print(f"[OK] Loaded RA: {ra_path}")
else:
    print(f"[FAIL] {ra_path} not found"); exit(1)

# ========== 加载阶段1的解码器权重（可训练） ==========
class DeeperDecoder(nn.Module):
    def __init__(self, feat_dim, hidden1=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, feat_dim),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)

decoder = DeeperDecoder(C).to(DEVICE)
dec_path = f'pretrained/sara_decoder_{FEAT_LAYER}_phase1.pth'
if os.path.exists(dec_path):
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE))
    print(f"[OK] Loaded decoder: {dec_path}")
else:
    print(f"[FAIL] {dec_path} not found"); exit(1)

# 统计参数量
n_sa = sum(p.numel() for p in sa_net.parameters())
n_ra = sum(p.numel() for p in ra_net.parameters())
n_dec = sum(p.numel() for p in decoder.parameters())
print(f"Params: SA={n_sa:,}, RA={n_ra:,}, Decoder={n_dec:,}, Total={n_sa+n_ra+n_dec:,}")

# ========== 优化器（全部参数，小学习率） ==========
optimizer = optim.Adam(
    list(sa_net.parameters()) + list(ra_net.parameters()) + list(decoder.parameters()),
    lr=LR
)
criterion = nn.MSELoss()

# ========== AWGN ==========
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

# ========== 保存阶段1的解码器基准评估 ==========
print("\nPhase 1 baseline (before fine-tuning):")
with torch.no_grad():
    x_eval = x_tensor[:200]
    phase1_mse = {}
    for snr_test in [0, 5, 10, 15, 20]:
        x_sa = sa_net(x_eval, snr_test)
        x_sara, _ = ra_net(x_sa, int(C * 0.5))
        x_t = x_sara.transpose(1, 2)
        x_n = awgn_channel(x_t, snr_test)
        x_r = decoder(x_n)
        mse = float(nn.MSELoss()(x_r, x_eval.transpose(1, 2)))
        phase1_mse[snr_test] = mse
        print(f"  SNR={snr_test:2d}dB: MSE={mse:.6f}")

# ========== 阶段2：联合微调 ==========
print("\nPhase 2: Joint Fine-tuning...")
best_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for x, in dataloader:
        B = x.shape[0]

        rate_ratio = np.random.choice(RATE_RATIOS)
        rate = max(1, int(C * rate_ratio))
        snr = torch.empty(B).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)

        # 全部模块都参与梯度计算
        x_sa = sa_net(x, snr)                    # (B, N_pts, C)
        x_sara, mask = ra_net(x_sa, rate)         # (B, N_pts, C)
        x_t = x_sara.transpose(1, 2)              # (B, C, N_pts)
        x_noisy = awgn_channel(x_t, snr)           # (B, C, N_pts)
        x_recon = decoder(x_noisy)                 # (B, C, N_pts)

        loss = criterion(x_recon, x.transpose(1, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B

    avg_loss = total_loss / len(dataset)
    if avg_loss < best_loss:
        best_loss = avg_loss

    if epoch % 10 == 0:
        # 实时评估
        with torch.no_grad():
            sa_net.eval(); ra_net.eval(); decoder.eval()
            x_sa = sa_net(x_eval, 10)
            x_sara, _ = ra_net(x_sa, int(C * 0.5))
            x_t = x_sara.transpose(1, 2)
            x_n = awgn_channel(x_t, 10)
            x_r = decoder(x_n)
            cur_mse = float(nn.MSELoss()(x_r, x_eval.transpose(1, 2)))
            sa_net.train(); ra_net.train(); decoder.train()
        print(f"Epoch {epoch:3d}/{EPOCHS}, Loss: {avg_loss:.6f}, "
              f"SNR=10dB MSE: {cur_mse:.6f} (best loss: {best_loss:.6f})")

# ========== 保存 ==========
os.makedirs('pretrained', exist_ok=True)
torch.save(sa_net.state_dict(), f'pretrained/sara_sa_net_{FEAT_LAYER}.pth')
torch.save(ra_net.state_dict(), f'pretrained/sara_ra_net_{FEAT_LAYER}.pth')
torch.save(decoder.state_dict(), f'pretrained/sara_decoder_{FEAT_LAYER}.pth')
print(f"\nPhase 2 done. Saved:")
print(f"  pretrained/sara_sa_net_{FEAT_LAYER}.pth")
print(f"  pretrained/sara_ra_net_{FEAT_LAYER}.pth")
print(f"  pretrained/sara_decoder_{FEAT_LAYER}.pth")

# ========== 最终评估：微调前后对比 ==========
print("\n" + "=" * 65)
print("Phase 1 vs Phase 2 comparison (rate=0.5)")
print(f"{'SNR':<6} {'Phase1 MSE':<14} {'Phase2 MSE':<14} {'Improvement':<12}")
print("-" * 65)
sa_net.eval(); ra_net.eval(); decoder.eval()
with torch.no_grad():
    for snr_test in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        x_sa = sa_net(x_eval, snr_test)
        x_sara, _ = ra_net(x_sa, int(C * 0.5))
        x_t = x_sara.transpose(1, 2)
        x_n = awgn_channel(x_t, snr_test)
        x_r = decoder(x_n)
        p2_mse = float(nn.MSELoss()(x_r, x_eval.transpose(1, 2)))
        p1_mse = phase1_mse.get(snr_test, float('nan'))
        improv = (1 - p2_mse / p1_mse) * 100 if p1_mse > 0 else 0
        print(f"{snr_test:<6} {p1_mse:<14.6f} {p2_mse:<14.6f} {improv:<+.1f}%")
