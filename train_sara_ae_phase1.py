# train_sara_ae_phase1.py
# SA+RA 联合训练 阶段1：解码器预热
# SA 和 RA 权重冻结，只训练新解码器学会"看懂" SA→RA 串联调制
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
EPOCHS = 50           # 阶段1只需解码器预热，epoch少一些
LR = 1e-3
SNR_MIN = 0
SNR_MAX = 20
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]

# ========== 加载数据 ==========
clean = np.load(f'results/clean_features_{FEAT_LAYER}.npy')
clean = clean[:2000]
N_pts = clean.shape[2]
C = clean.shape[1]
x_np = clean.transpose(0, 2, 1)       # (B, N_pts, C)
x_tensor = torch.from_numpy(x_np).float().to(DEVICE)
dataset = TensorDataset(x_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Device: {DEVICE}")
print(f"Layer: {FEAT_LAYER}, C={C}, N={N_pts}")
print(f"SNR range: [{SNR_MIN}, {SNR_MAX}], Rate ratios: {RATE_RATIOS}")
print(f"Samples: {len(dataset)}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

# ========== 加载预训练 SA（冻结） ==========
sa_net = ChannelModNet(feat_dim=C, hidden_dim=int(C * 1.5), num_layers=7).to(DEVICE)
sa_path = f'pretrained/sa_net_{FEAT_LAYER}_trained.pth'
if os.path.exists(sa_path):
    sa_net.load_state_dict(torch.load(sa_path, map_location=DEVICE))
    print(f"[OK] Loaded frozen SA: {sa_path}")
else:
    print(f"[FAIL] {sa_path} not found"); exit(1)
for p in sa_net.parameters():
    p.requires_grad = False
sa_net.eval()

# ========== 加载预训练 RA（冻结） ==========
ra_net = RateModNet(feat_dim=C, hidden_dim=int(C * 1.5), num_layers=7).to(DEVICE)
ra_path = f'pretrained/ra_net_{FEAT_LAYER}_trained.pth'
if os.path.exists(ra_path):
    ra_net.load_state_dict(torch.load(ra_path, map_location=DEVICE))
    print(f"[OK] Loaded frozen RA: {ra_path}")
else:
    print(f"[FAIL] {ra_path} not found"); exit(1)
for p in ra_net.parameters():
    p.requires_grad = False
ra_net.eval()

# ========== 解码器（3层，比单模块解码器更深） ==========
class DeeperDecoder(nn.Module):
    """3层MLP解码器，理解SA→RA串联调制的复杂统计特性"""
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
        # x: (B, C, N_pts) -> transpose -> Linear stack -> transpose back
        x = x.transpose(1, 2)            # (B, N_pts, C)
        x = self.net(x)                  # (B, N_pts, C)
        return x.transpose(1, 2)         # (B, C, N_pts)

decoder = DeeperDecoder(C).to(DEVICE)
print(f"Decoder: {C} -> 512 -> 256 -> {C}  (trainable params: {sum(p.numel() for p in decoder.parameters()):,})")

optimizer = optim.Adam(decoder.parameters(), lr=LR)
criterion = nn.MSELoss()

# ========== AWGN 信道 ==========
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

# ========== 训练 ==========
print("\nPhase 1: Decoder Warmup (SA + RA frozen)...")
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for x, in dataloader:
        # x: (B, N_pts, C)
        B = x.shape[0]

        # 随机采样 SNR 和 rate
        rate_ratio = np.random.choice(RATE_RATIOS)
        rate = max(1, int(C * rate_ratio))
        snr = torch.empty(B).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)

        with torch.no_grad():  # SA 和 RA 不产生梯度
            # 1. SA 调制（SNR自适应）
            x_sa = sa_net(x, snr)                  # (B, N_pts, C)
            # 2. RA 调制（速率自适应通道选择）
            x_sara, mask = ra_net(x_sa, rate)      # (B, N_pts, C)

        # 3. 转置 → 信道 → 解码器（只有解码器产生梯度）
        x_t = x_sara.transpose(1, 2)               # (B, C, N_pts)
        x_noisy = awgn_channel(x_t, snr)            # (B, C, N_pts)
        x_recon = decoder(x_noisy)                  # (B, C, N_pts)

        # 4. 损失：重建 vs 原始干净特征
        loss = criterion(x_recon, x.transpose(1, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B

    avg_loss = total_loss / len(dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}, Loss: {avg_loss:.6f}")

# ========== 保存 ==========
os.makedirs('pretrained', exist_ok=True)
torch.save(decoder.state_dict(), f'pretrained/sara_decoder_{FEAT_LAYER}_phase1.pth')
print(f"\nPhase 1 done. Saved: pretrained/sara_decoder_{FEAT_LAYER}_phase1.pth")

# ========== 快速评价 ==========
print("\nQuick evaluation (MSE on train subset, rate=0.5):")
with torch.no_grad():
    x_eval = x_tensor[:200]
    for snr_test in [0, 5, 10, 15, 20]:
        x_sa = sa_net(x_eval, snr_test)
        x_sara, _ = ra_net(x_sa, int(C * 0.5))
        x_t = x_sara.transpose(1, 2)
        x_n = awgn_channel(x_t, snr_test)
        x_r = decoder(x_n)
        mse = float(nn.MSELoss()(x_r, x_eval.transpose(1, 2)))
        print(f"  SNR={snr_test:2d}dB: MSE={mse:.6f}")
