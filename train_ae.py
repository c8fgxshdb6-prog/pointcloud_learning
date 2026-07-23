# train_ae.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append('./experiments/adapters')
from swin_adaptive_modules import ChannelModNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_LAYER = 'sa1'
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
SNR_MIN = 0
SNR_MAX = 20

# 加载数据
clean = np.load(f'results/clean_features_{FEAT_LAYER}.npy')
clean = clean[:2000]                # 取前2000个样本加快训练
N_pts = clean.shape[2]              # 点数
C = clean.shape[1]                  # 通道数
# 转换为 (N, N_pts, C) 格式
x_np = clean.transpose(0, 2, 1)    # (N, N_pts, C)
x_tensor = torch.from_numpy(x_np).float().to(DEVICE)
dataset = TensorDataset(x_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义 SA 模块
sa_net = ChannelModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(DEVICE)

# 定义解码器：输入 (B, C, N_pts) -> 输出 (B, C, N_pts)
class Decoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feat_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (B, C, N_pts) -> (B, N_pts, C)
        x = x.transpose(1, 2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)

decoder = Decoder(C).to(DEVICE)

optimizer = optim.Adam(list(sa_net.parameters()) + list(decoder.parameters()), lr=LR)
criterion = nn.MSELoss()

def awgn_channel(features, snr_db):
    # features: (B, C, N_pts)
    if isinstance(snr_db, (int, float)):
        snr_tensor = torch.full((features.shape[0], 1), snr_db, device=features.device)
    else:
        snr_tensor = snr_db.view(-1, 1).float()
    signal_power = torch.mean(features ** 2, dim=(1,2), keepdim=True)
    snr_linear = 10 ** (snr_tensor / 10.0)
    noise_power = signal_power / snr_linear.unsqueeze(-1)
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise

print("开始训练...")
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    for x, in dataloader:
        # x: (B, N_pts, C)
        snr = torch.empty(x.shape[0]).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)
        # 自适应调制
        x_mod = sa_net(x, snr)                 # (B, N_pts, C)
        x_mod_t = x_mod.transpose(1, 2)        # (B, C, N_pts)
        # 通过信道
        x_noisy = awgn_channel(x_mod_t, snr)   # (B, C, N_pts)
        # 解码重建
        x_recon = decoder(x_noisy)             # (B, C, N_pts)
        # 损失：重建特征与原始特征（注意原始 x 是 (B, N_pts, C) 格式，需要转置）
        loss = criterion(x_recon, x.transpose(1, 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}, Loss: {avg_loss:.6f}")

# 保存训练好的 SA 模块
os.makedirs('pretrained', exist_ok=True)
torch.save(sa_net.state_dict(), f'pretrained/sa_net_{FEAT_LAYER}_trained.pth')
print("训练完成，模型已保存。")
torch.save(decoder.state_dict(), f'pretrained/decoder_{FEAT_LAYER}.pth')