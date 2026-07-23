# train_ra_ae_sa1.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append('./experiments/adapters')
from swin_adaptive_modules import RateModNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_LAYER = 'sa1'
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
SNR_MIN = 0
SNR_MAX = 20
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]

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
print(f"Rate ratios: {RATE_RATIOS} -> channels: {[max(1, int(C*r)) for r in RATE_RATIOS]}")
print(f"SNR range: [{SNR_MIN}, {SNR_MAX}] dB")
print(f"Samples: {len(dataset)}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

ra_net = RateModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(DEVICE)

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

decoder = Decoder(C).to(DEVICE)
optimizer = optim.Adam(list(ra_net.parameters())+list(decoder.parameters()), lr=LR)
criterion = nn.MSELoss()

def awgn_channel(features, snr_db):
    if isinstance(snr_db, (int, float)):
        snr_tensor = torch.full((features.shape[0], 1), snr_db, device=features.device)
    else:
        snr_tensor = snr_db.view(-1, 1).float()
    signal_power = torch.mean(features**2, dim=(1,2), keepdim=True)
    snr_linear = 10**(snr_tensor/10.0)
    noise_power = signal_power / snr_linear.unsqueeze(-1)
    noise = torch.sqrt(noise_power) * torch.randn_like(features)
    return features + noise

print("\nTraining...")
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    for x, in dataloader:
        B = x.shape[0]
        rate_ratio = np.random.choice(RATE_RATIOS)
        rate = max(1, int(C*rate_ratio))
        snr = torch.empty(B).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)
        x_ra, mask = ra_net(x, rate)
        x_ra_t = x_ra.transpose(1, 2)
        x_noisy = awgn_channel(x_ra_t, snr)
        x_recon = decoder(x_noisy)
        loss = criterion(x_recon, x.transpose(1, 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
    avg_loss = total_loss / len(dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}, Loss: {avg_loss:.6f}")

os.makedirs('pretrained', exist_ok=True)
torch.save(ra_net.state_dict(), f'pretrained/ra_net_{FEAT_LAYER}_trained.pth')
torch.save(decoder.state_dict(), f'pretrained/ra_decoder_{FEAT_LAYER}.pth')
print(f"Done. Saved: ra_net_{FEAT_LAYER}_trained.pth, ra_decoder_{FEAT_LAYER}.pth")
