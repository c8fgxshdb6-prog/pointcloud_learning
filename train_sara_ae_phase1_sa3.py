# train_sara_ae_phase1_sa3.py - SA3 层阶段1
import torch, torch.nn as nn, torch.optim as optim, numpy as np, sys, os
from torch.utils.data import DataLoader, TensorDataset
sys.path.append('./experiments/adapters')
from swin_adaptive_modules import ChannelModNet, RateModNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_LAYER = 'sa3'
BATCH_SIZE = 128
EPOCHS = 50; LR = 1e-3
SNR_MIN, SNR_MAX = 0, 20
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]

clean = np.load(f'results/clean_features_{FEAT_LAYER}.npy'); clean = clean[:2000]
N_pts, C = clean.shape[2], clean.shape[1]  # sa3: C=1024, N=1
x_np = clean.transpose(0, 2, 1)
x_tensor = torch.from_numpy(x_np).float().to(DEVICE)
dataset = TensorDataset(x_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Layer: {FEAT_LAYER}, C={C}, N={N_pts}, Epochs: {EPOCHS}")

sa_net = ChannelModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(DEVICE)
sa_net.load_state_dict(torch.load(f'pretrained/sa_net_{FEAT_LAYER}_trained.pth', map_location=DEVICE))
for p in sa_net.parameters(): p.requires_grad = False
sa_net.eval()
print(f"[OK] Frozen SA")

ra_net = RateModNet(feat_dim=C, hidden_dim=int(C*1.5), num_layers=7).to(DEVICE)
ra_net.load_state_dict(torch.load(f'pretrained/ra_net_{FEAT_LAYER}_trained.pth', map_location=DEVICE))
for p in ra_net.parameters(): p.requires_grad = False
ra_net.eval()
print(f"[OK] Frozen RA")

class DeeperDecoder(nn.Module):
    def __init__(self, feat_dim, hidden1=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden1), nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, feat_dim))
    def forward(self, x): x = x.transpose(1, 2); x = self.net(x); return x.transpose(1, 2)

decoder = DeeperDecoder(C).to(DEVICE)
optimizer = optim.Adam(decoder.parameters(), lr=LR)
criterion = nn.MSELoss()

def awgn_channel(features, snr_db):
    if isinstance(snr_db, (int, float)): snr_tensor = torch.full((features.shape[0], 1), snr_db, device=features.device)
    else: snr_tensor = snr_db.view(-1, 1).float()
    sp = torch.mean(features**2, dim=(1,2), keepdim=True)
    nl = 10**(snr_tensor/10.0); npwr = sp / nl.unsqueeze(-1)
    return features + torch.sqrt(npwr) * torch.randn_like(features)

for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    for x, in dataloader:
        B = x.shape[0]; rate = max(1, int(C * np.random.choice(RATE_RATIOS)))
        snr = torch.empty(B).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)
        with torch.no_grad():
            x_sa = sa_net(x, snr); x_sara, _ = ra_net(x_sa, rate)
        x_t = x_sara.transpose(1, 2); x_noisy = awgn_channel(x_t, snr); x_recon = decoder(x_noisy)
        loss = criterion(x_recon, x.transpose(1, 2))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * B
    if epoch % 10 == 0: print(f"Epoch {epoch:3d}/{EPOCHS}, Loss: {total_loss/len(dataset):.6f}")

os.makedirs('pretrained', exist_ok=True)
torch.save(decoder.state_dict(), f'pretrained/sara_decoder_{FEAT_LAYER}_phase1.pth')
print(f"Done: pretrained/sara_decoder_{FEAT_LAYER}_phase1.pth")
