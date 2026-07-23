# train_phaseB_cls_decoder.py
# Phase B: 纯分类导向解码器训练
# SA 冻结 + RA 冻结 + 分类头冻结，解码器从随机初始化开始
# loss = NLLLoss(classify(decoder(SA→RA→AWGN→decoder)), label)  ← 只有 CE，没有 MSE
#
# 保留所有已有成果：SA/RA 权重不变，新解码器保存到新文件

import sys, os
sys.path.append('./Pointnet_Pointnet2_pytorch')
sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import glob
from swin_adaptive_modules import ChannelModNet, RateModNet

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'D:/Users/yxf/Desktop/pointcloud_learning/data/modelnet40_normal_resampled/test'
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3                  # 从随机初始化开始，可以用较大学习率
SNR_MIN, SNR_MAX = 0, 20
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]
N_TRAIN = 2000
SA3_C = 1024

print(f"Phase B: Classification-Only Decoder Training")
print(f"Device: {DEVICE}, LR: {LR}, Epochs: {EPOCHS}")
print(f"Rate ratios: {RATE_RATIOS}, SNR: [{SNR_MIN},{SNR_MAX}]")

# ============================================================
# 数据
# ============================================================
features_all = np.load('results/clean_features_sa3.npy')[:N_TRAIN]

class ModelNet40PLY(Dataset):
    def __init__(s, root, num_points=1024):
        s.root=root; s.num_points=num_points
        s.classes=sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        s.c2i={c:i for i,c in enumerate(s.classes)}; s.p=[]; s.l=[]
        for c in s.classes:
            for f in glob.glob(os.path.join(root,c,'*.txt')):
                s.p.append(f); s.l.append(s.c2i[c])
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        d=np.loadtxt(s.p[i],dtype=np.float32,delimiter=',')
        c=np.random.choice(d.shape[0],s.num_points,replace=d.shape[0]<s.num_points)
        return d[c,:], s.l[i]

dataset = ModelNet40PLY(DATA_ROOT, 1024)
labels_all = torch.tensor([dataset[i][1] for i in range(N_TRAIN)], dtype=torch.long)

feats_for_mod = torch.from_numpy(features_all).float().transpose(1, 2)  # (N, 1, 1024)
train_dataset = TensorDataset(feats_for_mod, labels_all)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# SA + RA (冻结) — 保留现有成果
# ============================================================
sa_net = ChannelModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
sa_net.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth', map_location=DEVICE))
for p in sa_net.parameters(): p.requires_grad = False
sa_net.eval()
print(f"[OK] SA frozen (from sara_sa_net_sa3.pth)")

ra_net = RateModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
ra_net.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth', map_location=DEVICE))
for p in ra_net.parameters(): p.requires_grad = False
ra_net.eval()
print(f"[OK] RA frozen (from sara_ra_net_sa3.pth)")

# ============================================================
# 解码器（随机初始化，可训练）
# ============================================================
class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1024),
        )
    def forward(s, x):
        x = x.transpose(1, 2)   # (B,C,N) → (B,N,C)
        x = s.net(x)            # (B,N,C) → (B,N,C)
        return x.transpose(1, 2) # (B,N,C) → (B,C,N)

decoder = Decoder().to(DEVICE)
n_dec = sum(p.numel() for p in decoder.parameters())
print(f"[OK] Decoder random init: {n_dec:,} trainable params")

# ============================================================
# 分类头（冻结）— 从 PointNet++ 提取
# ============================================================
from pointnet2_cls_msg import get_model
pn2 = get_model(40, True).to(DEVICE)
ck = torch.load(
    'Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth',
    map_location=DEVICE, weights_only=False,
)
pn2.load_state_dict(ck.get('model_state_dict', ck))
for p in pn2.parameters(): p.requires_grad = False
pn2.eval()
print("[OK] Classifier head frozen")

def classify(feat):
    x = feat.squeeze(-1)
    x = pn2.drop1(F.relu(pn2.bn1(pn2.fc1(x))))
    x = pn2.drop2(F.relu(pn2.bn2(pn2.fc2(x))))
    x = pn2.fc3(x)
    return F.log_softmax(x, -1)

# ============================================================
# AWGN
# ============================================================
def awgn(f, s):
    if isinstance(s, (int, float)):
        s = torch.full((f.shape[0], 1), s, device=f.device)
    else: s = s.view(-1, 1).float()
    sp = torch.mean(f**2, dim=(1, 2), keepdim=True)
    return f + torch.sqrt(sp / (10**(s/10.0)).unsqueeze(-1)) * torch.randn_like(f)

# ============================================================
# 优化器 + 损失（只有 NLL，没有 MSE）
# ============================================================
optimizer = optim.Adam(decoder.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
nll_loss = nn.NLLLoss()

# ============================================================
# 训练前基线：随机初始化解码器的分类精度
# ============================================================
x_val = feats_for_mod[:200].to(DEVICE)
lbl_val = labels_all[:200].to(DEVICE)

print("\nPre-training (random decoder) baseline:")
decoder.eval()
with torch.no_grad():
    for snr_t in [0, 5, 10, 15, 20]:
        xs = sa_net(x_val, snr_t)
        xsr, _ = ra_net(xs, int(SA3_C * 0.5))
        xn = awgn(xsr.transpose(1, 2), snr_t)
        xr = decoder(xn)
        acc = (classify(xr).argmax(-1) == lbl_val).float().mean().item()
        print(f"  SNR={snr_t:2d}dB: Acc={acc:.4f}")

# ============================================================
# 训练
# ============================================================
print(f"\nPhase B Training (CE-only, no MSE)...")
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    decoder.train()
    total_ce, total_acc = 0.0, 0.0

    for x_mod, labels in train_loader:
        B = x_mod.shape[0]
        x_mod = x_mod.to(DEVICE)
        labels = labels.to(DEVICE)

        rr = np.random.choice(RATE_RATIOS)
        rate = max(1, int(SA3_C * rr))
        snr = torch.empty(B).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)

        with torch.no_grad():
            x_sa = sa_net(x_mod, snr)
            x_sara, _ = ra_net(x_sa, rate)
            x_noisy = awgn(x_sara.transpose(1, 2), snr)

        x_recon = decoder(x_noisy)
        log_probs = classify(x_recon)
        loss = nll_loss(log_probs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (log_probs.argmax(-1) == labels).float().mean().item()
        total_ce += loss.item() * B
        total_acc += acc * B

    scheduler.step()

    if epoch % 10 == 0:
        # 验证集评估
        decoder.eval()
        with torch.no_grad():
            xs = sa_net(x_val, 10)
            xsr, _ = ra_net(xs, int(SA3_C * 0.5))
            xn = awgn(xsr.transpose(1, 2), 10)
            xr = decoder(xn)
            val_acc = (classify(xr).argmax(-1) == lbl_val).float().mean().item()

            # 保存最佳
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(decoder.state_dict(), 'pretrained/phaseB_decoder_sa3_best.pth')

        decoder.train()
        print(f"Epoch {epoch:3d}/{EPOCHS}: "
              f"CE={total_ce/N_TRAIN:.4f} TrainAcc={total_acc/N_TRAIN:.4f} "
              f"Val@10dB={val_acc:.4f} (best={best_acc:.4f}) "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

# ============================================================
# 保存
# ============================================================
torch.save(decoder.state_dict(), 'pretrained/phaseB_decoder_sa3.pth')
print(f"\nSaved: pretrained/phaseB_decoder_sa3.pth (best: pretrained/phaseB_decoder_sa3_best.pth)")

# ============================================================
# 最终评估 vs MSE 解码器 vs NoAdapt vs SA
# ============================================================
decoder.eval()

# 加载 MSE 解码器做对比
mse_decoder = Decoder().to(DEVICE)
mse_decoder.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth', map_location=DEVICE))
mse_decoder.eval()

print(f"\n{'='*75}")
print(f"Phase B Complete — Full SNR Comparison (rate=0.5, 200 val samples)")
print(f"{'SNR':<6} {'NoAdapt':<10} {'SA-MSE':<10} {'SA+RA-MSE':<12} {'SA+RA-CE':<12}")
print(f"{'':6} {'(no dec)':<10} {'(SA dec)':<10} {'(SARA dec)':<12} {'(Phase B)':<12}")
print("-" * 75)

with torch.no_grad():
    for snr_t in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        xs = sa_net(x_val, snr_t)
        xsr, _ = ra_net(xs, int(SA3_C * 0.5))
        xn = awgn(xsr.transpose(1, 2), snr_t)

        # NoAdapt: 直接加噪声到干净特征
        xn_clean = awgn(x_val.transpose(1, 2), snr_t)

        no_adapt_acc = (classify(xn_clean).argmax(-1) == lbl_val).float().mean().item()

        # SA + SA_decoder
        sa_only_x = sa_net(x_val, snr_t)
        sa_direct_n = awgn(sa_only_x.transpose(1, 2), snr_t)
        # SA decoder 2层
        sa_dec2 = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 1024)
        ).to(DEVICE)
        sa_dec2.load_state_dict(torch.load('pretrained/decoder_sa3.pth', map_location=DEVICE))
        sa_dec2.eval()
        sa_recon = sa_dec2(sa_direct_n.transpose(1, 2)).transpose(1, 2)
        sa_mse_acc = (classify(sa_recon).argmax(-1) == lbl_val).float().mean().item()

        # SA+RA + MSE decoder
        mse_recon = mse_decoder(xn)
        mse_acc = (classify(mse_recon).argmax(-1) == lbl_val).float().mean().item()

        # SA+RA + CE decoder
        ce_recon = decoder(xn)
        ce_acc = (classify(ce_recon).argmax(-1) == lbl_val).float().mean().item()

        best = max(no_adapt_acc, sa_mse_acc, mse_acc, ce_acc)
        stars = ['*' if a == best else ' ' for a in [no_adapt_acc, sa_mse_acc, mse_acc, ce_acc]]

        print(f"{snr_t:<6} {no_adapt_acc:<10.4f} {sa_mse_acc:<10.4f} "
              f"{mse_acc:<12.4f} {ce_acc:<12.4f}")

print("\nNext: run eval_phaseB.py on full 12311 test set for final numbers.")
