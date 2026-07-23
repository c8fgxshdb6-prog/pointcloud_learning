# train_sara_ae_phase2_sa3_task_full.py
# Phase A2: 全部解冻联合微调 (SA+RA+Decoder)
# loss = MSE(x_recon, x_clean) + lambda * NLL(classify(x_recon), label)

import sys, os
sys.path.append('./Pointnet_Pointnet2_pytorch')
sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import glob
from swin_adaptive_modules import ChannelModNet, RateModNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'D:/Users/yxf/Desktop/pointcloud_learning/data/modelnet40_normal_resampled/test'
BATCH_SIZE = 64; EPOCHS = 60; LR = 1e-5; LAMBDA_CLS = 0.01
SNR_MIN, SNR_MAX = 0, 20; RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]
N_TRAIN = 2000; SA3_C = 1024

print(f"Phase A2: Full fine-tuning (SA+RA+Decoder), lambda={LAMBDA_CLS}, LR={LR}")

# -- Data --
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

feats_for_mod = torch.from_numpy(features_all).float().transpose(1, 2)  # (N,1,1024)
train_dataset = TensorDataset(feats_for_mod, labels_all)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -- Load models (all trainable) --
sa_net = ChannelModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
sa_net.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth', map_location=DEVICE))
print(f"[OK] SA: {sum(p.numel() for p in sa_net.parameters()):,} params")

ra_net = RateModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
ra_net.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth', map_location=DEVICE))
print(f"[OK] RA: {sum(p.numel() for p in ra_net.parameters()):,} params")

class Dec(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1024))
    def forward(s,x): x=x.transpose(1,2); x=s.net(x); return x.transpose(1,2)

decoder = Dec().to(DEVICE)
decoder.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth', map_location=DEVICE))
print(f"[OK] Decoder: {sum(p.numel() for p in decoder.parameters()):,} params")
n_total = sum(p.numel() for m in [sa_net,ra_net,decoder] for p in m.parameters())
print(f"Total trainable: {n_total:,}")

# -- Classifier head (frozen) --
from pointnet2_cls_msg import get_model
pn2 = get_model(40, True).to(DEVICE)
ck = torch.load('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth', map_location=DEVICE, weights_only=False)
pn2.load_state_dict(ck.get('model_state_dict', ck))
for p in pn2.parameters(): p.requires_grad = False; pn2.eval()
print("[OK] Classifier frozen")

def classify(feat):
    x = feat.squeeze(-1)
    x = pn2.drop1(F.relu(pn2.bn1(pn2.fc1(x))))
    x = pn2.drop2(F.relu(pn2.bn2(pn2.fc2(x))))
    x = pn2.fc3(x); return F.log_softmax(x, -1)

# -- AWGN --
def awgn(f, s):
    if isinstance(s,(int,float)): s=torch.full((f.shape[0],1),s,device=f.device)
    else: s=s.view(-1,1).float()
    sp=torch.mean(f**2,dim=(1,2),keepdim=True)
    return f+torch.sqrt(sp/(10**(s/10.0)).unsqueeze(-1))*torch.randn_like(f)

# -- Optimizer --
optimizer = optim.Adam(
    list(sa_net.parameters())+list(ra_net.parameters())+list(decoder.parameters()), lr=LR)
mse_loss = nn.MSELoss(); nll_loss = nn.NLLLoss()

# -- Validation --
x_val = feats_for_mod[:200].to(DEVICE)
lbl_val = labels_all[:200].to(DEVICE); x_val_clean = x_val.transpose(1,2)

# -- Pre-finetune baseline --
print("\nPre-finetune:")
sa_net.eval(); ra_net.eval(); decoder.eval()
with torch.no_grad():
    for snr_t in [0,10,20]:
        xs=sa_net(x_val,snr_t); xsr,_=ra_net(xs,int(SA3_C*0.5))
        xn=awgn(xsr.transpose(1,2),snr_t); xr=decoder(xn)
        m=float(mse_loss(xr,x_val_clean)); a=(classify(xr).argmax(-1)==lbl_val).float().mean().item()
        print(f"  SNR={snr_t:2d}dB rate=0.5: MSE={m:.4f} Acc={a:.4f}")

# -- Training --
print(f"\nPhase A2 training (lambda_CE={LAMBDA_CLS})...")
for epoch in range(1, EPOCHS+1):
    sa_net.train(); ra_net.train(); decoder.train()
    tm, tc, tl = 0.0, 0.0, 0.0
    for x_mod, labels in train_loader:
        B=x_mod.shape[0]; x_mod=x_mod.to(DEVICE); x_clean=x_mod.transpose(1,2); labels=labels.to(DEVICE)
        rr=np.random.choice(RATE_RATIOS); rate=max(1,int(SA3_C*rr))
        snr=torch.empty(B).uniform_(SNR_MIN,SNR_MAX).to(DEVICE)
        xs=sa_net(x_mod,snr); xsr,_=ra_net(xs,rate)
        xn=awgn(xsr.transpose(1,2),snr); xr=decoder(xn)
        lm=mse_loss(xr,x_clean); lc=nll_loss(classify(xr),labels); lt=lm+LAMBDA_CLS*lc
        optimizer.zero_grad(); lt.backward(); optimizer.step()
        tm+=lm.item()*B; tc+=lc.item()*B; tl+=lt.item()*B
    if epoch%5==0:
        sa_net.eval(); ra_net.eval(); decoder.eval()
        with torch.no_grad():
            xs=sa_net(x_val,10); xsr,_=ra_net(xs,int(SA3_C*0.5))
            xn=awgn(xsr.transpose(1,2),10); xr=decoder(xn)
            cm=float(mse_loss(xr,x_val_clean)); ca=(classify(xr).argmax(-1)==lbl_val).float().mean().item()
        print(f"Ep {epoch:3d}/{EPOCHS}: loss={tl/N_TRAIN:.4f} (MSE={tm/N_TRAIN:.4f} CE={tc/N_TRAIN:.4f}) val@10dB: MSE={cm:.4f} Acc={ca:.4f}")

# -- Save --
os.makedirs('pretrained', exist_ok=True)
torch.save(sa_net.state_dict(), 'pretrained/sara_sa_net_sa3_task.pth')
torch.save(ra_net.state_dict(), 'pretrained/sara_ra_net_sa3_task.pth')
torch.save(decoder.state_dict(), 'pretrained/sara_decoder_sa3_task_full.pth')
print(f"\nSaved: sara_*_sa3_task*.pth")

# -- Full SNR eval --
print(f"\n{'='*60}\nPhase A2: Full SNR (rate=0.5, 200 val samples)\n{'='*60}")
sa_net.eval(); ra_net.eval(); decoder.eval()
with torch.no_grad():
    for snr_t in [0,2,4,6,8,10,12,14,16,18,20]:
        xs=sa_net(x_val,snr_t); xsr,_=ra_net(xs,int(SA3_C*0.5))
        xn=awgn(xsr.transpose(1,2),snr_t); xr=decoder(xn)
        m=float(mse_loss(xr,x_val_clean)); a=(classify(xr).argmax(-1)==lbl_val).float().mean().item()
        print(f"  SNR={snr_t:2d}dB: MSE={m:.6f} Acc={a:.4f}")
print("Done.")
