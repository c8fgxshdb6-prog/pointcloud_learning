# train_sara_ae_phase2_sa3_task.py
# Phase A: SA+RA 解码器任务驱动微调 (SA3)
# 在 MSE 优化的解码器基础上，加入分类损失联合微调
# 冻结 SA + RA + 分类头，只训练解码器
#
# loss = MSE(x_recon, x_clean) + lambda * NLL(classify(x_recon), true_label)

import sys, os
sys.path.append('./Pointnet_Pointnet2_pytorch')
sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
EPOCHS = 30
LR = 1e-4
LAMBDA_CLS = 0.05          # 分类损失权重（可调）
SNR_MIN, SNR_MAX = 0, 20
RATE_RATIOS = [0.2, 0.5, 0.8, 1.0]
N_TRAIN = 2000              # 使用前2000个样本微调
SA3_C = 1024

print(f"Device: {DEVICE}")
print(f"Lambda (classification weight): {LAMBDA_CLS}")
print(f"Epochs: {EPOCHS}, LR: {LR}, Train samples: {N_TRAIN}")

# ============================================================
# 1. 加载 SA3 特征 + 对应标签
# ============================================================

# 特征（已按固定顺序提取，shuffle=False）
features_all = np.load('results/clean_features_sa3.npy')[:N_TRAIN]  # (N, 1024, 1)
features_tensor = torch.from_numpy(features_all).float()

# 标签：用与 extract_features.py 完全相同的 Dataset 类，确保顺序一致
class ModelNet40PLY(Dataset):
    def __init__(self, root, num_points=1024, use_normals=True):
        self.root = root; self.num_points = num_points; self.use_normals = use_normals
        self.classes = sorted([d for d in os.listdir(self.root)
                               if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.points = []; self.labels = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            files = glob.glob(os.path.join(cls_path, '*.txt'))
            for f in files:
                self.points.append(f); self.labels.append(self.class_to_idx[cls])
    def __len__(self): return len(self.points)
    def __getitem__(self, idx):
        data = np.loadtxt(self.points[idx], dtype=np.float32, delimiter=',')
        if data.shape[0] < self.num_points:
            choice = np.random.choice(data.shape[0], self.num_points, replace=True)
        else:
            choice = np.random.choice(data.shape[0], self.num_points, replace=False)
        data = data[choice, :]
        if not self.use_normals: data = data[:, :3]
        return data, self.labels[idx]

dataset = ModelNet40PLY(root=DATA_ROOT, num_points=1024, use_normals=True)
labels_all = torch.tensor([dataset[i][1] for i in range(N_TRAIN)], dtype=torch.long)
print(f"Features: {features_tensor.shape}, Labels: {labels_all.shape}")

# 转为训练 DataLoader
# 特征格式: (N, 1024, 1) → 解码器需要 (N, C, N_pts)
# SA/RM 模块需要 (N, N_pts, C), N_pts=1 for SA3
feats_for_mod = features_tensor.transpose(1, 2)  # (N, 1, 1024)

train_dataset = TensorDataset(feats_for_mod, feats_for_mod, labels_all)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 2. 加载 SA+RA 预训练权重（冻结）
# ============================================================
sa_net = ChannelModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
sa_net.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth', map_location=DEVICE))
for p in sa_net.parameters(): p.requires_grad = False
sa_net.eval()
print("[OK] SA frozen")

ra_net = RateModNet(SA3_C, int(SA3_C*1.5), 7).to(DEVICE)
ra_net.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth', map_location=DEVICE))
for p in ra_net.parameters(): p.requires_grad = False
ra_net.eval()
print("[OK] RA frozen")

# ============================================================
# 3. 解码器（可训练）
# ============================================================
class DeeperDecoder(nn.Module):
    def __init__(self, feat_dim, hidden1=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, feat_dim),
        )
    def forward(self, x):
        # x: (B, C, N_pts) → (B, N_pts, C) → Linear → (B, C, N_pts)
        x = x.transpose(1, 2); x = self.net(x); return x.transpose(1, 2)

decoder = DeeperDecoder(SA3_C).to(DEVICE)
decoder.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth', map_location=DEVICE))
print(f"[OK] Decoder loaded, trainable params: {sum(p.numel() for p in decoder.parameters()):,}")

# ============================================================
# 4. 分类头（冻结）— 从 PointNet++ 提取 fc1/fc2/fc3 权重
# ============================================================
from pointnet2_cls_msg import get_model
pn2_model = get_model(num_class=40, normal_channel=True).to(DEVICE)
ckpt = torch.load(
    'D:/Users/yxf/Desktop/pointcloud_learning/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth',
    map_location=DEVICE, weights_only=False,
)
if 'model_state_dict' in ckpt:
    pn2_model.load_state_dict(ckpt['model_state_dict'])
else:
    pn2_model.load_state_dict(ckpt)

# 提取分类头各层
class ClassifierHead(nn.Module):
    """PointNet++ 分类头的精确复现（用于解码器输出 1024维→40类）"""
    def __init__(self, src_model):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 40)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.5)

        # 复制权重
        self.fc1.weight.data.copy_(src_model.fc1.weight.data)
        self.fc1.bias.data.copy_(src_model.fc1.bias.data)
        self.bn1.weight.data.copy_(src_model.bn1.weight.data)
        self.bn1.bias.data.copy_(src_model.bn1.bias.data)
        self.bn1.running_mean.copy_(src_model.bn1.running_mean)
        self.bn1.running_var.copy_(src_model.bn1.running_var)
        self.fc2.weight.data.copy_(src_model.fc2.weight.data)
        self.fc2.bias.data.copy_(src_model.fc2.bias.data)
        self.bn2.weight.data.copy_(src_model.bn2.weight.data)
        self.bn2.bias.data.copy_(src_model.bn2.bias.data)
        self.bn2.running_mean.copy_(src_model.bn2.running_mean)
        self.bn2.running_var.copy_(src_model.bn2.running_var)
        self.fc3.weight.data.copy_(src_model.fc3.weight.data)
        self.fc3.bias.data.copy_(src_model.fc3.bias.data)

    def forward(self, x):
        # x: (B, 1024) 或 (B, 1024, 1)
        if x.dim() == 3:
            x = x.squeeze(-1)  # (B, 1024, 1) → (B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

cls_head = ClassifierHead(pn2_model).to(DEVICE)
for p in cls_head.parameters(): p.requires_grad = False
cls_head.eval()
print("[OK] Classifier head frozen")

# 释放 PointNet++ 模型显存
del pn2_model; torch.cuda.empty_cache()

# ============================================================
# 5. AWGN 信道
# ============================================================
def awgn_channel(f, s):
    if isinstance(s, (int, float)):
        s = torch.full((f.shape[0], 1), s, device=f.device)
    else: s = s.view(-1, 1).float()
    sp = torch.mean(f**2, dim=(1, 2), keepdim=True)
    return f + torch.sqrt(sp / (10**(s/10.0)).unsqueeze(-1)) * torch.randn_like(f)

# ============================================================
# 6. 先算一次微调前基线
# ============================================================
print("\nPre-finetune baseline (on training subset):")
with torch.no_grad():
    x_eval = feats_for_mod[:200].to(DEVICE)
    lbl_eval = labels_all[:200].to(DEVICE)
    for snr_test in [0, 10, 20]:
        x_sa = sa_net(x_eval, snr_test)
        x_sara, _ = ra_net(x_sa, int(SA3_C * 0.5))
        x_t = x_sara.transpose(1, 2)
        x_n = awgn_channel(x_t, snr_test)
        x_r = decoder(x_n)
        mse_v = float(F.mse_loss(x_r, feats_for_mod[:200].to(DEVICE).transpose(1,2)))
        preds = cls_head(x_r).argmax(dim=-1)
        acc_v = (preds == lbl_eval).float().mean().item()
        print(f"  SNR={snr_test:2d}dB rate=0.5: MSE={mse_v:.4f}, Acc={acc_v:.4f}")

# ============================================================
# 7. 微调
# ============================================================
optimizer = optim.Adam(decoder.parameters(), lr=LR)
mse_criterion = nn.MSELoss()
nll_criterion = nn.NLLLoss()  # 配合 log_softmax

print(f"\nPhase A: Task-Driven Fine-tuning (lambda={LAMBDA_CLS})...")
for epoch in range(1, EPOCHS + 1):
    total_mse, total_ce, total_loss = 0.0, 0.0, 0.0
    n_batches = 0

    for x_mod, x_clean_mod, labels in train_loader:
        # x_mod: (B, 1, 1024) — 用于调制模块
        # x_clean_mod: (B, 1, 1024) — 同上
        # labels: (B,)
        B = x_mod.shape[0]
        x_mod = x_mod.to(DEVICE)
        x_clean_mod = x_clean_mod.to(DEVICE)
        x_clean = x_clean_mod.transpose(1, 2)  # (B, 1024, 1)
        labels = labels.to(DEVICE)

        # 随机采样 SNR 和 rate
        rate_ratio = np.random.choice(RATE_RATIOS)
        rate = max(1, int(SA3_C * rate_ratio))
        snr = torch.empty(B).uniform_(SNR_MIN, SNR_MAX).to(DEVICE)

        # SA (frozen) → RA (frozen) → AWGN → Decoder (trainable)
        with torch.no_grad():
            x_sa = sa_net(x_mod, snr)
            x_sara, _ = ra_net(x_sa, rate)
            x_t = x_sara.transpose(1, 2)
            x_noisy = awgn_channel(x_t, snr)

        x_recon = decoder(x_noisy)  # (B, 1024, 1)

        # MSE 损失
        loss_mse = mse_criterion(x_recon, x_clean)

        # 分类损失
        log_probs = cls_head(x_recon)  # (B, 40) log_softmax
        loss_ce = nll_criterion(log_probs, labels)

        # 联合损失
        loss = loss_mse + LAMBDA_CLS * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse += loss_mse.item() * B
        total_ce += loss_ce.item() * B
        total_loss += loss.item() * B
        n_batches += 1

    if epoch % 5 == 0:
        # 实时评估
        decoder.eval()
        with torch.no_grad():
            x_sa = sa_net(x_eval, 10)
            x_sara, _ = ra_net(x_sa, int(SA3_C * 0.5))
            x_n = awgn_channel(x_sara.transpose(1,2), 10)
            x_r = decoder(x_n)
            cur_mse = float(mse_criterion(x_r, feats_for_mod[:200].to(DEVICE).transpose(1,2)))
            preds = cls_head(x_r).argmax(dim=-1)
            cur_acc = (preds == lbl_eval).float().mean().item()
        decoder.train()

        print(f"Epoch {epoch:3d}/{EPOCHS}: "
              f"loss={total_loss/N_TRAIN:.4f} "
              f"(MSE={total_mse/N_TRAIN:.4f}, CE={total_ce/N_TRAIN:.4f}), "
              f"val@10dB: MSE={cur_mse:.4f}, Acc={cur_acc:.4f}")

# ============================================================
# 8. 保存 + 最终对比
# ============================================================
os.makedirs('pretrained', exist_ok=True)
torch.save(decoder.state_dict(), 'pretrained/sara_decoder_sa3_task.pth')
print(f"\nSaved: pretrained/sara_decoder_sa3_task.pth")

print("\n" + "=" * 65)
print("Phase A Complete: Pre vs Post Fine-tuning (rate=0.5)")
print(f"{'SNR':<6} {'Pre MSE':<12} {'Post MSE':<10} {'Pre Acc':<10} {'Post Acc':<10}")
print("-" * 65)

# 加载微调前解码器做对比
decoder_pre = DeeperDecoder(SA3_C).to(DEVICE)
decoder_pre.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth', map_location=DEVICE))
decoder_pre.eval()
decoder.eval()

with torch.no_grad():
    for snr_test in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        x_sa = sa_net(x_eval, snr_test)
        x_sara, _ = ra_net(x_sa, int(SA3_C * 0.5))
        x_t = x_sara.transpose(1, 2)
        x_n = awgn_channel(x_t, snr_test)

        x_pre = decoder_pre(x_n)
        x_post = decoder(x_n)

        mse_pre = float(mse_criterion(x_pre, feats_for_mod[:200].to(DEVICE).transpose(1,2)))
        mse_post = float(mse_criterion(x_post, feats_for_mod[:200].to(DEVICE).transpose(1,2)))
        acc_pre = (cls_head(x_pre).argmax(-1) == lbl_eval).float().mean().item()
        acc_post = (cls_head(x_post).argmax(-1) == lbl_eval).float().mean().item()

        print(f"{snr_test:<6} {mse_pre:<12.6f} {mse_post:<10.6f} "
              f"{acc_pre:<10.4f} {acc_post:<10.4f}")

print("\nDone. To re-evaluate on full test set, run evaluate_classification.py ")
print("and update the SA+RA decoder path to pretrained/sara_decoder_sa3_task.pth")
