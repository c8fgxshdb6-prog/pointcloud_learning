# eval_phaseA.py — 全量测试集评估微调前后的 SA+RA 分类精度
import sys, os, torch, torch.nn as nn, numpy as np, pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('./Pointnet_Pointnet2_pytorch'); sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')
from swin_adaptive_modules import ChannelModNet, RateModNet

DEV = torch.device('cuda')
C = 1024; SNRS = [0,2,4,6,8,10,12,14,16,18,20]; RRS = [0.2,0.5,0.8,1.0]; NR = 5
MB = 400  # 微批次大小，SA3特征小所以可以大一些

# Labels
import glob
class DS:
    def __init__(s,r):
        cls = sorted([d for d in os.listdir(r) if os.path.isdir(os.path.join(r,d))])
        s.p=[]; s.l=[]
        for c in cls:
            for f in glob.glob(os.path.join(r,c,'*.txt')):
                s.p.append(f); s.l.append(cls.index(c))
    def __len__(s): return len(s.p)

labels_all = torch.tensor(DS('data/modelnet40_normal_resampled/test').l).to(DEV)
print(f'Total test samples: {len(labels_all)}')

# Models
sa = ChannelModNet(C,int(C*1.5),7).to(DEV); sa.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth',map_location=DEV)); sa.eval()
ra = RateModNet(C,int(C*1.5),7).to(DEV); ra.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth',map_location=DEV)); ra.eval()

class Dec(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1024))
    def forward(s,x): x=x.transpose(1,2); x=s.net(x); return x.transpose(1,2)

dec_pre = Dec().to(DEV); dec_pre.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth',map_location=DEV)); dec_pre.eval()
dec_post = Dec().to(DEV); dec_post.load_state_dict(torch.load('pretrained/sara_decoder_sa3_task.pth',map_location=DEV)); dec_post.eval()

from pointnet2_cls_msg import get_model
m2 = get_model(40,True).to(DEV)
ck = torch.load('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth',map_location=DEV,weights_only=False)
m2.load_state_dict(ck.get('model_state_dict',ck)); m2.eval()
for p in m2.parameters(): p.requires_grad = False

def classify(feat):
    x = feat.squeeze(-1)
    x = m2.drop1(F.relu(m2.bn1(m2.fc1(x))))
    x = m2.drop2(F.relu(m2.bn2(m2.fc2(x))))
    x = m2.fc3(x); return F.log_softmax(x,-1).argmax(-1)

def awgn(f,s):
    if isinstance(s,(int,float)): s=torch.full((f.shape[0],1),s,device=f.device)
    else: s=s.view(-1,1).float()
    sp=torch.mean(f**2,dim=(1,2),keepdim=True)
    return f+torch.sqrt(sp/(10**(s/10.0)).unsqueeze(-1))*torch.randn_like(f)

# Load features (keep on disk, load in micro-batches)
feats_all = np.load('results/clean_features_sa3.npy')
Nt = feats_all.shape[0]
print(f'Features shape: {feats_all.shape}')

results = []
combos = [(s,rr,rep) for s in SNRS for rr in RRS for rep in range(NR)]

for start in range(0, Nt, MB):
    end = min(start+MB, Nt)
    micro = torch.from_numpy(feats_all[start:end]).float().to(DEV)  # (mB, 1024, 1)
    micro_mod = micro.transpose(1,2)  # (mB, 1, 1024)
    mB = micro.shape[0]
    lbl = labels_all[start:end]

    # 预计算 SA 缓存（11次）和 RA 缓存（44次）
    sa_cache = {}
    for snr in tqdm(SNRS, desc=f'[{start}:{end}] SA cache', leave=False):
        with torch.no_grad():
            sa_cache[snr] = sa(micro_mod, snr)

    ra_cache = {}
    for snr in tqdm(SNRS, desc=f'[{start}:{end}] RA cache', leave=False):
        for rr in RRS:
            rate = max(1, int(C*rr))
            with torch.no_grad():
                ra_cache[(snr, rr)] = ra(sa_cache[snr], rate)

    pbar = tqdm(combos, desc=f'[{start}:{end}] AWGN+Dec+Cls', leave=False)
    for snr, rr, rep in pbar:
        x_sara, _ = ra_cache[(snr, rr)]
        x_t = x_sara.transpose(1,2)
        with torch.no_grad():
            x_n = awgn(x_t, snr)
            pre_r = dec_pre(x_n)
            post_r = dec_post(x_n)
            pre_p = classify(pre_r)
            post_p = classify(post_r)
            pre_acc = (pre_p == lbl).float().mean().item()
            post_acc = (post_p == lbl).float().mean().item()

        results.append({'snr':snr,'rate_ratio':rr,'repeat':rep,
                        'micro_batch':start,'decoder':'pre','acc':pre_acc})
        results.append({'snr':snr,'rate_ratio':rr,'repeat':rep,
                        'micro_batch':start,'decoder':'post','acc':post_acc})

    del micro, micro_mod, sa_cache, ra_cache
    torch.cuda.empty_cache()

df = pd.DataFrame(results)
df.to_csv('results/phaseA_comparison.csv', index=False)
print(f'\nSaved: results/phaseA_comparison.csv ({len(df)} rows)')

# Aggregate over micro-batches (weighted by batch size)
pre_acc_summary = df[df['decoder']=='pre'].groupby(['snr','rate_ratio'])['acc'].mean()
post_acc_summary = df[df['decoder']=='post'].groupby(['snr','rate_ratio'])['acc'].mean()

print(); print('='*70)
print('PHASE A: Pre vs Post Task-Driven Fine-tuning (SA+RA, SA3)')
print(f'{"SNR":<6} {"Rate":<8} {"Pre Acc":<12} {"Post Acc":<12} {"Improve":<10}')
print('-'*70)
for s in [0,5,10,15,20]:
    for rr in [0.2,0.5,0.8,1.0]:
        pre = pre_acc_summary.get((s,rr), float('nan'))
        post = post_acc_summary.get((s,rr), float('nan'))
        if not np.isnan(pre):
            imp = (post-pre)/max(pre,0.001)*100
            marker = ' ***' if post>pre+0.005 else ''
            print(f'{s:<6} {rr:<8} {pre:<12.4f} {post:<12.4f} {imp:+.1f}%{marker}')
    print('-'*70)
