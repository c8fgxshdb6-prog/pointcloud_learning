# eval_phaseA2.py — 全量测试集评估 Phase A2 微调前后
import sys, os, torch, torch.nn as nn, numpy as np, pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('./Pointnet_Pointnet2_pytorch'); sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')
from swin_adaptive_modules import ChannelModNet, RateModNet

DEV = torch.device('cuda')
C = 1024; SNRS = [0,2,4,6,8,10,12,14,16,18,20]; RRS = [0.2,0.5,0.8,1.0]; NR = 3; MB = 400

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
print(f'Total test: {len(labels_all)}')

# Models (two sets: pre/post)
def build_models():
    sa = ChannelModNet(C,int(C*1.5),7).to(DEV); sa.eval()
    ra = RateModNet(C,int(C*1.5),7).to(DEV); ra.eval()
    class D(nn.Module):
        def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1024))
        def forward(s,x): x=x.transpose(1,2);x=s.net(x);return x.transpose(1,2)
    dec = D().to(DEV); dec.eval()
    return sa, ra, dec

# Pre: original SARA
sa_pre, ra_pre, dec_pre = build_models()
sa_pre.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth',map_location=DEV))
ra_pre.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth',map_location=DEV))
dec_pre.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth',map_location=DEV))

# Post: Phase A2 task fine-tuned
sa_post, ra_post, dec_post = build_models()
sa_post.load_state_dict(torch.load('pretrained/sara_sa_net_sa3_task.pth',map_location=DEV))
ra_post.load_state_dict(torch.load('pretrained/sara_ra_net_sa3_task.pth',map_location=DEV))
dec_post.load_state_dict(torch.load('pretrained/sara_decoder_sa3_task_full.pth',map_location=DEV))
print('[OK] Models loaded')

# Classifier
from pointnet2_cls_msg import get_model
pn2 = get_model(40,True).to(DEV)
ck = torch.load('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth',map_location=DEV,weights_only=False)
pn2.load_state_dict(ck.get('model_state_dict',ck)); pn2.eval()
for p in pn2.parameters(): p.requires_grad = False

def classify(feat):
    x=feat.squeeze(-1); x=pn2.drop1(F.relu(pn2.bn1(pn2.fc1(x))))
    x=pn2.drop2(F.relu(pn2.bn2(pn2.fc2(x)))); x=pn2.fc3(x)
    return F.log_softmax(x,-1).argmax(-1)

def awgn(f,s):
    if isinstance(s,(int,float)): s=torch.full((f.shape[0],1),s,device=f.device)
    else: s=s.view(-1,1).float()
    sp=torch.mean(f**2,dim=(1,2),keepdim=True)
    return f+torch.sqrt(sp/(10**(s/10.0)).unsqueeze(-1))*torch.randn_like(f)

# Eval loop
feats_all = np.load('results/clean_features_sa3.npy')
Nt = feats_all.shape[0]
combos = [(s,rr,rep) for s in SNRS for rr in RRS for rep in range(NR)]
results = []

for start in range(0, Nt, MB):
    end = min(start+MB, Nt)
    micro = torch.from_numpy(feats_all[start:end]).float().to(DEV)
    micro_mod = micro.transpose(1,2); mB = micro.shape[0]; lbl = labels_all[start:end]

    # SA cache for pre and post
    sa_pre_cache={}; sa_post_cache={}
    for snr in tqdm(SNRS, desc=f'[{start}:{end}] SA', leave=False):
        with torch.no_grad():
            sa_pre_cache[snr]=sa_pre(micro_mod,snr)
            sa_post_cache[snr]=sa_post(micro_mod,snr)

    # RA cache
    ra_pre_cache={}; ra_post_cache={}
    for snr in tqdm(SNRS, desc=f'[{start}:{end}] RA', leave=False):
        for rr in RRS:
            rate=max(1,int(C*rr))
            with torch.no_grad():
                ra_pre_cache[(snr,rr)]=ra_pre(sa_pre_cache[snr],rate)
                ra_post_cache[(snr,rr)]=ra_post(sa_post_cache[snr],rate)

    for snr, rr, rep in tqdm(combos, desc=f'[{start}:{end}] Dec+Cls', leave=False):
        x_sara_pre,_=ra_pre_cache[(snr,rr)]; x_sara_post,_=ra_post_cache[(snr,rr)]
        with torch.no_grad():
            xn_pre=awgn(x_sara_pre.transpose(1,2),snr); xn_post=awgn(x_sara_post.transpose(1,2),snr)
            for tag, x_r in [('pre',dec_pre(xn_pre)),('post',dec_post(xn_post))]:
                acc=(classify(x_r)==lbl).float().mean().item()
                results.append({'snr':snr,'rate_ratio':rr,'repeat':rep,'decoder':tag,'acc':acc})

    del micro,micro_mod,sa_pre_cache,sa_post_cache,ra_pre_cache,ra_post_cache
    torch.cuda.empty_cache()

df=pd.DataFrame(results)
df.to_csv('results/phaseA2_comparison.csv',index=False)
print(f'Saved: phaseA2_comparison.csv ({len(df)} rows)')

# Summary
pre_a=df[df['decoder']=='pre'].groupby(['snr','rate_ratio'])['acc'].mean()
post_a=df[df['decoder']=='post'].groupby(['snr','rate_ratio'])['acc'].mean()

print(); print('='*70)
print('PHASE A2: Full Fine-tuning (SA+RA+Decoder) — Full Test Set')
print(f'{"SNR":<6} {"Rate":<8} {"Pre Acc":<10} {"Post Acc":<10} {"Delta":<8}')
print('-'*70)
for s in [0,5,10,15,20]:
    for rr in [0.2,0.5,0.8,1.0]:
        pre=pre_a.get((s,rr),0); post=post_a.get((s,rr),0)
        d=post-pre; m='***' if post>pre+0.005 else ''
        print(f'{s:<6} {rr:<8} {pre:<10.4f} {post:<10.4f} {d:+.3f}{m}')
    print('-'*70)
