# eval_baselines.py
# 完整基线评估：MSE + Classification Accuracy
# 6 种方法 × SA3 全量测试集 × 11 SNR × 4 速率比 × 3 重复
#
# 方法：
#   1. NoAdapt (SA3 → AWGN)
#   2. 均匀量化 (8bit quant → AWGN → dequant)
#   3. Plain-JSCC (MSE训练的MLP自编码器，无自适应)
#   4. SA+MSE Decoder (你的SA)
#   5. SA+RA+MSE Decoder (你的最终方案)
#   6. SA+RA+CE Decoder (Phase B，纯分类解码器—Part 2 only)

import sys, os, torch, torch.nn as nn, numpy as np, pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('./Pointnet_Pointnet2_pytorch'); sys.path.append('./Pointnet_Pointnet2_pytorch/models')
sys.path.append('./experiments/adapters')
from swin_adaptive_modules import ChannelModNet, RateModNet

DEV = torch.device('cuda')
C = 1024; SNRS = [0,2,4,6,8,10,12,14,16,18,20]; RRS = [0.2,0.5,0.8,1.0]; NR=3; MB=400

# Labels
import glob
class DS:
    def __init__(s,r):
        cls=sorted([d for d in os.listdir(r) if os.path.isdir(os.path.join(r,d))])
        s.p=[c for clsname in cls for c in glob.glob(os.path.join(r,clsname,'*.txt'))]
        s.l=[cls.index(f.split(os.sep)[-2]) for f in s.p]
    def __len__(s): return len(s.p)
labels_all = torch.tensor(DS('data/modelnet40_normal_resampled/test').l).to(DEV)
print(f'Test: {len(labels_all)} samples')

# ---- Decoder classes ----
class Dec3(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1024))
    def forward(s,x): x=x.transpose(1,2);x=s.net(x);return x.transpose(1,2)

# ---- Plain-JSCC ----
class PlainEncoder(nn.Module):
    def __init__(s,bn): super().__init__(); s.net=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,bn))
    def forward(s,x): return s.net(x.squeeze(-1))
class PlainDecoder2(nn.Module):
    def __init__(s,bn): super().__init__(); s.net=nn.Sequential(nn.Linear(bn,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,1024))
    def forward(s,x): return s.net(x).unsqueeze(-1)

# ---- Classifier ----
from pointnet2_cls_msg import get_model
pn2 = get_model(40,True).to(DEV)
ck=torch.load('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth',map_location=DEV,weights_only=False)
pn2.load_state_dict(ck.get('model_state_dict',ck)); pn2.eval()
for p in pn2.parameters(): p.requires_grad=False
def classify(f):
    x=f.squeeze(-1); x=pn2.drop1(F.relu(pn2.bn1(pn2.fc1(x))))
    x=pn2.drop2(F.relu(pn2.bn2(pn2.fc2(x)))); x=pn2.fc3(x)
    return F.log_softmax(x,-1).argmax(-1)

def awgn(f,s):
    if isinstance(s,(int,float)): s=torch.full((f.shape[0],1),s,device=f.device)
    else: s=s.view(-1,1).float()
    sp=torch.mean(f**2,dim=(1,2),keepdim=True)
    return f+torch.sqrt(sp/(10**(s/10.0)).unsqueeze(-1))*torch.randn_like(f)

# ---- Load all models ----
print('Loading models...')

# SA+RA
sa = ChannelModNet(C,int(C*1.5),7).to(DEV); sa.eval()
sa.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth',map_location=DEV))
ra = RateModNet(C,int(C*1.5),7).to(DEV); ra.eval()
ra.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth',map_location=DEV))
dec_mse = Dec3().to(DEV); dec_mse.eval()
dec_mse.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth',map_location=DEV))

# SA only (decoder class matching train_ae.py structure exactly)
class SADecoder(nn.Module):
    def __init__(s,fd=1024,hd=256): super().__init__(); s.fc1=nn.Linear(fd,hd); s.fc2=nn.Linear(hd,fd); s.relu=nn.ReLU()
    def forward(s,x): x=x.transpose(1,2); x=s.relu(s.fc1(x)); x=s.fc2(x); return x.transpose(1,2)
sa_only_net = ChannelModNet(C,int(C*1.5),7).to(DEV); sa_only_net.eval()
sa_only_net.load_state_dict(torch.load('pretrained/sa_net_sa3_trained.pth',map_location=DEV))
sa_dec2 = SADecoder().to(DEV); sa_dec2.eval()
sa_dec2.load_state_dict(torch.load('pretrained/decoder_sa3.pth',map_location=DEV))

# CE decoder (Phase B)
try:
    dec_ce = Dec3().to(DEV); dec_ce.eval()
    dec_ce.load_state_dict(torch.load('pretrained/phaseB_decoder_sa3.pth',map_location=DEV))
    has_ce = True
except: has_ce = False; print('[WARN] CE decoder not found, skipping')

# Plain-JSCC per rate
plain_encs={}; plain_decs={}
for rr in RRS:
    bn = {0.2:205,0.5:512,0.8:820,1.0:1024}[rr]
    enc = PlainEncoder(bn).to(DEV); enc.eval()
    enc.load_state_dict(torch.load(f'pretrained/plain_jscc_encoder_r{rr}.pth',map_location=DEV))
    dec = PlainDecoder2(bn).to(DEV); dec.eval()
    dec.load_state_dict(torch.load(f'pretrained/plain_jscc_decoder_r{rr}.pth',map_location=DEV))
    plain_encs[rr]=enc; plain_decs[rr]=dec
print('[OK] All models loaded')

# ---- Eval loop ----
feats_all = np.load('results/clean_features_sa3.npy')
Nt = feats_all.shape[0]
combos = [(s,rr,rep) for s in SNRS for rr in RRS for rep in range(NR)]
results = []

for start in range(0, Nt, MB):
    end = min(start+MB,Nt); micro = torch.from_numpy(feats_all[start:end]).float().to(DEV)
    micro_mod = micro.transpose(1,2); lbl = labels_all[start:end]; mB = micro.shape[0]

    # Cache SA outputs
    sa_cache={}
    for snr in tqdm(SNRS,desc=f'[{start}:{end}] SA cache',leave=False):
        with torch.no_grad(): sa_cache[snr]=sa(micro_mod,snr)

    ra_cache={}
    for snr in tqdm(SNRS,desc=f'[{start}:{end}] RA cache',leave=False):
        for rr in RRS:
            rate=max(1,int(C*rr))
            with torch.no_grad(): ra_cache[(snr,rr)]=ra(sa_cache[snr],rate)

    for snr,rr,rep in tqdm(combos,desc=f'[{start}:{end}] Eval',leave=False):
        rate = max(1,int(C*rr))
        with torch.no_grad():
            # === Method 1: NoAdapt ===
            xn = awgn(micro,snr)
            results.append({'method':'NoAdapt','snr':snr,'rate_ratio':rr,
                'mse':float(F.mse_loss(xn,micro)),
                'acc':(classify(xn)==lbl).float().mean().item(),'repeat':rep})

            # === Method 2: Uniform Quantization ===
            scale = micro.abs().max(dim=1,keepdim=True)[0].max(dim=2,keepdim=True)[0].clamp(min=1e-6)
            quant = torch.round(micro/scale*127).clamp(-127,127)/127*scale
            qn = awgn(quant,snr)
            results.append({'method':'Quant(8bit)','snr':snr,'rate_ratio':rr,
                'mse':float(F.mse_loss(qn,micro)),
                'acc':(classify(qn)==lbl).float().mean().item(),'repeat':rep})

            # === Method 3: Plain-JSCC ===
            enc=plain_encs[rr]; dec_pl=plain_decs[rr]
            z=enc(micro)                                    # (mB, bn)
            zn=awgn(z.unsqueeze(-1),snr)                   # (mB, bn, 1)
            xr_pl=dec_pl(zn.squeeze(-1))                   # (mB, 1024, 1)
            results.append({'method':'Plain-JSCC','snr':snr,'rate_ratio':rr,
                'mse':float(F.mse_loss(xr_pl,micro)),
                'acc':(classify(xr_pl)==lbl).float().mean().item(),'repeat':rep})

            # === Method 4: SA-only ===
            xsa = sa_only_net(micro_mod, snr)
            xsa_t = xsa.transpose(1,2); xsa_n = awgn(xsa_t, snr)
            xsa_r = sa_dec2(xsa_n)
            results.append({'method':'SA-only','snr':snr,'rate_ratio':rr,
                'mse':float(F.mse_loss(xsa_r,micro)),
                'acc':(classify(xsa_r)==lbl).float().mean().item(),'repeat':rep})

            # === Method 5: SA+RA+MSE ===
            xsara,_=ra_cache[(snr,rr)]; xsr_t=xsara.transpose(1,2)
            xsr_n=awgn(xsr_t,snr); xsr_r=dec_mse(xsr_n)
            results.append({'method':'SA+RA+MSE','snr':snr,'rate_ratio':rr,
                'mse':float(F.mse_loss(xsr_r,micro)),
                'acc':(classify(xsr_r)==lbl).float().mean().item(),'repeat':rep})

            # === Method 6: SA+RA+CE (if available) ===
            if has_ce:
                xsr_r2=dec_ce(xsr_n)
                results.append({'method':'SA+RA+CE','snr':snr,'rate_ratio':rr,
                    'mse':0.0, # CE decoder doesn't optimize MSE
                    'acc':(classify(xsr_r2)==lbl).float().mean().item(),'repeat':rep})

    del micro,micro_mod,sa_cache,ra_cache; torch.cuda.empty_cache()

df=pd.DataFrame(results)
df.to_csv('results/baselines_full_comparison.csv',index=False)
print(f'\nSaved: baselines_full_comparison.csv ({len(df)} rows)')

# ---- Summary ----
print(); print('='*80)
print('COMPLETE BASELINE COMPARISON (SA3, full 12311 test)')
print(f'{"Method":<20} {"Rate":<8} {"SNR=0dB":<20} {"SNR=10dB":<20} {"SNR=20dB":<20}')
print(f'{"":20} {"":8} {"MSE":<10} {"Acc":<10} {"MSE":<10} {"Acc":<10} {"MSE":<10} {"Acc":<10}')
print('='*80)
methods_ordered = ['NoAdapt','Quant(8bit)','Plain-JSCC','SA-only','SA+RA+MSE']
if has_ce: methods_ordered.append('SA+RA+CE')
for m in methods_ordered:
    sub=df[df['method']==m]
    for rr in [0.5, 1.0]:
        row0=sub[(sub['snr']==0)&(sub['rate_ratio']==rr)]
        row10=sub[(sub['snr']==10)&(sub['rate_ratio']==rr)]
        row20=sub[(sub['snr']==20)&(sub['rate_ratio']==rr)]
        m0=row0['mse'].mean(); a0=row0['acc'].mean()
        m10=row10['mse'].mean(); a10=row10['acc'].mean()
        m20=row20['mse'].mean(); a20=row20['acc'].mean()
        lbl = f'{m} r={rr}'
        print(f'{lbl:<20} {rr:<8} {m0:<10.4f} {a0:<10.4f} {m10:<10.4f} {a10:<10.4f} {m20:<10.4f} {a20:<10.4f}')
    print('-'*80)
