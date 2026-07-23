# eval_phaseB.py — 全量 12311 测试：CE解码器 vs MSE解码器 vs NoAdapt vs SA
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
        cls=sorted([d for d in os.listdir(r) if os.path.isdir(os.path.join(r,d))])
        s.p=[]; s.l=[]
        for c in cls:
            for f in glob.glob(os.path.join(r,c,'*.txt')): s.p.append(f); s.l.append(cls.index(c))
    def __len__(s): return len(s.p)
labels_all = torch.tensor(DS('data/modelnet40_normal_resampled/test').l).to(DEV)
print(f'Total test: {len(labels_all)}')

# Decoder class
class Dec(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1024))
    def forward(s,x): x=x.transpose(1,2);x=s.net(x);return x.transpose(1,2)

# SA + RA (shared for all methods)
sa_net = ChannelModNet(C,int(C*1.5),7).to(DEV); sa_net.eval()
sa_net.load_state_dict(torch.load('pretrained/sara_sa_net_sa3.pth',map_location=DEV))
ra_net = RateModNet(C,int(C*1.5),7).to(DEV); ra_net.eval()
ra_net.load_state_dict(torch.load('pretrained/sara_ra_net_sa3.pth',map_location=DEV))

# Decoders
dec_ce = Dec().to(DEV); dec_ce.eval()
dec_ce.load_state_dict(torch.load('pretrained/phaseB_decoder_sa3.pth',map_location=DEV))
dec_mse = Dec().to(DEV); dec_mse.eval()
dec_mse.load_state_dict(torch.load('pretrained/sara_decoder_sa3.pth',map_location=DEV))
print('[OK] CE decoder + MSE decoder loaded')

# Classifier
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

# Eval
feats_all = np.load('results/clean_features_sa3.npy')
Nt = feats_all.shape[0]
combos = [(s,rr,rep) for s in SNRS for rr in RRS for rep in range(NR)]
results = []

for start in range(0, Nt, MB):
    end = min(start+MB,Nt); micro = torch.from_numpy(feats_all[start:end]).float().to(DEV)
    micro_mod = micro.transpose(1,2); lbl = labels_all[start:end]

    # SA cache
    sa_cache={}
    for snr in tqdm(SNRS,desc=f'[{start}:{end}] SA',leave=False):
        with torch.no_grad(): sa_cache[snr]=sa_net(micro_mod,snr)

    # RA cache
    ra_cache={}
    for snr in tqdm(SNRS,desc=f'[{start}:{end}] RA',leave=False):
        for rr in RRS:
            with torch.no_grad(): ra_cache[(snr,rr)]=ra_net(sa_cache[snr],max(1,int(C*rr)))

    # Eval combos
    for snr,rr,rep in tqdm(combos,desc=f'[{start}:{end}] Eval',leave=False):
        x_sara,_=ra_cache[(snr,rr)]; rate=max(1,int(C*rr))
        with torch.no_grad():
            xn=awgn(x_sara.transpose(1,2),snr)
            # NoAdapt
            xn_clean=awgn(micro,snr)
            # CE decoder
            xr_ce=dec_ce(xn)
            # MSE decoder
            xr_mse=dec_mse(xn)

        for tag,xr in [('NoAdapt',xn_clean),('SA+RA+MSE',xr_mse),('SA+RA+CE',xr_ce)]:
            acc=(classify(xr)==lbl).float().mean().item()
            results.append({'snr':snr,'rate_ratio':rr,'rate':rate,'repeat':rep,'method':tag,'acc':acc})

    del micro,micro_mod,sa_cache,ra_cache; torch.cuda.empty_cache()

df=pd.DataFrame(results)
df.to_csv('results/phaseB_classification.csv',index=False)
print(f'Saved: phaseB_classification.csv ({len(df)} rows)')

# Summary
print(); print('='*75)
print('PHASE B: CE Decoder vs MSE Decoder vs NoAdapt (SA3, full 12311 test)')
print(f'{"SNR":<6} {"Rate":<8} {"NoAdapt":<12} {"SA+RA+MSE":<12} {"SA+RA+CE":<12} {"CE Gain":<10}')
print('-'*75)
for s in [0,5,10,15,20]:
    for rr in [0.2,0.5,0.8,1.0]:
        no=df[(df['method']=='NoAdapt')&(df['snr']==s)&(df['rate_ratio']==rr)]['acc'].mean()
        mse=df[(df['method']=='SA+RA+MSE')&(df['snr']==s)&(df['rate_ratio']==rr)]['acc'].mean()
        ce=df[(df['method']=='SA+RA+CE')&(df['snr']==s)&(df['rate_ratio']==rr)]['acc'].mean()
        gain=(ce-mse)/max(mse,0.001)*100; marker='***' if ce>mse+0.02 else ('!' if ce>mse else '')
        print(f'{s:<6} {rr:<8} {no:<12.4f} {mse:<12.4f} {ce:<12.4f} {gain:+.1f}% {marker}')
    print('-'*75)
