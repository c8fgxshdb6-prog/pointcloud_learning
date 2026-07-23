# run_sara_joint_exp.py - 高效版本
# SA输出按SNR缓存，RA输出按(SNR,rate)缓存，只重复AWGN+Decoder
import sys, os, torch, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm

p = r'D:\Users\yxf\Desktop\pointcloud_learning'
sys.path.insert(0, p); sys.path.insert(0, p+'/SwinJSCC'); sys.path.insert(0, p+'/experiments')
from adapters.swin_adaptive_modules import ChannelModNet, RateModNet
from metrics.feature_metrics import compute_all_metrics

class DeeperDecoder(nn.Module):
    def __init__(self, fd, h1=512, h2=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fd,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,fd))
    def forward(self,x): x=x.transpose(1,2); x=self.net(x); return x.transpose(1,2)

def awgn(f,s):
    if isinstance(s,(int,float)): s=torch.full((f.shape[0],1),s,device=f.device)
    else: s=s.view(-1,1).float()
    sp=torch.mean(f**2,dim=(1,2),keepdim=True)
    return f+torch.sqrt(sp/(10**(s/10.0)).unsqueeze(-1))*torch.randn_like(f)

def sample_metrics(ct,rt,n=150):
    B=ct.shape[0]; n=min(n,B); idx=np.random.choice(B,n,replace=False)
    m=compute_all_metrics(ct[idx].cpu().numpy(),rt[idx].cpu().numpy())
    return {k:(v.mean() if isinstance(v,np.ndarray) else v) for k,v in m.items()}

DEV=torch.device('cuda')
SNRS=[0,2,4,6,8,10,12,14,16,18,20]
RRS=[0.2,0.5,0.8,1.0]
NR=10
NT=500
NM=150

# SA1用小配置避免OOM
CFG={'sa1':{'C':320,'N':512,'mb':25,'nt':200,'nr':5},
     'sa2':{'C':640,'N':128,'mb':100,'nt':500,'nr':10},
     'sa3':{'C':1024,'N':1,'mb':200,'nt':500,'nr':10}}

def run_layer(layer,ch_mode='awgn',rch=None):
    C,N,mb=CFG[layer]['C'],CFG[layer]['N'],CFG[layer]['mb']
    nt,nr_local=CFG[layer]['nt'],CFG[layer]['nr']
    clean=np.load(f'results/clean_features_{layer}.npy')[:nt]
    Bt=clean.shape[0]

    sa=ChannelModNet(C,int(C*1.5),7).to(DEV); sa.eval()
    sa.load_state_dict(torch.load(f'pretrained/sara_sa_net_{layer}.pth',map_location=DEV))
    ra=RateModNet(C,int(C*1.5),7).to(DEV); ra.eval()
    ra.load_state_dict(torch.load(f'pretrained/sara_ra_net_{layer}.pth',map_location=DEV))
    dec=DeeperDecoder(C).to(DEV); dec.eval()
    dec.load_state_dict(torch.load(f'pretrained/sara_decoder_{layer}.pth',map_location=DEV))

    results=[]
    for start in range(0,Bt,mb):
        end=min(start+mb,Bt)
        micro=torch.from_numpy(clean[start:end]).float().to(DEV)  # (mB,C,N)
        mB=micro.shape[0]

        # 1. 预计算SA输出: 每个SNR一次 (11次SA)
        sa_cache={}
        for snr in tqdm(SNRS,desc=f'  {layer}[{start}:{end}] SA cache',leave=False):
            with torch.no_grad():
                sa_cache[snr]=sa(micro.transpose(1,2),snr)  # (mB,N,C)

        # 2. 对每个(SNR,rate_ratio): 预计算RA输出 (44次RA) + 重复AWGN+Decoder
        combos=[(s,r,rep) for s in SNRS for r in RRS for rep in range(nr_local)]
        ra_cache={}
        pbar=tqdm(combos,desc=f'  {layer}[{start}:{end}] RA+Ch+Dec',leave=False)
        for snr,rr,rep in pbar:
            rate=max(1,int(C*rr))
            key=(snr,rr)
            if key not in ra_cache:
                with torch.no_grad():
                    x_sara,_=ra(sa_cache[snr],rate)
                ra_cache[key]=x_sara
            # AWGN + Decoder (每次repeat不同噪声)
            with torch.no_grad():
                xt=ra_cache[key].transpose(1,2)
                xn=awgn(xt,snr) if ch_mode=='awgn' else rch.forward(xt,chan_param=snr,avg_pwr=False)
                xr=dec(xn)
            m=sample_metrics(micro,xr,n=NM)
            results.append({'layer':layer,'snr':snr,'rate_ratio':rr,'rate':rate,
                'repeat':rep,'bandwidth':m['bandwidth'],'mse':m['mse'],
                'psnr':m['psnr'],'cos_sim':m['cos_sim']})

        del micro,sa_cache,ra_cache; torch.cuda.empty_cache()
    del sa,ra,dec; torch.cuda.empty_cache()
    return results

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--channel',default='awgn',choices=['awgn','rayleigh','both'])
    a=ap.parse_args()
    channels=['awgn','rayleigh'] if a.channel=='both' else [a.channel]

    for ch in channels:
        print(f"\n{'='*50}\nSA+RA Phase2: {ch.upper()}\n{'='*50}")
        rch=None
        if ch=='rayleigh':
            from net.channel import Channel
            class DA: channel_type='rayleigh'; multiple_snr='10'
            class DC: device=DEV; CUDA=True; logger=None
            rch=Channel(DA(),DC())

        all_r=[]
        for layer in ['sa3','sa2','sa1']:
            print(f"\nLayer {layer}: C={CFG[layer]['C']}, N={CFG[layer]['N']}")
            res=run_layer(layer,ch_mode=ch,rch=rch)
            all_r.extend(res)

        df=pd.DataFrame(all_r)
        fn=f'results/channel_exp_results_sara_joint{"" if ch=="awgn" else "_rayleigh"}.csv'
        os.makedirs('results',exist_ok=True)
        df.to_csv(fn,index=False)
        print(f"Saved: {fn} ({len(df)} rows)")

if __name__=='__main__': main()
