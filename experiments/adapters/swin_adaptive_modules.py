# swin_adaptive_modules.py
import torch
import torch.nn as nn

class AdaptiveModulator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class ChannelModNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim=None, num_layers=7):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(feat_dim * 1.5)
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(feat_dim, hidden_dim))
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else feat_dim
            self.sm_list.append(nn.Linear(hidden_dim, out_dim))

        self.bm_list = nn.ModuleList([
            AdaptiveModulator(hidden_dim) for _ in range(num_layers)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr):
        B, N, C = x.shape
        device = x.device
        # 将 snr 转换为 (B, 1) 张量
        if isinstance(snr, (int, float)):
            cond = torch.full((B, 1), snr, dtype=torch.float, device=device)
        elif isinstance(snr, torch.Tensor):
            cond = snr.view(B, 1).float()
        else:
            raise TypeError(f"snr must be int, float or Tensor, got {type(snr)}")

        temp = x.detach()
        for i in range(self.num_layers):
            temp = self.sm_list[i](temp)
            mod_factor = self.bm_list[i](cond)  # (B, hidden_dim)
            mod_factor = mod_factor.unsqueeze(1).expand(-1, N, -1)
            temp = temp * mod_factor
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        return x * mod_val


class RateModNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim=None, num_layers=7):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(feat_dim * 1.5)
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(feat_dim, hidden_dim))
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else feat_dim
            self.sm_list.append(nn.Linear(hidden_dim, out_dim))

        self.bm_list = nn.ModuleList([
            AdaptiveModulator(hidden_dim) for _ in range(num_layers)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rate):
        B, N, C = x.shape
        device = x.device
        # 将 rate 转换为 (B, 1) 张量
        if isinstance(rate, (int, float)):
            cond = torch.full((B, 1), rate, dtype=torch.float, device=device)
        elif isinstance(rate, torch.Tensor):
            cond = rate.view(B, 1).float()
        else:
            raise TypeError(f"rate must be int, float or Tensor, got {type(rate)}")

        temp = x.detach()
        for i in range(self.num_layers):
            temp = self.sm_list[i](temp)
            mod_factor = self.bm_list[i](cond)
            mod_factor = mod_factor.unsqueeze(1).expand(-1, N, -1)
            temp = temp * mod_factor
        mod_val = self.sigmoid(self.sm_list[-1](temp))  # (B, N, C)

        # 通道选择
        importance = mod_val.mean(dim=1)  # (B, C)
        k = int(rate) if isinstance(rate, (int, float)) else int(rate[0].item())
        k = min(k, C)
        topk_vals, topk_idx = torch.topk(importance, k, dim=1)
        mask = torch.zeros_like(importance)
        mask.scatter_(1, topk_idx, 1.0)
        mask = mask.unsqueeze(1).expand(-1, N, -1)
        return x * mask, mask


class AdaptiveModulatorCombined(nn.Module):
    def __init__(self, feat_dim, hidden_dim=None, num_layers=7):
        super().__init__()
        self.sa_net = ChannelModNet(feat_dim, hidden_dim, num_layers)
        self.ra_net = RateModNet(feat_dim, hidden_dim, num_layers)

    def forward(self, x, snr, rate):
        x = self.sa_net(x, snr)
        x, mask = self.ra_net(x, rate)
        return x, mask