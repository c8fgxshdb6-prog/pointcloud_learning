"""
统一图表生成脚本
===============
从 load_data.py 读取数据，生成论文级对比图表。

用法:
  python generate_figures.py              # 生成全部图表
  python generate_figures.py --group sa   # 只生成 SA 自适应组
  python generate_figures.py --group ra   # 只生成 RA 自适应组
  python generate_figures.py --group summary  # 只生成综合对比

生成的图表保存在 results/figures/ 目录下。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')                    # 无 GUI 后端，适合服务器
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings

from load_data import (
    RESULTS_DIR, DATASETS, LAYER_INFO, METRIC_CONFIG,
    load_dataset, load_and_aggregate, aggregate,
    filter_layer, filter_rate, merge_ra_datasets,
    compute_per_sample_kb,
    load_baselines_classification, load_baselines_feature_metric,
    METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES, METHOD_ORDER,
)

# ============================================================
# 中文字体检测 — 无 CJK 字体时自动用英文子图标题
# ============================================================
_CJK_CANDIDATES = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'Noto Sans SC',
]
_CJK_FONT = None
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _CJK_CANDIDATES:
    if _font_name in _available_fonts:
        _CJK_FONT = _font_name
        break

# ============================================================
# 全局绘图设置
# ============================================================
plt.style.use('seaborn-v0_8-darkgrid')

if _CJK_FONT:
    plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[font] CJK font found: {_CJK_FONT}")
else:
    # 无 CJK 字体: 子图标题改用英文避免方块
    print("[font] No CJK font found. Subplot titles will use English.")
    for _layer in LAYER_INFO:
        LAYER_INFO[_layer]['name_cn'] = LAYER_INFO[_layer]['name']
FIG_DIR = RESULTS_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 全局参数
DPI = 200
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9

# 速率比专用配色（用于同一张图中区分不同 rate_ratio）
RATE_COLORS = {
    0.2: '#E41A1C',   # 红
    0.5: '#377EB8',   # 蓝
    0.8: '#4DAF4A',   # 绿
    1.0: '#984EA3',   # 紫
}

plt.rcParams.update({
    'font.size': FONT_SIZE_LABEL,
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.labelsize': FONT_SIZE_LABEL,
    'xtick.labelsize': FONT_SIZE_TICK,
    'ytick.labelsize': FONT_SIZE_TICK,
    'legend.fontsize': 8,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})


# ============================================================
# 辅助绘图函数
# ============================================================

def _plot_with_error(ax, x, y_mean, y_std, label, color, linestyle='-',
                     marker='o', markersize=4, linewidth=1.5, alpha=0.15):
    """在指定 axes 上画一条带误差阴影的曲线。"""
    ax.plot(x, y_mean, color=color, linestyle=linestyle, marker=marker,
            markersize=markersize, linewidth=linewidth, label=label)
    if y_std is not None and len(y_std) == len(x):
        ax.fill_between(x,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=color, alpha=alpha, linewidth=0)


def _add_metric_label(ax, metric):
    """设置坐标轴标签。"""
    cfg = METRIC_CONFIG.get(metric, {})
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(cfg.get('ylabel', metric.upper()))
    ax.grid(True, alpha=0.3)


def _three_layer_subplots(figsize=(18, 5)):
    """创建 1 行 3 列的标准三层子图布局。"""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    return fig, axes


# ============================================================
# 图表 1: SA 自适应 vs 基线 — AWGN, MSE (三层分图)
# ============================================================

def fig_sa_awgn_mse():
    """
    三个子图 (SA1/SA2/SA3)，每张图两条曲线:
      - 基线 (无自适应直接传输)
      - SA + 解码器 (训练后 SNR 自适应)
    横轴: SNR (0~20 dB), 纵轴: MSE
    证明: SA 自适应的降噪效果在各层均有效
    """
    metric = 'mse'
    bl_mean, bl_std = load_and_aggregate('baseline_awgn')
    sa_mean, sa_std = load_and_aggregate('sa_trained_awgn')

    if bl_mean is None or sa_mean is None:
        print("  [跳过] SA AWGN MSE — 数据缺失")
        return

    fig, axes = _three_layer_subplots()

    for i, layer in enumerate(['sa1', 'sa2', 'sa3']):
        ax = axes[i]
        info = LAYER_INFO[layer]

        # 基线: rate_ratio=1.0（全通道）
        bl_m, bl_s = filter_layer(bl_mean, bl_std, layer)
        bl_m, bl_s = filter_rate(bl_m, bl_s, 1.0)

        # SA: rate_ratio=1.0
        sa_m, sa_s = filter_layer(sa_mean, sa_std, layer)
        sa_m, sa_s = filter_rate(sa_m, sa_s, 1.0)

        # 基线
        _plot_with_error(ax, bl_m['snr'], bl_m[metric], bl_s[metric] if bl_s is not None else None,
                         label='Baseline (No Adapt)', color=DATASETS['baseline_awgn']['color'],
                         linestyle='--', marker='s')

        # SA + 解码器
        _plot_with_error(ax, sa_m['snr'], sa_m[metric], sa_s[metric] if sa_s is not None else None,
                         label='SA + Decoder (Trained)', color=DATASETS['sa_trained_awgn']['color'],
                         linestyle='-', marker='o')

        _add_metric_label(ax, metric)
        ax.set_title(info['name_cn'], fontweight='bold')
        ax.legend(loc='upper right')

    fig.suptitle('AWGN Channel: SA Adaptation vs Baseline (MSE)', fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sa_awgn_mse.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 2: SA 自适应 vs 基线 — AWGN, Cosine Similarity (三层分图)
# ============================================================

def fig_sa_awgn_cosine():
    """
    同图表 1，但纵轴为余弦相似度。
    余弦相似度对分类等语义任务更有意义——
    特征的空间方向比精确数值更重要。
    """
    metric = 'cos_sim'
    bl_mean, bl_std = load_and_aggregate('baseline_awgn')
    sa_mean, sa_std = load_and_aggregate('sa_trained_awgn')

    if bl_mean is None or sa_mean is None:
        print("  [跳过] SA AWGN Cosine — 数据缺失")
        return

    fig, axes = _three_layer_subplots()

    for i, layer in enumerate(['sa1', 'sa2', 'sa3']):
        ax = axes[i]
        bl_m, bl_s = filter_layer(bl_mean, bl_std, layer)
        bl_m, bl_s = filter_rate(bl_m, bl_s, 1.0)
        sa_m, sa_s = filter_layer(sa_mean, sa_std, layer)
        sa_m, sa_s = filter_rate(sa_m, sa_s, 1.0)

        _plot_with_error(ax, bl_m['snr'], bl_m[metric], bl_s[metric] if bl_s is not None else None,
                         label='Baseline (No Adapt)', color=DATASETS['baseline_awgn']['color'],
                         linestyle='--', marker='s')
        _plot_with_error(ax, sa_m['snr'], sa_m[metric], sa_s[metric] if sa_s is not None else None,
                         label='SA + Decoder (Trained)', color=DATASETS['sa_trained_awgn']['color'],
                         linestyle='-', marker='o')

        _add_metric_label(ax, metric)
        ax.set_title(LAYER_INFO[layer]['name_cn'], fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')

    fig.suptitle('AWGN Channel: SA Adaptation vs Baseline (Cosine Similarity)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sa_awgn_cosine.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 3: AWGN vs Rayleigh 信道对比 (SA2, MSE+PSNR)
# ============================================================

def fig_sa_channel_comparison():
    """
    1×2 子图 (MSE | PSNR)，SA2 层。
    四条曲线: AWGN Baseline, AWGN SA+Decoder, Rayleigh Baseline, Rayleigh SA+Decoder
    证明: SA 自适应在两种信道下均有效
    """
    layer = 'sa2'

    bl_awgn_m, bl_awgn_s = load_and_aggregate('baseline_awgn')
    sa_awgn_m, sa_awgn_s = load_and_aggregate('sa_trained_awgn')
    bl_ray_m, bl_ray_s = load_and_aggregate('baseline_rayleigh')
    sa_ray_m, sa_ray_s = load_and_aggregate('sa_trained_rayleigh')

    # 检查数据完整性
    if any(x is None for x in [bl_awgn_m, sa_awgn_m, bl_ray_m, sa_ray_m]):
        print("  [跳过] SA Channel Comparison — 数据缺失")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets_line = [
        ('baseline_awgn', bl_awgn_m, bl_awgn_s, 'AWGN Baseline'),
        ('sa_trained_awgn', sa_awgn_m, sa_awgn_s, 'AWGN SA+Decoder'),
        ('baseline_rayleigh', bl_ray_m, bl_ray_s, 'Rayleigh Baseline'),
        ('sa_trained_rayleigh', sa_ray_m, sa_ray_s, 'Rayleigh SA+Decoder'),
    ]

    for ax, metric in zip(axes, ['mse', 'psnr']):
        for ds_name, mean_df, std_df, custom_label in datasets_line:
            m, s = filter_layer(mean_df, std_df, layer)
            m, s = filter_rate(m, s, 1.0)

            if m.empty:
                continue

            info = DATASETS[ds_name]
            is_baseline = 'Baseline' in custom_label

            _plot_with_error(ax, m['snr'], m[metric], s[metric] if s is not None else None,
                             label=custom_label,
                             color=info['color'],
                             linestyle='--' if is_baseline else '-',
                             marker='s' if is_baseline else 'o')

        _add_metric_label(ax, metric)
        ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — {metric.upper()}', fontweight='bold')
        ax.legend(loc='best', fontsize=7)

    fig.suptitle('Channel Robustness: AWGN vs Rayleigh Fading (SA2)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sa_channel_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 3b: SA+RA Joint AWGN vs Rayleigh 信道对比 (SA2, MSE+PSNR)
# ============================================================

def fig_sara_channel_comparison():
    """
    1×2 子图 (MSE | PSNR)，SA2 层。
    四条曲线: AWGN Baseline, AWGN SA+RA, Rayleigh Baseline, Rayleigh SA+RA
    证明: SA+RA 联合系统在两种信道下均有效
    """
    layer = 'sa2'

    bl_awgn_m, bl_awgn_s = load_and_aggregate('baseline_awgn')
    sara_awgn_m, sara_awgn_s = load_and_aggregate('sara_joint_awgn')
    bl_ray_m, bl_ray_s = load_and_aggregate('baseline_rayleigh')
    sara_ray_m, sara_ray_s = load_and_aggregate('sara_joint_rayleigh')

    if any(x is None for x in [bl_awgn_m, sara_awgn_m, bl_ray_m, sara_ray_m]):
        print("  [跳过] SA+RA Channel Comparison — 数据缺失")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets_line = [
        ('baseline_awgn', bl_awgn_m, bl_awgn_s, 'AWGN Baseline'),
        ('sara_joint_awgn', sara_awgn_m, sara_awgn_s, 'AWGN SA+RA'),
        ('baseline_rayleigh', bl_ray_m, bl_ray_s, 'Rayleigh Baseline'),
        ('sara_joint_rayleigh', sara_ray_m, sara_ray_s, 'Rayleigh SA+RA'),
    ]

    for ax, metric in zip(axes, ['mse', 'psnr']):
        for ds_name, mean_df, std_df, custom_label in datasets_line:
            m, s = filter_layer(mean_df, std_df, layer)
            m, s = filter_rate(m, s, 1.0)

            if m.empty:
                continue

            info = DATASETS[ds_name]
            is_baseline = 'Baseline' in custom_label

            _plot_with_error(ax, m['snr'], m[metric], s[metric] if s is not None else None,
                             label=custom_label,
                             color=info['color'],
                             linestyle='--' if is_baseline else '-',
                             marker='s' if is_baseline else 'D')

        _add_metric_label(ax, metric)
        ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — {metric.upper()}', fontweight='bold')
        ax.legend(loc='best', fontsize=7)

    fig.suptitle('Channel Robustness: SA+RA Joint — AWGN vs Rayleigh Fading (SA2)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sara_channel_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 4: RA 速率自适应 — 不同速率比下的性能 (SA2)
# ============================================================

def fig_ra_rate_sweep():
    """
    1×2 子图 (MSE | PSNR)，SA2 层。
    5条曲线: Baseline (rate=1.0) + RA 训练后 4 个速率比 (0.2/0.5/0.8/1.0)
    证明: RA 在速率受限时仍保持可用质量；训练后 RA 比无自适应好
    """
    metric_pairs = [('mse', 0), ('psnr', 1)]
    layer = 'sa2'
    rate_ratios = [0.2, 0.5, 0.8, 1.0]

    ra_mean, ra_std = load_and_aggregate('ra_trained_sa2_awgn')
    bl_mean, bl_std = load_and_aggregate('baseline_awgn')

    if ra_mean is None or bl_mean is None:
        print("  [跳过] RA Rate Sweep — 数据缺失")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, _) in enumerate(metric_pairs):
        ax = axes[ax_idx]

        # 基线 (rate=1.0)
        bl_m, bl_s = filter_layer(bl_mean, bl_std, layer)
        bl_m, bl_s = filter_rate(bl_m, bl_s, 1.0)
        _plot_with_error(ax, bl_m['snr'], bl_m[metric],
                         bl_s[metric] if bl_s is not None else None,
                         label='Baseline (No Adapt, rate=1.0)',
                         color='#333333', linestyle='--', marker='s',
                         linewidth=2, alpha=0.10)

        # RA 训练后各速率比
        ra_m, ra_s = filter_layer(ra_mean, ra_std, layer)
        for rr in rate_ratios:
            ra_m_rr, ra_s_rr = filter_rate(ra_m, ra_s.copy() if ra_s is not None else None, rr)
            if ra_m_rr.empty:
                continue
            _plot_with_error(ax, ra_m_rr['snr'], ra_m_rr[metric],
                             ra_s_rr[metric] if ra_s_rr is not None else None,
                             label=f'RA Trained (rate={rr})',
                             color=RATE_COLORS[rr],
                             linestyle='-', marker='o', markersize=3, linewidth=1.2)

        _add_metric_label(ax, metric)
        ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — {metric.upper()}', fontweight='bold')
        ax.legend(loc='best', fontsize=7)

    fig.suptitle('AWGN Channel: Rate-Adaptive (RA) Performance at Different Compression Rates',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'ra_rate_sweep.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 5: RA 速率-质量折衷曲线
# ============================================================

def fig_ra_rate_tradeoff():
    """
    固定 SNR=10dB，展示 MSE 随 rate_ratio 的变化。
    一条递减曲线，展示从 rate=0.2 到 1.0 的质量增益递减效应。
    基线 (rate=1.0, No Adapt) 作为水平参考线。
    核心含义: 用更少的带宽换取了可接受的质量下降。
    """
    metric = 'mse'
    snr_fixed = 10
    layer = 'sa2'

    ra_mean, ra_std = load_and_aggregate('ra_trained_sa2_awgn')
    bl_mean, bl_std = load_and_aggregate('baseline_awgn')

    if ra_mean is None or bl_mean is None:
        print("  [跳过] RA Rate Tradeoff — 数据缺失")
        return

    # RA 训练后数据
    ra_m, ra_s = filter_layer(ra_mean, ra_std, layer)
    ra_m_snr = ra_m[ra_m['snr'] == snr_fixed].sort_values('rate_ratio')
    ra_s_snr = ra_s[ra_s['snr'] == snr_fixed].sort_values('rate_ratio') if ra_s is not None else None

    # 基线在 rate=1.0, SNR=10 的参考值
    bl_m, bl_s = filter_layer(bl_mean, bl_std, layer)
    bl_m, bl_s = filter_rate(bl_m, bl_s, 1.0)
    bl_ref = bl_m[bl_m['snr'] == snr_fixed]

    fig, ax = plt.subplots(figsize=(7, 5))

    # RA 曲线: MSE vs rate_ratio
    x = ra_m_snr['rate_ratio'].values
    y = ra_m_snr[metric].values
    y_err = ra_s_snr[metric].values if ra_s_snr is not None else None

    ax.plot(x, y, color=DATASETS['ra_trained_sa2_awgn']['color'],
            marker='o', linewidth=2, markersize=8, label='RA + Decoder (Trained)')
    if y_err is not None:
        ax.fill_between(x, y - y_err, y + y_err,
                        color=DATASETS['ra_trained_sa2_awgn']['color'],
                        alpha=0.15, linewidth=0)

    # 在每个点标注 MSE 值
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.4f}', (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8, color='#555555')

    # 基线水平参考线
    if not bl_ref.empty:
        bl_val = bl_ref[metric].values[0]
        ax.axhline(y=bl_val, color='#333333', linestyle='--', linewidth=1.5,
                   label=f'Baseline (No Adapt, rate=1.0): MSE={bl_val:.4f}')

    ax.set_xlabel('Rate Ratio (compression)')
    ax.set_ylabel('MSE')
    ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — Rate-Quality Tradeoff (SNR={snr_fixed}dB)',
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0])

    fig.tight_layout()
    path = FIG_DIR / 'ra_rate_tradeoff.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 6: RA 训练后三层对比
# ============================================================

def fig_ra_three_layer():
    """
    三个子图 (SA1/SA2/SA3)，每张图两条曲线:
      - 基线 (rate=1.0, No Adapt)
      - RA 训练 + 解码器 (rate=1.0)
    横轴: SNR, 纵轴: MSE
    证明: RA 训练在 rate=1.0 时也能提供信道保护（通过解码器的去噪能力）
    """
    metric = 'mse'

    bl_mean, bl_std = load_and_aggregate('baseline_awgn')
    ra_merged_mean, ra_merged_std = merge_ra_datasets()

    if bl_mean is None or ra_merged_mean is None:
        print("  [跳过] RA Three Layer — 数据缺失")
        return

    fig, axes = _three_layer_subplots()

    for i, layer in enumerate(['sa1', 'sa2', 'sa3']):
        ax = axes[i]

        # 基线
        bl_m, bl_s = filter_layer(bl_mean, bl_std, layer)
        bl_m, bl_s = filter_rate(bl_m, bl_s, 1.0)

        _plot_with_error(ax, bl_m['snr'], bl_m[metric],
                         bl_s[metric] if bl_s is not None else None,
                         label='Baseline (No Adapt)',
                         color=DATASETS['baseline_awgn']['color'],
                         linestyle='--', marker='s')

        # RA 训练后
        ra_m, ra_s = filter_layer(ra_merged_mean, ra_merged_std, layer)
        ra_m, ra_s = filter_rate(ra_m, ra_s, 1.0)

        if not ra_m.empty:
            _plot_with_error(ax, ra_m['snr'], ra_m[metric],
                             ra_s[metric] if ra_s is not None else None,
                             label='RA + Decoder (Trained)',
                             color=DATASETS['ra_trained_sa2_awgn']['color'],
                             linestyle='-', marker='^')

        _add_metric_label(ax, metric)
        ax.set_title(LAYER_INFO[layer]['name_cn'], fontweight='bold')
        ax.legend(loc='upper right')

    fig.suptitle('AWGN Channel: RA Adaptation vs Baseline (rate=1.0, All Layers)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'ra_three_layer_mse.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 7: 带宽-质量散点图
# ============================================================

def fig_bandwidth_quality():
    """
    散点图: x = 每样本有效带宽 (KB), y = MSE (SNR=10dB)
    每种实验条件一个点，颜色编码方法 (Baseline/SA/RA)。
    连线标注帕累托前沿。
    直接对应论文标题中的"带宽受限"。
    """
    metric = 'mse'
    snr_fixed = 10

    # 收集所有有效数据集在 SNR=10 的数据
    points = []  # list of dicts: {label, group, layer, rate_ratio, bandwidth_kb, mse}

    datasets_to_load = [
        ('baseline_awgn', 'Baseline'),
        ('sa_trained_awgn', 'SA'),
        ('ra_trained_sa2_awgn', 'RA'),
        ('ra_trained_sa1_sa3_awgn', 'RA'),
        ('sara_joint_awgn', 'SA+RA'),
        ('sara_joint_rayleigh', 'SA+RA'),
    ]

    for ds_name, group_label in datasets_to_load:
        mean_df, std_df = load_and_aggregate(ds_name)
        if mean_df is None:
            continue

        df_snr = mean_df[mean_df['snr'] == snr_fixed].copy()
        if df_snr.empty:
            continue

        for _, row in df_snr.iterrows():
            bw_kb = compute_per_sample_kb(row)
            points.append({
                'label': group_label,
                'group': group_label,
                'layer': row['layer'],
                'rate_ratio': row['rate_ratio'],
                'bandwidth_kb': bw_kb,
                'mse': row[metric],
            })

    if not points:
        print("  [跳过] Bandwidth-Quality — 无可用数据")
        return

    pts_df = pd.DataFrame(points)

    # 颜色
    color_map = {
        'Baseline': '#555555',
        'SA': DATASETS['sa_trained_awgn']['color'],
        'RA': DATASETS['ra_trained_sa2_awgn']['color'],
        'SA+RA': DATASETS['sara_joint_awgn']['color'],
    }
    marker_map = {
        'sa1': 'o',
        'sa2': 's',
        'sa3': '^',
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for group in ['Baseline', 'SA', 'RA', 'SA+RA']:
        subset = pts_df[pts_df['group'] == group]
        if subset.empty:
            continue

        for layer in ['sa1', 'sa2', 'sa3']:
            sub = subset[subset['layer'] == layer]
            if sub.empty:
                continue

            marker = marker_map.get(layer, 'o')
            ax.scatter(sub['bandwidth_kb'], sub['mse'],
                       c=color_map[group], marker=marker, s=60,
                       label=f'{group} — {LAYER_INFO[layer]["name"]}' if group == 'Baseline' else f'{group} — {layer.upper()}',
                       edgecolors='white', linewidth=0.5, zorder=3)

    ax.set_xlabel('Effective Bandwidth per Sample (KB)')
    ax.set_ylabel('MSE')
    ax.set_title(f'Bandwidth vs Quality Tradeoff (SNR={snr_fixed}dB)',
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 标注: 左下 = 更好（低带宽 + 低MSE）
    ax.annotate('Better\n(Less BW + Lower MSE)', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=8, color='#2ca02c',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d9f0d9', alpha=0.8))

    fig.tight_layout()
    path = FIG_DIR / 'bandwidth_quality.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 8: 综合摘要 2×2 网格
# ============================================================

def fig_summary():
    """
    一张 2×2 综合图，覆盖论文核心卖点:
      左上: SA2 AWGN — SA vs Baseline (MSE)
      右上: SA2 AWGN — RA 不同速率比 (MSE)
      左下: SA2 — AWGN vs Rayleigh (SA+Decoder only, MSE)
      右下: Rate-Quality 折衷 (MSE vs rate_ratio, SNR=10dB)
    """
    layer = 'sa2'
    metric = 'mse'
    snr_fixed = 10

    # 加载所有数据
    bl_awgn_m, bl_awgn_s = load_and_aggregate('baseline_awgn')
    sa_awgn_m, sa_awgn_s = load_and_aggregate('sa_trained_awgn')
    bl_ray_m, bl_ray_s = load_and_aggregate('baseline_rayleigh')
    sa_ray_m, sa_ray_s = load_and_aggregate('sa_trained_rayleigh')
    ra_m, ra_s = load_and_aggregate('ra_trained_sa2_awgn')
    sara_m, sara_s = load_and_aggregate('sara_joint_awgn')

    if any(x is None for x in [bl_awgn_m, sa_awgn_m, ra_m]):
        print("  [跳过] Summary — 数据缺失")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- 左上: SA vs Baseline (AWGN) ----
    ax = axes[0, 0]
    bl_m, bl_s = filter_layer(bl_awgn_m, bl_awgn_s, layer)
    bl_m, bl_s = filter_rate(bl_m, bl_s, 1.0)
    sa_m, sa_s = filter_layer(sa_awgn_m, sa_awgn_s, layer)
    sa_m, sa_s = filter_rate(sa_m, sa_s, 1.0)

    _plot_with_error(ax, bl_m['snr'], bl_m[metric],
                     bl_s[metric] if bl_s is not None else None,
                     'Baseline', '#555555', '--', 's')
    _plot_with_error(ax, sa_m['snr'], sa_m[metric],
                     sa_s[metric] if sa_s is not None else None,
                     'SA + Decoder', DATASETS['sa_trained_awgn']['color'], '-', 'o')
    _add_metric_label(ax, metric)
    ax.set_title('(a) SA Adaptation — AWGN, SA2', fontweight='bold')
    ax.legend(loc='upper right')

    # ---- 右上: RA 不同速率比 ----
    ax = axes[0, 1]
    bl_m2, bl_s2 = filter_layer(bl_awgn_m, bl_awgn_s, layer)
    bl_m2, bl_s2 = filter_rate(bl_m2, bl_s2, 1.0)
    _plot_with_error(ax, bl_m2['snr'], bl_m2[metric],
                     bl_s2[metric] if bl_s2 is not None else None,
                     'Baseline (rate=1.0)', '#333333', '--', 's', linewidth=1.5)

    ra_m2, ra_s2 = filter_layer(ra_m, ra_s, layer)
    for rr in [0.2, 0.5, 0.8, 1.0]:
        ra_m_rr, ra_s_rr = filter_rate(ra_m2, ra_s2.copy() if ra_s2 is not None else None, rr)
        if ra_m_rr.empty:
            continue
        _plot_with_error(ax, ra_m_rr['snr'], ra_m_rr[metric],
                         ra_s_rr[metric] if ra_s_rr is not None else None,
                         f'RA rate={rr}', RATE_COLORS[rr], '-', 'o', markersize=3, linewidth=1.2)
    _add_metric_label(ax, metric)
    ax.set_title('(b) RA Rate Adaptation — AWGN, SA2', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)

    # ---- 左下: AWGN vs Rayleigh ----
    ax = axes[1, 0]
    if bl_ray_m is not None and sa_ray_m is not None:
        for ds_name, mean_df, std_df, label in [
            ('baseline_awgn', bl_awgn_m, bl_awgn_s, 'AWGN Baseline'),
            ('sa_trained_awgn', sa_awgn_m, sa_awgn_s, 'AWGN SA'),
            ('baseline_rayleigh', bl_ray_m, bl_ray_s, 'Rayleigh Baseline'),
            ('sa_trained_rayleigh', sa_ray_m, sa_ray_s, 'Rayleigh SA'),
        ]:
            m, s = filter_layer(mean_df, std_df, layer)
            m, s = filter_rate(m, s, 1.0)
            if m.empty:
                continue
            info = DATASETS[ds_name]
            is_base = 'Baseline' in label
            _plot_with_error(ax, m['snr'], m[metric],
                             s[metric] if s is not None else None,
                             label, info['color'], '--' if is_base else '-',
                             's' if is_base else 'o')
    _add_metric_label(ax, metric)
    ax.set_title('(c) Channel Robustness — SA2', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)

    # ---- 右下: Rate-Quality 折衷 (RA + SA+RA) ----
    ax = axes[1, 1]
    ra_m3, ra_s3 = filter_layer(ra_m, ra_s, layer)
    ra_m_snr = ra_m3[ra_m3['snr'] == snr_fixed].sort_values('rate_ratio')
    ra_s_snr = ra_s3[ra_s3['snr'] == snr_fixed].sort_values('rate_ratio') if ra_s3 is not None else None

    x = ra_m_snr['rate_ratio'].values
    y_ra = ra_m_snr[metric].values
    y_err_ra = ra_s_snr[metric].values if ra_s_snr is not None else None

    ax.plot(x, y_ra, color=DATASETS['ra_trained_sa2_awgn']['color'],
            marker='o', linewidth=2, markersize=6, label='RA + Decoder')
    if y_err_ra is not None:
        ax.fill_between(x, y_ra - y_err_ra, y_ra + y_err_ra,
                        color=DATASETS['ra_trained_sa2_awgn']['color'],
                        alpha=0.12, linewidth=0)

    # SA+RA Joint 也加入折衷图
    if sara_m is not None:
        sara_m3, sara_s3 = filter_layer(sara_m, sara_s, layer)
        sara_m_snr = sara_m3[sara_m3['snr'] == snr_fixed].sort_values('rate_ratio')
        sara_s_snr = sara_s3[sara_s3['snr'] == snr_fixed].sort_values('rate_ratio') if sara_s3 is not None else None
        x2 = sara_m_snr['rate_ratio'].values
        y_sara = sara_m_snr[metric].values
        y_err_sara = sara_s_snr[metric].values if sara_s_snr is not None else None
        ax.plot(x2, y_sara, color=DATASETS['sara_joint_awgn']['color'],
                marker='D', linewidth=2, markersize=6, label='SA+RA Joint')
        if y_err_sara is not None:
            ax.fill_between(x2, y_sara - y_err_sara, y_sara + y_err_sara,
                            color=DATASETS['sara_joint_awgn']['color'],
                            alpha=0.12, linewidth=0)

    bl_m3, bl_s3 = filter_layer(bl_awgn_m, bl_awgn_s, layer)
    bl_m3, bl_s3 = filter_rate(bl_m3, bl_s3, 1.0)
    bl_ref = bl_m3[bl_m3['snr'] == snr_fixed]
    if not bl_ref.empty:
        ax.axhline(y=bl_ref[metric].values[0], color='#333333', linestyle='--',
                   linewidth=1.5, label=f'Baseline (No Adapt)')

    ax.set_xlabel('Rate Ratio')
    ax.set_ylabel('MSE')
    ax.set_title(f'(d) Rate-Quality Tradeoff (SNR={snr_fixed}dB, SA2)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Point Cloud Feature Transmission: Adaptive Modulation Summary',
                 fontweight='bold', fontsize=14, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / 'summary.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 8: SA+RA vs SA vs RA vs Baseline 四方对比
# ============================================================

def fig_four_way_comparison():
    """
    1×2 子图 (MSE | PSNR)，SA2 层，rate=1.0。
    四条曲线: Baseline / SA+Decoder / RA+Decoder / SA+RA Joint+Decoder
    一张图讲清楚所有方法的优劣排序。
    """
    layer = 'sa2'

    bl_m, bl_s = load_and_aggregate('baseline_awgn')
    sa_m, sa_s = load_and_aggregate('sa_trained_awgn')
    ra_m, ra_s = load_and_aggregate('ra_trained_sa2_awgn')
    sara_m, sara_s = load_and_aggregate('sara_joint_awgn')

    if any(x is None for x in [bl_m, sa_m, ra_m, sara_m]):
        print("  [跳过] Four-Way Comparison — 数据缺失")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = [
        ('baseline_awgn', bl_m, bl_s, 'Baseline (No Adapt)'),
        ('sa_trained_awgn', sa_m, sa_s, 'SA + Decoder'),
        ('ra_trained_sa2_awgn', ra_m, ra_s, 'RA + Decoder'),
        ('sara_joint_awgn', sara_m, sara_s, 'SA+RA Joint + Decoder'),
    ]

    for ax, metric in zip(axes, ['mse', 'psnr']):
        for ds_name, mean_df, std_df, custom_label in methods:
            m, s = filter_layer(mean_df, std_df, layer)
            m, s = filter_rate(m, s, 1.0)
            if m.empty:
                continue
            info = DATASETS[ds_name]
            is_base = 'Baseline' in custom_label
            _plot_with_error(ax, m['snr'], m[metric],
                             s[metric] if s is not None else None,
                             label=custom_label,
                             color=info['color'],
                             linestyle='--' if is_base else '-',
                             marker=info['marker'],
                             linewidth=2 if is_base else 1.5)

        _add_metric_label(ax, metric)
        ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — {metric.upper()}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)

    fig.suptitle('AWGN Channel: Full Method Comparison (SA2, rate=1.0)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'four_way_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 9: SA+RA Joint 速率扫描 — 证明 SA 在联合系统中的价值
# ============================================================

def fig_sara_rate_sweep():
    """
    1×2 子图 (MSE | PSNR)，SA2 层。
    5条曲线: Baseline + SA+RA Joint 4个速率比。
    证明: SA+RA 联合系统在各速率下均有效，且远超基线。
    """
    layer = 'sa2'
    metric_pairs = [('mse', 0), ('psnr', 1)]
    rate_ratios = [0.2, 0.5, 0.8, 1.0]

    bl_m, bl_s = load_and_aggregate('baseline_awgn')
    sara_m, sara_s = load_and_aggregate('sara_joint_awgn')

    if bl_m is None or sara_m is None:
        print("  [跳过] SA+RA Rate Sweep — 数据缺失")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, _) in enumerate(metric_pairs):
        ax = axes[ax_idx]

        # 基线
        bl_ml, bl_sl = filter_layer(bl_m, bl_s, layer)
        bl_ml, bl_sl = filter_rate(bl_ml, bl_sl, 1.0)
        _plot_with_error(ax, bl_ml['snr'], bl_ml[metric],
                         bl_sl[metric] if bl_sl is not None else None,
                         label='Baseline (No Adapt, rate=1.0)',
                         color='#333333', linestyle='--', marker='s',
                         linewidth=2, alpha=0.10)

        # SA+RA 各速率比
        sara_ml, sara_sl = filter_layer(sara_m, sara_s, layer)
        for rr in rate_ratios:
            m_rr, s_rr = filter_rate(sara_ml, sara_sl.copy() if sara_sl is not None else None, rr)
            if m_rr.empty:
                continue
            _plot_with_error(ax, m_rr['snr'], m_rr[metric],
                             s_rr[metric] if s_rr is not None else None,
                             label=f'SA+RA rate={rr}',
                             color=RATE_COLORS[rr],
                             linestyle='-', marker='D', markersize=3, linewidth=1.2)

        _add_metric_label(ax, metric)
        ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — {metric.upper()}', fontweight='bold')
        ax.legend(loc='best', fontsize=7)

    fig.suptitle('AWGN Channel: SA+RA Joint Performance at Different Compression Rates',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sara_rate_sweep.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 10: SA+RA Joint Rayleigh 速率扫描
# ============================================================

def fig_sara_rate_sweep_rayleigh():
    """
    1×2 子图 (MSE | PSNR)，SA2 层，Rayleigh 衰落信道。
    5条曲线: Baseline (Rayleigh) + SA+RA Joint 4个速率比。
    证明: SA+RA 联合系统在 Rayleigh 衰落信道下各速率仍远超基线。
    """
    layer = 'sa2'
    metric_pairs = [('mse', 0), ('psnr', 1)]
    rate_ratios = [0.2, 0.5, 0.8, 1.0]

    bl_m, bl_s = load_and_aggregate('baseline_rayleigh')
    sara_m, sara_s = load_and_aggregate('sara_joint_rayleigh')

    if bl_m is None or sara_m is None:
        print("  [跳过] SA+RA Rate Sweep Rayleigh — 数据缺失")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, _) in enumerate(metric_pairs):
        ax = axes[ax_idx]

        # 基线
        bl_ml, bl_sl = filter_layer(bl_m, bl_s, layer)
        bl_ml, bl_sl = filter_rate(bl_ml, bl_sl, 1.0)
        _plot_with_error(ax, bl_ml['snr'], bl_ml[metric],
                         bl_sl[metric] if bl_sl is not None else None,
                         label='Baseline (No Adapt, rate=1.0)',
                         color='#333333', linestyle='--', marker='s',
                         linewidth=2, alpha=0.10)

        # SA+RA 各速率比
        sara_ml, sara_sl = filter_layer(sara_m, sara_s, layer)
        for rr in rate_ratios:
            m_rr, s_rr = filter_rate(sara_ml, sara_sl.copy() if sara_sl is not None else None, rr)
            if m_rr.empty:
                continue
            _plot_with_error(ax, m_rr['snr'], m_rr[metric],
                             s_rr[metric] if s_rr is not None else None,
                             label=f'SA+RA rate={rr}',
                             color=RATE_COLORS[rr],
                             linestyle='-', marker='D', markersize=3, linewidth=1.2)

        _add_metric_label(ax, metric)
        ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — {metric.upper()}', fontweight='bold')
        ax.legend(loc='best', fontsize=7)

    fig.suptitle('Rayleigh Channel: SA+RA Joint Performance at Different Compression Rates',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sara_rate_sweep_rayleigh.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 11: SA+RA vs RA 对比 — 证明 SA 组件在联合系统中的增益
# ============================================================

def fig_sara_vs_ra():
    """
    固定 SNR=10dB，SA2 层。对比 RA 和 SA+RA 在各速率比下的 MSE。
    核心叙事: SA 组件在联合系统中提供了额外的 SNR 保护，
    且在低速率下增益最大（因为此时 SNR 保护更关键）。
    """
    metric = 'mse'
    snr_fixed = 10
    layer = 'sa2'

    ra_m, ra_s = load_and_aggregate('ra_trained_sa2_awgn')
    sara_m, sara_s = load_and_aggregate('sara_joint_awgn')

    if ra_m is None or sara_m is None:
        print("  [跳过] SA+RA vs RA — 数据缺失")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    rate_ratios = [0.2, 0.5, 0.8, 1.0]
    ra_values = []
    sara_values = []

    ra_ml, ra_sl = filter_layer(ra_m, ra_s, layer)
    sara_ml, sara_sl = filter_layer(sara_m, sara_s, layer)

    for rr in rate_ratios:
        ra_sub = ra_ml[(ra_ml['snr'] == snr_fixed) & (ra_ml['rate_ratio'] == rr)]
        sara_sub = sara_ml[(sara_ml['snr'] == snr_fixed) & (sara_ml['rate_ratio'] == rr)]
        if not ra_sub.empty:
            ra_values.append(ra_sub[metric].values[0])
        else:
            ra_values.append(None)
        if not sara_sub.empty:
            sara_values.append(sara_sub[metric].values[0])
        else:
            sara_values.append(None)

    x_pos = np.arange(len(rate_ratios))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, ra_values, width,
                   label='RA + Decoder',
                   color=DATASETS['ra_trained_sa2_awgn']['color'],
                   edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, sara_values, width,
                   label='SA+RA Joint + Decoder',
                   color=DATASETS['sara_joint_awgn']['color'],
                   edgecolor='white', linewidth=0.8)

    # 标注改善百分比
    for i, (rv, sv) in enumerate(zip(ra_values, sara_values)):
        if rv is not None and sv is not None:
            improvement = (rv - sv) / rv * 100
            ax.annotate(f'↓{improvement:.0f}%', (x_pos[i] + width/2, sv),
                        textcoords="offset points", xytext=(0, -18),
                        ha='center', fontsize=9, fontweight='bold',
                        color=DATASETS['sara_joint_awgn']['color'])

    # 基线参考线
    bl_m, bl_s = load_and_aggregate('baseline_awgn')
    if bl_m is not None:
        bl_ml, bl_sl = filter_layer(bl_m, bl_s, layer)
        bl_ml, bl_sl = filter_rate(bl_ml, bl_sl, 1.0)
        bl_ref = bl_ml[bl_ml['snr'] == snr_fixed]
        if not bl_ref.empty:
            bl_val = bl_ref[metric].values[0]
            ax.axhline(y=bl_val, color='#333333', linestyle='--', linewidth=1.5,
                       label=f'Baseline (No Adapt, rate=1.0): MSE={bl_val:.4f}')

    ax.set_xlabel('Rate Ratio')
    ax.set_ylabel('MSE')
    ax.set_title(f'{LAYER_INFO[layer]["name_cn"]} — SA+RA vs RA (SNR={snr_fixed}dB)',
                 fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(rr) for rr in rate_ratios])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = FIG_DIR / 'sara_vs_ra.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ============================================================
# 图表 12: 六方法 MSE 对比 (SA3, rate=0.5)
# ============================================================

def fig_six_way_mse():
    """SA3 层，6 条 MSE vs SNR 曲线（rate=0.5）。展示所有方法的特征保真度排序。"""
    metric = 'mse'; layer = 'sa3'; rr = 0.5

    agg_m, _ = load_baselines_feature_metric(metric)
    if agg_m is None:
        print("  [跳过] Six-Way MSE — 数据缺失"); return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        sub = agg_m[(agg_m['method'] == method) & (agg_m['rate_ratio'] == rr)]
        if sub.empty: continue
        ax.semilogy(sub['snr'], sub[metric],
                    color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                    linestyle=METHOD_LINESTYLES[method], linewidth=2, label=method)

    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('MSE (log scale)')
    ax.set_title(f'SA3 Feature-Level MSE: 6 Methods (rate=0.5)', fontweight='bold')
    ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / 'six_way_mse.png', dpi=DPI)
    plt.close(fig); print(f"  [OK] six_way_mse.png")


# ============================================================
# 图表 13: 六方法分类精度对比 (SA3, rate=0.5) ⭐ 论文核心图
# ============================================================

def fig_six_way_accuracy():
    """SA3 层，6 条 Accuracy vs SNR 曲线（rate=0.5）。展示梯级崩塌效应。"""
    agg_m, _ = load_baselines_classification()
    if agg_m is None:
        print("  [跳过] Six-Way Accuracy — 数据缺失"); return
    rr = 0.5

    clean_acc = 0.9748  # 干净特征上界

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        sub = agg_m[(agg_m['method'] == method) & (agg_m['rate_ratio'] == rr)]
        if sub.empty: continue
        ax.plot(sub['snr'], sub['accuracy'],
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method], linewidth=2, label=method)

    ax.axhline(y=clean_acc, color='#333333', ls=':', lw=1.5, label=f'Clean ({clean_acc:.3f})')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Classification Accuracy')
    ax.set_title(f'SA3 Task-Level Accuracy: 6 Methods (rate=0.5)', fontweight='bold')
    ax.legend(ncol=2, fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(FIG_DIR / 'six_way_accuracy.png', dpi=DPI)
    plt.close(fig); print(f"  [OK] six_way_accuracy.png")


# ============================================================
# 图表 14: 梯级崩塌图 ⭐ 论文核心卖点图
# ============================================================

def fig_stair_step_collapse():
    """
    SA3, SNR=10dB, rate=0.5。X 轴 = 分类精度，Y 轴 = MSE（对数）。
    6 个散点标注方法名 + 连线展示梯级崩塌方向。
    核心叙事：MSE 越低（越左），分类精度越差（越右）— 完全反相关。
    """
    mse_m, _ = load_baselines_feature_metric('mse')
    cls_m, _ = load_baselines_classification()
    if mse_m is None or cls_m is None:
        print("  [跳过] Stair-Step Collapse — 数据缺失"); return
    snr_fixed = 10; rr = 0.5

    points = []
    for method in METHOD_ORDER:
        mse_sub = mse_m[(mse_m['method'] == method) & (mse_m['snr'] == snr_fixed) & (mse_m['rate_ratio'] == rr)]
        cls_sub = cls_m[(cls_m['method'] == method) & (cls_m['snr'] == snr_fixed) & (cls_m['rate_ratio'] == rr)]
        if mse_sub.empty or cls_sub.empty: continue
        points.append((method, float(mse_sub['mse'].iloc[0]), float(cls_sub['accuracy'].iloc[0])))

    fig, ax = plt.subplots(figsize=(10, 6))

    xs = [p[1] for p in points]; ys = [p[2] for p in points]
    ax.scatter(xs, ys, c=[METHOD_COLORS[p[0]] for p in points], s=200,
               edgecolors='white', linewidth=1.5, zorder=5)

    for i, (method, mse_v, cls_v) in enumerate(points):
        offset = (12, 12) if i % 2 == 0 else (12, -18)
        ax.annotate(method, (mse_v, cls_v), textcoords="offset points",
                    xytext=offset, ha='center', fontsize=10, fontweight='bold',
                    color=METHOD_COLORS[method])

    # 梯级连线
    sorted_pts = sorted(points, key=lambda p: p[1])
    ax.plot([p[1] for p in sorted_pts], [p[2] for p in sorted_pts],
            color='#888888', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    # 箭头标注
    mid_pts = sorted_pts[1:4]
    for pt in mid_pts:
        ax.annotate('', xy=(pt[1], pt[2]),
                    xytext=(pt[1]+0.02, pt[2]-0.05),
                    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5))

    ax.axhline(y=0.9748, color='#333333', ls=':', lw=1, alpha=0.5, label='Clean Acc = 97.5%')
    ax.set_xlabel('Feature MSE (lower = better feature quality)')
    ax.set_ylabel('Classification Accuracy (higher = better task quality)')
    ax.set_title(f'Stair-Step Collapse: MSE vs Task Accuracy (SNR={snr_fixed}dB, rate=0.5)',
                 fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 象限标注
    ax.annotate('Better\n(MSE low + Acc high)', xy=(0.01, 0.98), xycoords='axes fraction',
                fontsize=9, color='#2ca02c', ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='#d9f0d9', alpha=0.6))
    ax.annotate('Worse\n(MSE low but Acc low)', xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=9, color='#d62728', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='#fdd', alpha=0.6))

    fig.tight_layout(); fig.savefig(FIG_DIR / 'stair_step_collapse.png', dpi=DPI)
    plt.close(fig); print(f"  [OK] stair_step_collapse.png")


# ============================================================
# 图表 15: 6 方法 Summary 2x3 网格 ⭐ 论文摘要图/海报图
# ============================================================

def fig_full_summary_2x3():
    """2 行 3 列，覆盖论文全部核心发现。"""
    layer = 'sa3'; rr = 0.5; snr_fixed = 10

    mse_m, _ = load_baselines_feature_metric('mse')
    cls_m, _ = load_baselines_classification()
    sa_m, sa_s = load_and_aggregate('sa_trained_awgn')
    sara_m, sara_s = load_and_aggregate('sara_joint_awgn')

    if any(x is None for x in [mse_m, cls_m]):
        print("  [跳过] Full Summary 2x3 — 数据缺失"); return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (1,1) 六方法 MSE
    ax = axes[0, 0]
    for method in METHOD_ORDER:
        sub = mse_m[(mse_m['method'] == method) & (mse_m['rate_ratio'] == rr)]
        if sub.empty: continue
        ax.semilogy(sub['snr'], sub['mse'], color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method], ls=METHOD_LINESTYLES[method], lw=1.5, label=method)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('MSE (log)')
    ax.set_title('(a) Feature-Level MSE (SA3, rate=0.5)', fontweight='bold')
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    # (1,2) 六方法 Accuracy
    ax = axes[0, 1]
    for method in METHOD_ORDER:
        sub = cls_m[(cls_m['method'] == method) & (cls_m['rate_ratio'] == rr)]
        if sub.empty: continue
        ax.plot(sub['snr'], sub['accuracy'], color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], ls=METHOD_LINESTYLES[method], lw=1.5, label=method)
    ax.axhline(y=0.9748, color='#333', ls=':', lw=1)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Accuracy')
    ax.set_title('(b) Task-Level Accuracy (SA3, rate=0.5)', fontweight='bold')
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # (1,3) 梯级崩塌（MSE-Acc 散点）
    ax = axes[0, 2]
    pts = []
    for method in METHOD_ORDER:
        m_sub = mse_m[(mse_m['method'] == method) & (mse_m['snr'] == snr_fixed) & (mse_m['rate_ratio'] == rr)]
        c_sub = cls_m[(cls_m['method'] == method) & (cls_m['snr'] == snr_fixed) & (cls_m['rate_ratio'] == rr)]
        if m_sub.empty or c_sub.empty: continue
        pts.append((method, float(m_sub['mse'].iloc[0]), float(c_sub['accuracy'].iloc[0])))
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.scatter(xs, ys, c=[METHOD_COLORS[p[0]] for p in pts], s=120, edgecolors='white', lw=1)
    for method, xv, yv in pts:
        ax.annotate(method, (xv, yv), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=7, fontweight='bold', color=METHOD_COLORS[method])
    ax.set_xlabel('MSE'); ax.set_ylabel('Accuracy')
    ax.set_title(f'(c) MSE-Accuracy Tradeoff (SNR={snr_fixed}dB)', fontweight='bold')
    ax.set_xscale('log'); ax.grid(alpha=0.3)

    # (2,1) SA+RA 速率扫描（MSE）
    ax = axes[1, 0]
    if sara_m is not None:
        s_m, s_s = filter_layer(sara_m, sara_s, 'sa2')
        for r in [0.2, 0.5, 0.8, 1.0]:
            sm, _ = filter_rate(s_m, s_s, r)
            if sm.empty: continue
            ax.plot(sm['snr'], sm['mse'], color=RATE_COLORS[r], marker='o', ms=3, lw=1.2, label=f'rate={r}')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('MSE')
    ax.set_title('(d) SA+RA Rate Sweep (SA2, MSE)', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (2,2) Accuracy vs Rate Ratio
    ax = axes[1, 1]
    for method in ['NoAdapt', 'Plain-JSCC', 'SA-only', 'SA+RA+MSE']:
        sub = cls_m[(cls_m['method'] == method) & (cls_m['snr'] == snr_fixed)]
        if sub.empty: continue
        grp = sub.groupby('rate_ratio')['accuracy'].mean()
        ax.plot(grp.index, grp.values, color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], lw=1.5, label=method)
    ax.axhline(y=0.9748, color='#333', ls=':', lw=1)
    ax.set_xlabel('Rate Ratio'); ax.set_ylabel('Accuracy')
    ax.set_title(f'(e) Accuracy vs Bandwidth (SNR={snr_fixed}dB)', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_xticks([0.2, 0.5, 0.8, 1.0])

    # (2,3) 修复实验对比（Phase A/A2/B）
    ax = axes[1, 2]
    phases = ['SA+RA+MSE\n(original)', 'Phase A\n(decoder only)', 'Phase A2\n(all 100M)', 'Phase B\n(CE decoder)']
    acc_vals = [0.2796, 0.2738, 0.2839, 0.1888]
    colors_phases = ['#009E73', '#2B7BBD', '#E68613', '#882255']
    bars = ax.bar(phases, acc_vals, color=colors_phases, edgecolor='white', lw=0.8)
    ax.axhline(y=0.9571, color='#555555', ls='--', lw=1.5, label='NoAdapt = 95.7%')
    for b, v in zip(bars, acc_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Accuracy @ 0dB'); ax.set_title('(f) Repair Experiments (all failed)', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, 1.0)

    fig.suptitle('3D Point Cloud Semantic Feature Transmission: Complete Evaluation',
                 fontweight='bold', fontsize=15, y=1.01)
    fig.tight_layout(); fig.savefig(FIG_DIR / 'full_summary_2x3.png', dpi=DPI)
    plt.close(fig); print(f"  [OK] full_summary_2x3.png")


# ============================================================
# 图表注册表
# ============================================================

FIG_REGISTRY = {
    'sa': {
        'functions': [fig_sa_awgn_mse, fig_sa_awgn_cosine, fig_sa_channel_comparison],
        'description': 'SA (SNR-Adaptive) modulation experiments',
    },
    'ra': {
        'functions': [fig_ra_rate_sweep, fig_ra_rate_tradeoff, fig_ra_three_layer],
        'description': 'RA (Rate-Adaptive) modulation experiments',
    },
    'sara': {
        'functions': [
            fig_sara_rate_sweep, fig_sara_rate_sweep_rayleigh,
            fig_sara_channel_comparison, fig_sara_vs_ra,
            fig_four_way_comparison,
        ],
        'description': 'SA+RA Joint modulation (AWGN + Rayleigh) + cross-method comparisons',
    },
    'baselines': {
        'functions': [fig_six_way_mse, fig_six_way_accuracy, fig_stair_step_collapse],
        'description': '6-Method comparison with external baselines (Plain-JSCC + Quant)',
    },
    'combined': {
        'functions': [fig_bandwidth_quality],
        'description': 'Cross-method bandwidth-quality tradeoff',
    },
    'summary': {
        'functions': [fig_summary, fig_full_summary_2x3],
        'description': 'Summary grids (2x2 + 2x3) for paper overview',
    },
}


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='统一图表生成 — 点云语义通信实验结果可视化'
    )
    parser.add_argument(
        '--group', type=str, default='all',
        choices=['all', 'sa', 'ra', 'sara', 'baselines', 'combined', 'summary'],
        help='要生成的图表组 (默认 all)'
    )
    args = parser.parse_args()

    if args.group == 'all':
        groups_to_run = list(FIG_REGISTRY.keys())
    else:
        groups_to_run = [args.group]

    print(f"\n{'='*60}")
    print(f"Figure generation starting — output: {FIG_DIR}")
    print(f"{'='*60}\n")

    total_ok = 0
    total_skip = 0

    for group_name in groups_to_run:
        group = FIG_REGISTRY[group_name]
        print(f"\n--- {group_name.upper()}: {group['description']} ---")
        for func in group['functions']:
            func()

    print(f"\n{'='*60}")
    print(f"Done. Figures saved to: {FIG_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
