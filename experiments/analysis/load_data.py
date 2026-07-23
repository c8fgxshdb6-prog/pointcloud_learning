"""
统一数据加载模块
===============
管理所有实验CSV文件的读取、聚合、元数据。
新增一种实验时，只需在此注册数据源即可被所有绘图函数使用。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# ============================================================
# 路径配置
# ============================================================
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _BASE_DIR / 'results'

# ============================================================
# 数据源注册表
# ============================================================
# 每个数据源定义: 文件名、实验描述、方法标签、信道类型等
DATASETS = {
    'baseline_awgn': {
        'file': 'no_adapt_results.csv',
        'label': 'Baseline (No Adapt)',
        'channel': 'AWGN',
        'modulation': 'None',
        'group': 'baseline',       # 用于分组配色
        'linestyle': '--',
        'marker': 's',
        'color': '#555555',        # 灰色 = 基线
    },
    'baseline_rayleigh': {
        'file': 'no_adapt_results_rayleigh.csv',
        'label': 'Baseline (Rayleigh)',
        'channel': 'Rayleigh',
        'modulation': 'None',
        'group': 'baseline',
        'linestyle': '--',
        'marker': 's',
        'color': '#555555',
    },
    'sa_trained_awgn': {
        'file': 'channel_exp_results_trained_with_decoder.csv',
        'label': 'SA + Decoder (AWGN)',
        'channel': 'AWGN',
        'modulation': 'SA',
        'group': 'sa',
        'linestyle': '-',
        'marker': 'o',
        'color': '#2B7BBD',        # 蓝色 = SA 自适应
    },
    'sa_trained_rayleigh': {
        'file': 'channel_exp_results_rayleigh_trained_with_decoder.csv',
        'label': 'SA + Decoder (Rayleigh)',
        'channel': 'Rayleigh',
        'modulation': 'SA',
        'group': 'sa',
        'linestyle': '-',
        'marker': 'o',
        'color': '#2B7BBD',
    },
    'ra_trained_sa2_awgn': {
        'file': 'channel_exp_results_ra_trained.csv',
        'label': 'RA + Decoder',
        'channel': 'AWGN',
        'modulation': 'RA',
        'group': 'ra',
        'linestyle': '-',
        'marker': '^',
        'color': '#E68613',        # 橙色 = RA 自适应
        'layers_available': ['sa2'],
    },
    'ra_trained_sa1_sa3_awgn': {
        'file': 'channel_exp_results_ra_trained_sa1_sa3.csv',
        'label': 'RA + Decoder',
        'channel': 'AWGN',
        'modulation': 'RA',
        'group': 'ra',
        'linestyle': '-',
        'marker': '^',
        'color': '#E68613',
        'layers_available': ['sa1', 'sa3'],
        'metrics_available': ['mse', 'psnr', 'cos_sim'],
    },
    'sara_joint_awgn': {
        'file': 'channel_exp_results_sara_joint.csv',
        'label': 'SA+RA Joint + Decoder (AWGN)',
        'channel': 'AWGN',
        'modulation': 'SA+RA',
        'group': 'sara',
        'linestyle': '-',
        'marker': 'D',
        'color': '#009E73',        # 绿色 = SA+RA 联合
        'metrics_available': ['mse', 'psnr', 'cos_sim'],
    },
    'sara_joint_rayleigh': {
        'file': 'channel_exp_results_sara_joint_rayleigh.csv',
        'label': 'SA+RA Joint + Decoder (Rayleigh)',
        'channel': 'Rayleigh',
        'modulation': 'SA+RA',
        'group': 'sara',
        'linestyle': '-',
        'marker': 'D',
        'color': '#009E73',
        'metrics_available': ['mse', 'psnr', 'cos_sim'],
    },
    # ---- 外部基线 ----
    'plain_jscc_awgn': {
        'file': 'baselines_full_comparison.csv',  # 共享文件，由 group 筛选 method 列
        'label': 'Plain-JSCC',
        'channel': 'AWGN',
        'modulation': 'Plain-JSCC',
        'group': 'plain_jscc',
        'linestyle': '--',
        'marker': 'P',
        'color': '#D55E00',        # 深橙 = 简单 JSCC 基线
        'metrics_available': ['mse'],
        'method_filter': 'Plain-JSCC',  # 从合并 CSV 中筛选
    },
    'quant_awgn': {
        'file': 'baselines_full_comparison.csv',
        'label': 'Quant (8-bit)',
        'channel': 'AWGN',
        'modulation': 'Quant',
        'group': 'quant',
        'linestyle': ':',
        'marker': 'x',
        'color': '#CC79A7',        # 紫色 = 量化基线
        'metrics_available': ['mse'],
        'method_filter': 'Quant(8bit)',
    },
    # ---- 分类精度数据源 ----
    'cls_baselines': {
        'file': 'baselines_full_comparison.csv',
        'label': 'Classification Accuracy (6 methods)',
        'channel': 'AWGN',
        'modulation': 'All',
        'group': 'cls',
        'metrics_available': ['acc'],
        'is_classification': True,
    },
}

# ============================================================
# 层级元数据
# ============================================================
LAYER_INFO = {
    'sa1': {
        'name': 'SA1 (Local Geometry)',
        'name_cn': 'SA1 (局部几何)',
        'feat_dim': 320,      # 全通道数
        'n_points': 512,       # 每样本点数
    },
    'sa2': {
        'name': 'SA2 (Part Structure)',
        'name_cn': 'SA2 (部件结构)',
        'feat_dim': 640,
        'n_points': 128,
    },
    'sa3': {
        'name': 'SA3 (Global Semantics)',
        'name_cn': 'SA3 (全局语义)',
        'feat_dim': 1024,
        'n_points': 1,
    },
}

# ============================================================
# 指标显示配置
# ============================================================
METRIC_CONFIG = {
    'mse':    {'label': 'MSE',         'ylabel': 'Mean Squared Error',      'lower_is_better': True},
    'psnr':   {'label': 'PSNR',        'ylabel': 'PSNR (dB)',               'lower_is_better': False},
    'cos_sim': {'label': 'Cosine Sim',  'ylabel': 'Cosine Similarity',       'lower_is_better': False},
    'ssim':   {'label': 'SSIM',        'ylabel': 'Structural Similarity',   'lower_is_better': False},
    'acc':    {'label': 'Accuracy',    'ylabel': 'Classification Accuracy', 'lower_is_better': False},
}

# ============================================================
# 公共函数
# ============================================================

def load_dataset(name):
    """
    加载单个数据源。
    返回带元数据列的 DataFrame。
    如果文件不存在，返回 None 并发出警告。
    """
    if name not in DATASETS:
        raise KeyError(f"未知数据源: {name}。可用: {list(DATASETS.keys())}")

    info = DATASETS[name]
    path = RESULTS_DIR / info['file']

    if not path.exists():
        warnings.warn(f"数据文件不存在: {path}。跳过数据源 '{name}'。")
        return None

    df = pd.read_csv(path)

    # 注入元数据列，方便合并后区分来源
    df['_dataset'] = name
    df['_label'] = info['label']
    df['_channel'] = info['channel']
    df['_modulation'] = info['modulation']
    df['_group'] = info['group']
    return df


def aggregate(df, groupby=None):
    """
    对 repeat 维度聚合，返回 (mean_df, std_df)。
    """
    if groupby is None:
        # 默认按 layer, snr, rate_ratio 聚合
        groupby = ['layer', 'snr', 'rate_ratio']

    # 保留元数据列
    meta_cols = ['_dataset', '_label', '_channel', '_modulation', '_group']
    meta_cols_present = [c for c in meta_cols if c in df.columns]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_agg = [c for c in groupby + numeric_cols if c in df.columns]

    # 去重：groupby 列可能同时出现在 numeric_cols 中
    all_cols = list(dict.fromkeys(cols_to_agg + meta_cols_present))

    grouped = df[all_cols].groupby(groupby, dropna=False)

    agg_mean = grouped.mean(numeric_only=True).reset_index()
    agg_std = grouped.std(numeric_only=True).reset_index()

    # 恢复元数据（groupby 后丢失非分组非数值列）
    for meta_col in meta_cols_present:
        first_vals = df.groupby(groupby, dropna=False)[meta_col].first().reset_index(drop=True)
        if meta_col not in agg_mean.columns:
            agg_mean[meta_col] = first_vals
        if meta_col not in agg_std.columns:
            agg_std[meta_col] = first_vals

    return agg_mean, agg_std


def load_and_aggregate(name, groupby=None):
    """加载 + 聚合一步完成。"""
    df = load_dataset(name)
    if df is None:
        return None, None
    return aggregate(df, groupby)


def filter_layer(mean_df, std_df, layer):
    """从一个聚合结果中筛选指定层。"""
    m = mean_df[mean_df['layer'] == layer].copy()
    s = std_df[std_df['layer'] == layer].copy() if std_df is not None else None
    return m, s


def filter_rate(mean_df, std_df, rate_ratio):
    """从聚合结果中筛选指定速率比。"""
    m = mean_df[mean_df['rate_ratio'] == rate_ratio].copy()
    s = std_df[std_df['rate_ratio'] == rate_ratio].copy() if std_df is not None else None
    return m, s


def merge_ra_datasets():
    """
    将 ra_trained_sa2_awgn 和 ra_trained_sa1_sa3_awgn 合并为一个
    包含所有三层 RA 训练数据的 DataFrame，然后聚合。
    返回 (mean_df, std_df) 或 (None, None)。
    """
    dfs = []
    for name in ['ra_trained_sa2_awgn', 'ra_trained_sa1_sa3_awgn']:
        df = load_dataset(name)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None, None

    merged = pd.concat(dfs, ignore_index=True)
    return aggregate(merged)


def compute_per_sample_kb(row):
    """
    根据 rate 和层的 n_points 计算每样本有效带宽 (KB)。
    统一使用: KB_per_sample = rate × n_points × 4 / 1024
    这比 CSV 中的 bandwidth 列更准确（后者在 RA 实验中未反映通道裁减）。
    """
    layer = row['layer']
    n_pts = LAYER_INFO[layer]['n_points']
    rate = row['rate']

    # 处理 rate 可能是 float 的情况（如 rate_ratio=0.2 时 0.2*320=64.0）
    if isinstance(rate, (int, float, np.floating)):
        actual_channels = int(rate)
    else:
        actual_channels = int(float(rate))

    return actual_channels * n_pts * 4 / 1024


def list_available_layers(dataset_name):
    """返回某个数据源包含的层级列表。"""
    info = DATASETS.get(dataset_name, {})
    if 'layers_available' in info:
        return info['layers_available']

    # 否则从 CSV 中读取
    df = load_dataset(dataset_name)
    if df is None:
        return []
    return sorted(df['layer'].unique().tolist())


def list_available_metrics(dataset_name):
    """返回某个数据源包含的指标列。"""
    info = DATASETS.get(dataset_name, {})
    if 'metrics_available' in info:
        return info['metrics_available']

    df = load_dataset(dataset_name)
    if df is None:
        return []
    standard_metrics = ['mse', 'psnr', 'cos_sim', 'ssim']
    return [m for m in standard_metrics if m in df.columns]


# ============================================================
# 启动时自检
# ============================================================
def _self_check():
    """启动时检查所有注册的数据文件是否存在。"""
    missing = []
    for name, info in DATASETS.items():
        path = RESULTS_DIR / info['file']
        if not path.exists():
            missing.append(f"  [{name}] {info['file']}")

    if missing:
        warnings.warn(
            "Missing data files (some figures will be skipped):\n" + "\n".join(missing)
        )

    available = []
    for name, info in DATASETS.items():
        path = RESULTS_DIR / info['file']
        if path.exists():
            df = pd.read_csv(path)
            # 分类精度 CSV 用 'method' 而非 'layer' 区分
            if 'layer' in df.columns:
                layers = sorted(df['layer'].unique())
            elif 'method' in df.columns:
                layers = sorted(df['method'].unique())
            else:
                layers = ['(unknown)']
            snrs = sorted(df['snr'].unique()) if 'snr' in df.columns else []
            rates = sorted(df['rate_ratio'].unique()) if 'rate_ratio' in df.columns else []
            available.append(
                f"  [{name}] {info['file']}: "
                f"layers={layers}, SNRs={snrs}, rate_ratios={rates}, "
                f"rows={len(df)}"
            )

    print(f"Data loader ready. {len(available)}/{len(DATASETS)} datasets available:")
    for a in available:
        print(a)

def load_baselines_classification():
    """
    从 baselines_full_comparison.csv 中加载全部 6 种方法的分类精度数据。
    CSV 列: method, snr, rate_ratio, mse, acc, repeat
    返回 (mean_df, std_df) 按 (method, snr, rate_ratio) 聚合。
    """
    path = RESULTS_DIR / 'baselines_full_comparison.csv'
    if not path.exists():
        warnings.warn(f"分类精度文件不存在: {path}")
        return None, None
    df = pd.read_csv(path)
    # 聚合
    grouped = df.groupby(['method', 'snr', 'rate_ratio'], dropna=False)
    agg_m = grouped['acc'].mean().reset_index()
    agg_s = grouped['acc'].std().reset_index()
    agg_m.rename(columns={'acc': 'accuracy'}, inplace=True)
    agg_s.rename(columns={'acc': 'accuracy'}, inplace=True)
    return agg_m, agg_s


METHOD_COLORS = {
    'NoAdapt':        '#555555',
    'Quant(8bit)':    '#CC79A7',
    'Plain-JSCC':     '#D55E00',
    'SA-only':        '#2B7BBD',
    'SA+RA+MSE':      '#009E73',
    'SA+RA+CE':       '#882255',
}
METHOD_MARKERS = {
    'NoAdapt':        's',
    'Quant(8bit)':    'x',
    'Plain-JSCC':     'P',
    'SA-only':        'o',
    'SA+RA+MSE':      'D',
    'SA+RA+CE':       '*',
}
METHOD_LINESTYLES = {
    'NoAdapt':        '--',
    'Quant(8bit)':    ':',
    'Plain-JSCC':     '--',
    'SA-only':        '-',
    'SA+RA+MSE':      '-',
    'SA+RA+CE':       '-.',
}
METHOD_ORDER = ['NoAdapt', 'Quant(8bit)', 'Plain-JSCC', 'SA-only', 'SA+RA+MSE', 'SA+RA+CE']


def load_baselines_feature_metric(metric='mse'):
    """
    从 baselines_full_comparison.csv 中加载全部方法的特征级指标。
    返回 list of (method, mean_df, std_df)，和 `aggregate` 格式一致。
    mean_df 有列: method, snr, rate_ratio, {metric}
    """
    path = RESULTS_DIR / 'baselines_full_comparison.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    grouped = df.groupby(['method', 'snr', 'rate_ratio'], dropna=False)
    agg_m = grouped[metric].mean().reset_index()
    agg_s = grouped[metric].std().reset_index()
    return agg_m, agg_s


# 模块导入时自检
_self_check()
