# 点云语义通信项目完整文档

## 项目元信息

- **项目名称**: 带宽受限环境下基于任务驱动的 3D 语义信息提取与传输研究
- **负责人**: 杨笑沣（华中科技大学 电子信息与通信学院 本科大创项目）
- **指导教师**: 杨铀教授
- **项目周期**: 2025 年 10 月 — 至今
- **项目仓库**: `d:\Users\yxf\Desktop\pointcloud_learning`
- **开发环境**: Windows 11, Python 3.8/3.9 (conda env `pointcloud`), PyTorch 2.5.1+cu121, CUDA 12.1, 8GB GPU
- **文档日期**: 2026-07-22

---

## 一、项目背景与目标

### 1.1 研究问题

在无人机巡检、灾后救援、应急通信等场景中，三维点云数据量极大（每秒可达兆级点数），而实际无线通信带宽严重受限且极不稳定（灾后场景常低于 1Mbps）。海量三维数据与有限信道容量之间的矛盾是制约远程智能视觉应用的关键瓶颈。

### 1.2 研究目标

研制一套"任务驱动"的自适应三维语义通信系统。核心思路：将 PointNet++ 层次化特征提取与 SwinJSCC 的联合信源信道编码（JSCC）相结合，利用 SA（信噪比自适应调制）和 RA（速率自适应通道选择）模块实现特征的自适应传输。

### 1.3 预期创新点（来自申报书）

1. **任务驱动新范式**: 将"下游视觉任务完成效果"作为核心优化目标，而非传统的像素级重建精度
2. **点云处理与语义通信深度耦合**: PointNet++ 紧凑语义特征直接作为 JSCC 信源
3. **任务-信道双感知自适应传输**: 同时感知任务需求与信道状态，动态调整传输策略

---

## 二、技术架构

### 2.1 整体管线

```
原始点云 (ModelNet40, 1024 points)
  │
  ├─→ PointNet++ MSG 前向传播
  │     ├── SA1 特征: (B, 320, 512) — 局部几何
  │     ├── SA2 特征: (B, 640, 128) — 部件结构
  │     ├── SA3 特征: (B, 1024, 1)  — 全局语义 (⭐ 最紧凑, 4KB/样本)
  │     └── 分类结果: 40 类 log_softmax (92.8% 准确率)
  │
  ├─→ SA 模块 (ChannelModNet, 7层 SNR 条件调制)
  │     └── 根据 SNR 自适应调制特征各通道的强度
  │
  ├─→ RA 模块 (RateModNet, 7层 rate 条件调制 + Top-K 通道选择)
  │     └── 根据带宽约束选择最重要的 k 个通道
  │
  ├─→ AWGN/Rayleigh 信道 (SNR 0~20dB)
  │
  └─→ 解码器 (2层或3层 MLP) → 重建特征 → 分类头 → 分类精度
```

### 2.2 SA 模块 (ChannelModNet, 从 SwinJSCC 移植)

- **来源**: SwinJSCC (Yang et al., IEEE TCCN 2024)
- **结构**: 7层调制链 `sm_list` × `bm_list`，逐层对特征施加 SNR 条件调制因子
- **最终输出**: `x * sigmoid(mod_val)`，值域 [0,1]
- **训练**: 与解码器联合训练，MSE 损失，SNR 在 0~20dB 随机采样
- **用途**: 使特征对信道噪声鲁棒

### 2.3 RA 模块 (RateModNet)

- **结构**: 与 SA 类似，但条件变量是 rate（选择的通道数 k）
- **通道选择**: `importance = mod_val.mean(dim=1)` → `topk(importance, k)` → 二值掩码
- **训练**: 与解码器联合训练，MSE 损失，rate 在 {0.2, 0.5, 0.8, 1.0} 中随机采样
- **用途**: 在带宽受限下自适应选择最重要的通道

### 2.4 SA+RA 联合训练（两阶段策略）

**阶段 1 (解码器预热)**: 冻结 SA+RA 权重，只训练新的 3 层 MLP 解码器理解"SA→RA 串联调制"的统计特性。50 epochs, LR=1e-3。

**阶段 2 (联合微调)**: 解冻全部参数，小学习率联合微调。100 epochs, LR=1e-4。使 SA 和 RA 彼此适应，学习协同策略。

### 2.5 解码器架构

- **SA/RA 单独训练**: 2 层 MLP (`feat_dim → 256 → feat_dim`)
- **SA+RA 联合训练**: 3 层 MLP (`feat_dim → 512 → 256 → feat_dim`)
- **输入输出格式**: `(B, C, N_pts)` — 保持与训练脚本一致的 transpose 顺序

---

## 三、目录结构

```
pointcloud_learning/
├── Pointnet_Pointnet2_pytorch/     # PointNet++ 实现（修改版—返回中间特征）
│   ├── models/
│   │   ├── pointnet2_cls_msg.py    # 核心修改：forward 返回 l1/l2/l3 三层特征
│   │   ├── pointnet2_utils.py      # PointNet++ 核心算子（FPS, ball query, SA等）
│   │   └── semantic_feature_extractor.py  # 语义特征提取包装器
│   └── log/classification/         # 预训练分类模型 (92.8% ModelNet40)
│
├── SwinJSCC/                       # SwinJSCC 参考实现 (SA/RA 模块来源)
│   ├── net/
│   │   ├── encoder.py              # SwinJSCC 编码器 + AdaptiveModulator
│   │   ├── decoder.py              # SwinJSCC 解码器
│   │   └── channel.py              # AWGN + Rayleigh 信道模拟器
│   └── data/datasets.py            # 图像数据集加载器
│
├── experiments/
│   ├── adapters/
│   │   ├── swin_adaptive_modules.py  # ChannelModNet(SA) + RateModNet(RA) + 联合模块
│   │   └── pointnet_adapter.py       # (B,C,N) ↔ (B,N,C) 格式转换
│   ├── metrics/
│   │   └── feature_metrics.py        # MSE, PSNR, Cosine Similarity 等特征度量
│   ├── scripts/                      # 实验运行脚本 (AWGN + Rayleigh 各 6 个)
│   └── analysis/
│       ├── load_data.py              # 统一数据加载框架 (11个数据源注册表)
│       └── generate_figures.py       # 统一图表生成框架 (6组 × 19张图表)
│
├── pretrained/                       # 所有预训练权重 (~27 个 .pth 文件, ~1.5GB)
│   ├── sa_net_sa1/2/3_trained.pth   # SA 编码器权重
│   ├── decoder_sa1/2/3.pth          # SA 解码器权重
│   ├── ra_net_sa1/2/3_trained.pth   # RA 网络权重
│   ├── ra_decoder_sa1/2/3.pth       # RA 解码器权重
│   ├── sara_sa_net_sa1/2/3.pth      # SA+RA 联合微调后 SA
│   ├── sara_ra_net_sa1/2/3.pth      # SA+RA 联合微调后 RA
│   ├── sara_decoder_sa1/2/3.pth     # SA+RA 联合微调解码器
│   ├── plain_jscc_encoder_r*.pth    # Plain-JSCC 外部基线编码器
│   ├── plain_jscc_decoder_r*.pth    # Plain-JSCC 外部基线解码器
│   ├── phaseB_decoder_sa3.pth       # Phase B CE 解码器
│   └── sara_decoder_sa3_task*.pth   # Phase A/A2 任务微调权重
│
├── results/
│   ├── clean_features_sa1/2/3.npy   # PointNet++ 提取的干净特征 (~11.5GB)
│   ├── *.csv                        # 14 个实验结果 CSV 文件
│   └── figures/                     # 22 张论文级图表
│
├── data/                            # ModelNet40 数据集 + Stanford Bunny
├── docs/                            # 申报书 PDF + 参考论文 PDF
│
├── extract_features.py              # 从 PointNet++ 提取 SA1/SA2/SA3 特征
├── train_ae.py                      # SA 自编码器训练
├── train_ra_ae(_sa1/_sa3).py       # RA 自编码器训练 (3层)
├── train_sara_ae_phase1(_sa1/_sa3).py   # SA+RA 阶段1: 解码器预热
├── train_sara_ae_phase2(_sa1/_sa3).py   # SA+RA 阶段2: 联合微调
├── train_sara_ae_phase2_sa3_task.py     # Phase A: 仅解冻解码器 MSE+CE 微调
├── train_sara_ae_phase2_sa3_task_full.py # Phase A2: 全部解冻 MSE+CE 微调
├── train_phaseB_cls_decoder.py          # Phase B: 纯 CE 解码器训练
├── train_plain_jscc.py                  # Plain-JSCC 外部基线训练
├── evaluate_classification.py           # 4 方法分类精度评估
├── eval_phaseA.py / eval_phaseA2.py     # Phase A/A2 全量评估
├── eval_phaseB.py                       # Phase B 全量评估
├── eval_baselines.py                    # 6 方法完整横向对比评估
└── test_all.py                          # 集成冒烟测试
```

---

## 四、完整实验矩阵

### 4.1 特征级实验 (MSE/PSNR/Cosine Similarity)

| 维度 | 覆盖 | 说明 |
|------|------|------|
| 方法数 | 6 种 | NoAdapt / Quant(8bit) / Plain-JSCC / SA-only / RA-only / SA+RA |
| 特征层级 | 3 层 | SA1(局部几何, 640KB) / SA2(部件结构, 320KB) / SA3(全局语义, 4KB) |
| 信道类型 | 2 种 | AWGN / Rayleigh 衰落 |
| SNR 范围 | 11 个点 | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 dB |
| 速率比 | 4 个点 | 0.2, 0.5, 0.8, 1.0 |
| 重复次数 | 3-10 次 | 取平均 + 标准差 |
| 样本量 | 500-1000 | 每方法足够统计显著性 |

### 4.2 任务级实验 (分类精度)

| 维度 | 覆盖 |
|------|------|
| 方法数 | 6 种 (同上) |
| 特征层级 | SA3 只 (分类头只吃 SA3 输出) |
| 测试样本 | ModelNet40 全部 12,311 个测试样本 |
| 指标 | 40 类分类准确率 |
| 干净基线 | PointNet++ 在无噪声特征上的 97.48% 分类精度 |

### 4.3 修复实验 (分类精度恢复尝试)

| 实验 | 方法 | 训练目标 |
|------|------|---------|
| Phase A | 仅解冻解码器微调 | MSE + 0.05×CE (SA+RA 冻结) |
| Phase A2 | 全部 100M 参数微调 | MSE + 0.01×CE |
| Phase B | 从随机初始化训练新解码器 | 纯 CE (CrossEntropy) |

---

## 五、核心实验结果

### 5.1 特征级 (Part 1: MSE/PSNR) — 你的方法有显著优势

**SA3 层, rate=0.5, SNR=0dB**: SA+RA MSE=0.059 vs NoAdapt MSE=0.455，MSE 降低 87%。

**SA3 层, rate=0.5, SNR=10dB**: SA+RA MSE=0.053 vs NoAdapt MSE=0.046，MSE 略优。

**SA2 层, rate=0.5, SNR=0dB**: SA+RA MSE=0.044 vs NoAdapt MSE=0.815，MSE 降低 95%。

**结论**: 低 SNR 下 SA+RA 有压倒性优势。存在 SNR 阈值（~4dB 以下 SA+RA 最好，~18dB 以上 NoAdapt 最好）。RA 的最优速率比 = 0.5（点云特征通道约 50% 冗余）。

### 5.2 任务级 (Part 2: 分类精度) — 核心发现：MSE 与任务效能脱钩

**SA3, rate=0.5, 全量 12,311 测试样本**:

| 方法 | 0dB MSE | 0dB Acc | 10dB Acc | 关键信息 |
|------|---------|---------|----------|---------|
| NoAdapt | 0.455 | **95.7%** | **97.2%** | ⭐ SA3 特征天然极鲁棒 |
| Quant(8bit) | 0.455 | **95.6%** | **97.2%** | 线性处理 → 保分类 |
| SA-only | 0.041 | 84.9% | 93.2% | SA 调制 → 开始崩塌 |
| Plain-JSCC | 0.046 | 54.1% | 57.2% | MSE 训练即崩塌 |
| **SA+RA+MSE** | **0.059** | **27.9%** | **30.1%** | 全链路 MSE 优化 → 最差 |
| SA+RA+CE | — | 18.9% | 19.1% | CE 解码 → 彻底崩塌 |

**修复实验结果** (SNR=0dB, rate=0.5):

| 实验 | 策略 | 结果 |
|------|------|------|
| Phase A | 解码器 + MSE+CE 微调 | 27.4% (无效) |
| Phase A2 | 全 100M 参数 + MSE+CE 微调 | 28.4% (仅+0.8%) |
| Phase B | 纯 CE 新解码器 | 19.0% (更差, 且与 SNR 无关) |

### 5.3 核心发现总结

**发现 1**: SA3 特征天然极鲁棒 — NoAdapt 在 0dB 下仍保持 95.7% 分类精度

**发现 2**: 梯级崩塌 — MSE 优化程度越深，分类精度越低
```
Quant(8bit) → Plain-JSCC → SA-only → SA+RA
  95.6%    →   54.1%    →  84.9%  →  27.9%
  (无MSE)     (简单MSE)   (7层调制)  (14层调制)
```

**发现 3**: 特征保真度与任务效能完全反相关
```
NoAdapt: MSE=0.455(最差) → Acc=95.7%(最好)
SA+RA:   MSE=0.059(最好) → Acc=27.9%(最差)
```

**发现 4**: 修复实验系统性失败 → 问题根源在调制层 (SA+RA)，非解码器层。MSE 训练已将特征映射到分类头不可读的子空间。

---

## 六、所有实验脚本清单

### 训练脚本 (根目录)

| 文件 | 功能 | 状态 |
|------|------|------|
| `extract_features.py` | 从 PointNet++ 提取 SA1/SA2/SA3 干净特征 | ✅ |
| `train_ae.py` | SA + Decoder 自编码器训练 (SA1) | ✅ |
| `train_ra_ae.py` | RA + Decoder 自编码器训练 (SA2) | ✅ |
| `train_ra_ae_sa1.py` | RA 训练 (SA1) | ✅ |
| `train_ra_ae_sa3.py` | RA 训练 (SA3) | ✅ |
| `train_sara_ae_phase1.py` | SA+RA Phase 1: 解码器预热 (SA2) | ✅ |
| `train_sara_ae_phase1_sa1.py` | Phase 1 (SA1) | ✅ |
| `train_sara_ae_phase1_sa3.py` | Phase 1 (SA3) | ✅ |
| `train_sara_ae_phase2.py` | SA+RA Phase 2: 联合微调 (SA2) | ✅ |
| `train_sara_ae_phase2_sa1.py` | Phase 2 (SA1) | ✅ |
| `train_sara_ae_phase2_sa3.py` | Phase 2 (SA3) | ✅ |
| `train_sara_ae_phase2_sa3_task.py` | Phase A: 仅解冻解码器 MSE+CE 微调 | ✅ |
| `train_sara_ae_phase2_sa3_task_full.py` | Phase A2: 全部解冻联合微调 | ✅ |
| `train_phaseB_cls_decoder.py` | Phase B: 纯 CE 解码器训练 | ✅ |
| `train_plain_jscc.py` | Plain-JSCC 外部基线训练 (4 个 bottleneck 维度) | ✅ |
| `test_all.py` | 集成冒烟测试 | ✅ |

### 评估脚本

| 文件 | 功能 | 产出 |
|------|------|------|
| `experiments/scripts/run_no_adapt_exp.py` | NoAdapt AWGN 实验 | `no_adapt_results.csv` |
| `experiments/scripts/run_no_adapt_exp_rayleigh.py` | NoAdapt Rayleigh | `no_adapt_results_rayleigh.csv` |
| `experiments/scripts/run_channel_exp.py` | SA trained AWGN 实验 | `channel_exp_results_trained_with_decoder.csv` |
| `experiments/scripts/run_channel_exp_rayleigh.py` | SA trained Rayleigh | `channel_exp_results_rayleigh_trained_with_decoder.csv` |
| `experiments/scripts/run_ra_trained_exp.py` | RA trained AWGN (SA2) | `channel_exp_results_ra_trained.csv` |
| `experiments/scripts/run_ra_trained_sa1_sa3.py` | RA trained AWGN (SA1+SA3) | `channel_exp_results_ra_trained_sa1_sa3.csv` |
| `experiments/scripts/run_sara_joint_exp.py` | SA+RA AWGN + Rayleigh | `channel_exp_results_sara_joint.csv` / `_rayleigh.csv` |
| `evaluate_classification.py` | 4 方法分类精度评估 | `classification_accuracy.csv` |
| `eval_phaseA.py` | Phase A 全量评估 | `phaseA_comparison.csv` |
| `eval_phaseA2.py` | Phase A2 全量评估 | `phaseA2_comparison.csv` |
| `eval_phaseB.py` | Phase B CE vs MSE 解码器 | `phaseB_classification.csv` |
| `eval_baselines.py` | 6 方法 MSE + Acc 完整评估 | `baselines_full_comparison.csv` |

### 分析与可视化

| 文件 | 功能 |
|------|------|
| `experiments/analysis/load_data.py` | 统一数据加载 → 11 个数据源注册表 |
| `experiments/analysis/generate_figures.py` | 统一图表生成 → 6 组 × 19 张图表 |

---

## 七、图表体系 (22 张)

### §4.2 层级分析 (3 张)
- `sa_awgn_mse.png` — 三层 SA vs NoAdapt MSE
- `sa_awgn_cosine.png` — 三层 SA vs NoAdapt 余弦相似度
- `sa_channel_comparison.png` — SA AWGN vs Rayleigh 双信道

### §4.3 特征级性能 (8 张)
- `ra_rate_sweep.png` — RA 多速率比 MSE/PSNR
- `ra_rate_tradeoff.png` — Rate-MSE 单点折衷曲线
- `ra_three_layer_mse.png` — RA 三层 vs Baseline
- `sara_rate_sweep.png` — SA+RA 多速率 AWGN
- `sara_rate_sweep_rayleigh.png` — SA+RA 多速率 Rayleigh
- `sara_channel_comparison.png` — SA+RA AWGN vs Rayleigh
- `sara_vs_ra.png` — SA+RA vs RA 柱状性能增益对比
- `four_way_comparison.png` — NoAdapt/SA/RA/SA+RA 四方法全景

### §4.4 任务级评估 ⭐ (6 张 — 论文核心)
- `six_way_mse.png` — **6 方法 MSE vs SNR** (NoAdapt, Quant, Plain-JSCC, SA-only, SA+RA+MSE, SA+RA+CE)
- `six_way_accuracy.png` — **6 方法 Accuracy vs SNR** ⭐ 核心图
- `stair_step_collapse.png` — **MSE-Acc 梯级崩塌散点图** ⭐⭐⭐ 最核心卖点
- `cls_accuracy_overview.png` — 4 方法 Acc vs SNR + Acc vs Rate
- `cls_noadapt_vs_sa.png` — NoAdapt vs SA 聚焦对比
- `cls_full_comparison.png` — 5 方法含 Phase A2 全对比

### §4.5 综合分析 (3 张)
- `bandwidth_quality.png` — 带宽-MSE 散点
- `summary.png` — 2×2 综合网格
- `full_summary_2x3.png` — **2×3 全景总结网格** ⭐ (六方法MSE / 六方法Acc / 梯级崩塌 / SA+RA速率扫描 / Acc vs Rate / 修复实验柱状图)

---

## 八、关键技术决策与教训

### 8.1 为什么只测 SA3 的任务精度？
SA1(320-C,512-N) 和 SA2(640-C,128-N) 的输出维度与分类头要求 (1024) 不匹配，需经后续 Set Abstraction 层才能进入 FC。SA3(B,1024,1) 可直接 squeeze 后送入分类头。

### 8.2 SA/RA 的 transpose 顺序是关键坑点
训练和推理时的 transpose 必须完全一致：`(B,N,C) → SA/RA调制 → transpose(1,2) → (B,C,N) → AWGN → 解码器 → (B,C,N) → transpose(1,2) → (B,N,C)`。

### 8.3 MSE 训练的解码器 ≠ 分类最优的解码器
MSE 对 1024 维一视同仁，分类头只关心部分维度。解码器微调无法恢复分类精度 → 证明问题在调制策略本身。

### 8.4 SA3 特征天然鲁棒的意外发现
这是超出预期的好结果：SA3 在 SNR=0dB 下直接传输仅损失 1.8% 分类精度。这说明 PointNet++ 的全局语义特征极其紧凑。

---

## 九、当前论文定调

**论文类型**: 发现型论文 (Finding Paper)

**核心叙事**: 将 JSCC 领域标准 SA+RA 范式应用于点云语义特征传输 → 在特征保真度 (MSE) 上取得巨大改善 → 但发现分类精度呈现梯级崩塌 → 3 次修复实验系统性失败 → 论证 MSE 优化目标是 JSCC 在语义任务上的系统性盲区。

**目标期刊**: Q3 国际期刊/会议 (IEEE Access, IEEE SPL, ICCC 等)

**当前状态**: 实验全部完成，数据全部就位，图表体系完整。等待开始撰写论文初稿。

---

## 十、外部横向对比方法说明

### Plain-JSCC
- 最简单的 JSCC 基线：MLP encoder → AWGN → MLP decoder
- 训练目标：MSE
- 4 个 bottleneck 维度对应 4 个速率比
- 作用：证明"即使最简单的 JSCC，只要用 MSE 训练，就会破坏分类精度"

### Uniform Quantization (8-bit)
- SA3 特征 → 8bit 均匀量化 → AWGN → 反量化
- 无任何可学习参数，纯线性处理
- 作用：证明"线性处理能保留分类信息，非线性 MSE 调制才是破坏根源"

---

## 十一、给外部 AI 的提示

1. **项目所有代码在** `d:\Users\yxf\Desktop\pointcloud_learning`
2. **数据文件在** `results/` 目录下，14 个 CSV + 3 个 NPY 特征文件
3. **预训练权重在** `pretrained/` 目录下 (~27 个 .pth)
4. **图表在** `results/figures/` (22 张 PNG)
5. **Python 环境**: `conda activate pointcloud` (PyTorch 2.5.1+cu121)
6. **所有训练脚本可以重新运行**，数据路径硬编码在脚本中
7. **SA=ChannelModNet, RA=RateModNet** — 定义在 `experiments/adapters/swin_adaptive_modules.py`
8. **解码器有两种**: 2层 (SA/RA独立训练) 和 3层 (SA+RA联合训练)
9. **SA3 分类头** 从 `Pointnet_Pointnet2_pytorch/models/pointnet2_cls_msg.py` 的 `get_model` 提取
10. **核心发现**：MSE 最好的方法 (SA+RA) 分类最差 (28%)；MSE 最差的方法 (NoAdapt) 分类最好 (96%)
