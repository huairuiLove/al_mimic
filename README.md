# CoMAL on MIMIC-III

本目录在 MIMIC-III 1.4 上复现 CoMAL 的多标签主动学习流程。`CoMAL-main/` 是原作者代码，
适配工作全部位于 `mimic_comal/`、`configs/` 和 `scripts/`；实验运行时还会将原始 Python 文件的
SHA-256 写入 `source_integrity.json`，便于确认原代码没有被改动。

可随时独立校验：`python scripts/verify_original_comal.py`。

## 实验协议

- 样本单位：一次住院 `HADM_ID`。
- 输入：`NOTEEVENTS` 中的 `Discharge summary`，多个 report/addendum 按住院聚合。
- 标签：`DIAGNOSES_ICD` 的三位 ICD-9 组，默认只保留训练集内频率最高的 50 类。
- 划分：按 `SUBJECT_ID` 确定性地做 80/10/10 划分，同一患者的不同住院不会跨 split。
- 默认特征：训练集拟合 TF-IDF + TruncatedSVD，验证和测试只做 transform，避免词表泄漏。
- A800 特征：使用仓库内 `CoMAL-main/bert/bert-base-uncased`，BF16 仅用于一次性冻结编码。
- 分类器：`LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear(labels)`，即三个 Linear。
- CoMAL：标签级隐表示、每个阳性标签原型、共享背景负原型、监督对比损失和特征/标签重构。
- 查询：默认严格复现原发布代码中的“正原型证据倒数 × 标签 cardinality 偏差”几何均值；
  `acquisition.formula=weighted` 可切换为不确定性/新颖度诊断变体。
- 评价：micro/macro AUPRC、micro/macro F1、AUROC、P@1/3/5 和 N@1/3/5 proxy。

标签词表严格只用 train split 统计。当前目标是 admission-level diagnosis-group 预测，并不是逐个
ICD-9 完整编码，也不应把结果直接与使用不同标签空间的工作横向比较。

TF-IDF 与 BERT 分别缓存到 `features_tfidf/` 和 `features_bert/`；加载时会校验 encoder 元数据，
不会在两组实验之间静默复用错误特征。

## 安装与检查

```bash
python -m pip install -e '.[dev]'
python main.py hardware --config configs/mimic_comal.yaml
python main.py prepare --config configs/mimic_comal.yaml
python main.py validate-data --config configs/mimic_comal.yaml
python main.py explore --config configs/mimic_comal.yaml
python main.py features --config configs/mimic_comal.yaml
pytest
```

`prepare` 会自动识别 `.csv.gz` 或已解压的 `.csv`，并顺序扫描 `NOTEEVENTS`，但只运行一次。生成的
`prepared/mimic_iii/records.jsonl` 含临床文本，因此默认被 `.gitignore` 排除；诊断和图表不会输出
原始文本。

先跑小规模端到端检查：

```bash
python main.py all --config configs/mimic_smoke.yaml
```

## 正式实验

完整 CoMAL 与 random 配对实验：

```bash
./scripts/run_main_experiments.sh all
```

也可分阶段运行，以便复用数据和 feature cache：

```bash
./scripts/run_main_experiments.sh prepare
./scripts/run_main_experiments.sh features
./scripts/run_main_experiments.sh active
```

固定 seed、split、初始标注集和查询预算后，脚本会输出 CoMAL/random 学习曲线对比。每个实验位于
`experiments/<name>/`：

```text
active_state.json             逐轮查询、指标、loss、时间和采样统计
final_metrics.json            最终 validation/test 指标与标注成本
final_predictions.npz         无文本的标签/概率数组
diagnostics/round_*.json      原型、校准、稀有标签和采样可靠性诊断
figures/*.png                 学习曲线、loss、时间和逐标签 AUPRC
checkpoints/final.pt          最终分类器与 CoMAL 权重
source_integrity.json         原始 CoMAL 文件哈希
```

跨轮冷启动/继承对比使用相同 feature cache、seed 和预算：

```bash
./scripts/run_cross_round_comparison.sh
```

继承组只继承分类器/CoMAL 权重，不继承 optimizer；round 0 使用 `20/10` epoch，之后使用
`3/2` epoch。逐轮实际初始化方式和 epoch 会写入 `active_state.json` 各 round record 的
`training_plan`。

数据探索额外生成标签 prevalence、cardinality、共现矩阵和 CoMAL 显存规模图。

## A800（当前机器：80GB + 18 核 CPU 配额）

本机 `cpu.max` 配额为 18 核；虽然拓扑能看到 144 逻辑 CPU，但超过配额只会调度抖动。

```bash
chmod +x scripts/*.sh
./scripts/run_a800.sh all
# 等价于 ./scripts/run_a800_144c.sh all
```

`configs/mimic_a800_144c.yaml` 的关键取舍：

- 本地 BERT 冻结编码使用 BF16、batch 384，CPU tokenize 与 GPU encode 流水线重叠；结果缓存为 FP16。
- 分类器和 CoMAL 训练使用 FP32 + TF32，并把全部缓存特征常驻 GPU（`gpu_resident_features`）。
- 分类 batch 16384 / 评估 batch 32768；CoMAL batch 96，`anchor_chunk_size=2048`。
- OMP/MKL/OpenBLAS/PyTorch 线程全部钉在 18，不使用会偷配额的 DataLoader worker。
- 禁止 `CUDA_DEVICE_MAX_CONNECTIONS=1`，避免切断 H2D 与计算重叠。

启动后用下面的命令检查实际吞吐：

```bash
python scripts/summarize_timing.py experiments/mimic_iii_comal_a800
nvidia-smi dmon -s pucm
```

如果 BERT 阶段 GPU 利用率低而 CPU 满载，优先降低 `features.batch_size`；如果 GPU 仍有余量，再按 32
的步长提高 BERT batch。不要先盲目提高 `training.comal_batch_size`。

## 复现边界

原实现绑定 AAPD/RCV1/Just Dance 的预处理格式、旧版 Transformers/Apex 和固定数据参数，不能直接
读 MIMIC-III。这里复现的是 CoMAL 的模型结构、两阶段训练、共享负原型与主动查询思想，并通过独立
适配器接入 MIMIC。默认 TF-IDF 配置用于可离线、快速、可测试的基线；论文级主结果应使用
`mimic_a800_144c.yaml` 的本地 BERT 配置，并同时报告 random 基线、数据版本、标签空间和硬件。
