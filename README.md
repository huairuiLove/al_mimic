# CoMAL, MM-CoMAL and MoSAIC on MIMIC-III

本仓库在 MIMIC-III 1.4 上进行多标签主动学习。当前默认输入包含三种模态：

1. 临床文本：`NOTEEVENTS` 的 `Discharge summary`。
2. 生理测量时序：第一段 ICU stay 起始后 48 小时内的 `CHARTEVENTS`，每 2 小时分箱；包含心率、
   收缩压、舒张压、平均压、呼吸频率、血氧和体温，同时保留观测掩码。
3. 人口统计/住院信息：年龄、性别、入院类型和 ICU 可用性。

`DIAGNOSES_ICD` 的三位 ICD-9 组是监督标签，不计作输入模态。旧版本实际上只有出院小结文本作为
输入；结构化诊断码只用来生成标签。

## 论文与代码依据

采用 [Multimodal Pretraining of Medical Time Series and Notes](https://arxiv.org/abs/2312.06855)
（ML4H 2023）的“测量时序 Transformer + 文本编码器 + 多模态融合”设计。论文的
[官方代码](https://github.com/kingrc15/multimodal-clinical-pretraining) 已下载到
`third_party/multimodal-clinical-pretraining/`，固定来源提交记录在该目录的 `UPSTREAM.md`。

这里不是论文结果的逐项复现：论文研究 IHM/Phenotyping 和多模态预训练，本仓库研究 ICD-9 多标签
主动学习。应用的是其编码/融合架构，保留当前 CoMAL 标签原型和查询流程。

## 三种可复现方案

三臂使用同一份 `prepared/mimic_iii`、同一特征缓存、数据划分、随机种子、初始标注集和预算：

| 配置 | 方法 | 采集信号 |
|---|---|---|
| `configs/mimic_comal.yaml` | 纯 CoMAL | 发布代码的正原型证据乘基数失配 |
| `configs/mimic_mm_comal.yaml` | MM-CoMAL | 四视图标签原型、逐模态可靠度与证据离散度 |
| `configs/mimic_mosaic.yaml` | MoSAIC | Fisher c-最优价值的 8 联盟 Möbius 协同分解 |

三臂限定在完备模态 ICU 队列：首个 ICU stay 的 48 小时窗口至少有 6 个有效测量分箱。
`prepare` 会把门控前后的记录数、患者数、标签频次和平均标签基数写入 `audit.json` 与
`manifest.json`。这是队列选择条件，不是输入特征。

## 从头训练保证

- 不调用 `from_pretrained`，不加载模型 checkpoint，不接受预训练权重路径。
- 文本词表统计和 SVD 只在当前训练 split 上拟合，不使用外部语料或模型。
- 时序 Transformer、三个模态分支、融合 Transformer、分类头和 CoMAL 均随机初始化。
- `model.initialization` 必须是 `random`；配置中出现有效的 pretrained/checkpoint 输入会直接报错。
- 每个冷启动主动学习轮次重新随机初始化；只有显式使用 `inherit_across_rounds: true` 的对照实验会
  继承上一轮由本数据从头训练得到的权重。

第三方论文源码仅用于研究溯源，运行包不会 import 它。它包含作者提供的可选 ClinicalBERT 路径，
但本仓库的默认配置和训练实现均不会调用该路径，也未下载任何论文 checkpoint。

## 数据准备

从 [PhysioNet MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) 解压以下表到
`mimic-iii-clinical-database-1.4/`：

```text
ADMISSIONS.csv.gz
CHARTEVENTS.csv.gz
DIAGNOSES_ICD.csv.gz
ICUSTAYS.csv.gz
NOTEEVENTS.csv.gz
PATIENTS.csv.gz
```

数据按 `SUBJECT_ID` 做确定性的 80/10/10 划分，同一患者不会跨 train/validation/test。文本 SVD、
生理值均值/标准差以及年龄标准化参数全部只由 train split 估计。生成目录 `prepared/` 已被忽略，
不会提交含临床文本的数据。

## 运行

默认配置就是 A800 80GB 配置 `configs/mimic_a800_144c.yaml`，CLI 可省略 `--config`：

```bash
python -m pip install -e '.[dev]'
python main.py hardware
python main.py prepare
python main.py validate-data
python main.py features
python main.py active
python main.py visualize
```

完整 CoMAL/random 配对实验：

```bash
chmod +x scripts/*.sh
./scripts/run_a800.sh all
```

一次运行三种方案并生成 `experiments/three_method_comparison.png`：

```bash
chmod +x scripts/run_three_methods.sh
./scripts/run_three_methods.sh all
```

也可以只运行训练阶段；它会复用已经生成的共享数据和特征：

```bash
./scripts/run_three_methods.sh active
```

小规模 CPU 结构检查使用：

```bash
python main.py all --config configs/mimic_smoke.yaml
python main.py active --config configs/mimic_mm_comal_smoke.yaml
python main.py active --config configs/mimic_mosaic_smoke.yaml
pytest
```

MM-CoMAL 每轮额外记录逐视图可靠度/权重与证据离散度。MoSAIC 记录筛选分支证书、联盟高阶项均值、
线性化告警和实际 deflation 步数。正式 MoSAIC 配置精确枚举 3 个模态的全部 8 个联盟，每个联盟使用
4 个独立伙伴；默认按该轮精算分数整批选取。`mosaic.deflation_steps > 0` 会对相应数量的批内选择执行
Sherman-Morrison 更新并重算全格，适合小批量消融，但计算量会线性放大。

首次 `features` 必须顺序扫描体积很大的 `CHARTEVENTS`，之后会复用带源文件指纹的缓存。A800 配置
将数值缓存常驻显存，以 BF16 训练随机初始化的 Transformer，并使用 FP32/TF32 完成需要稳定性的
归约操作。

## 实验输出

每个实验位于 `experiments/<name>/`：

```text
active_state.json
final_metrics.json
final_predictions.npz
diagnostics/round_*.json
figures/*.png
checkpoints/final.pt
source_integrity.json
```

最终 checkpoint 会明确记录 `model_initialization: random`、`pretrained_weights: false` 和模态布局。
标签词表严格由 train split 统计；当前目标是 admission-level diagnosis-group 预测，不应与不同标签
空间或不同预测时间点的论文结果直接横向比较。
