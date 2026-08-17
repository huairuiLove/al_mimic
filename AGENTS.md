# AGENTS.md

面向在本仓库工作的 coding agent 的项目约束与当前状态。

## 运行环境

- 整个项目运行在单张 **NVIDIA A800 80G** GPU 上（Ampere，正式训练使用 FP32；硬件支持 FP16 / BF16 / TF32 张量核）。
- 代码在 macOS 本地开发，正式实验在 A800 服务器上执行；涉及数据路径的验证以服务器上的
  `dataset/` 目录为准（本地可能是断链的 symlink）。
- 正式实验要求 CUDA（`ActiveLearningExperiment` 在无 CUDA 时直接报错）。
- **所有环境安装配置必须在开启显卡之前做好**：驱动、CUDA 工具链、`uv sync` 依赖安装、
  数据准备与校验等，一律在显卡启用（上电/接入）之前完成；显卡开启后不再执行任何安装或配置操作。

## 当前任务优先级（2026-08）

1. **当前只做 MIMIC-III 任务**（`src/al_mimic/tasks/mimic_iii/`，三个 native task：
   `icd9_diagnoses`、`phenotyping_25`、`phenotyping_ccs_239`，以及 `modimix` 方法插件）。
2. 其余任务（`brset`、`mds_ed`）以及它们的适配、实验、文档更新，
   **等 MIMIC-III 全部完成后再进行**。在 MIMIC-III 完成前，不要主动改动
   `src/al_mimic/tasks/brset/`、`src/al_mimic/tasks/mds_ed/` 及其配置。

## 常用命令

```bash
uv sync --dev                  # 安装依赖
uv run al-mimic tasks          # 列出任务插件
uv run al-mimic validate-data --task mimic_iii --config configs/experiments/mimic_iii/comal.yaml
uv run al-mimic active --task mimic_iii --method comal --config configs/experiments/mimic_iii/comal.yaml
uv run pytest                  # 全部测试（tests/unit、tests/architecture、tests/integration）
uv run pytest tests/unit -x -q # 快速单元测试
```

实验一律从仓库根目录运行，使用 `configs/experiments/` 下的组合配置
（`extends` 链解析 `configs/tasks/` 与 `configs/methods/`）。

断点续训：`al-mimic active` 默认自动续训。每轮结束后把最终 ckpt
（`checkpoints/round_XXX.pt`）、该轮指标记录和循环进度（`checkpoints/progress.json`）
落盘；再次运行同一实验会从最后一个完成的轮继续（逐位还原，轮内不继承权重，
与正式协议一致）。`--no-resume` 强制从头重跑，并清掉该实验旧的 checkpoints。
进度文件与当前配置的 strategy / seed / 轮次计划不一致时续训会拒绝并报错。

## 改动约束

- 唯一支持的入口是 `al-mimic` CLI；不要新增根目录脚本或绕过 `src/al_mimic` 的执行路径。
- 架构边界见 `ARCHITECTURE.md`（`tasks` / `methods` / `core` / `utils`），
  有对应的 `tests/architecture/test_boundaries.py` 守护，改完必须跑。
- MIMIC-III 是正式（formal）对照实验：训练协议（轮数、step 预算、batch size、
  学习率、精度设置等）改动会影响与已完成实验的可比性，未经确认不要改
  `configs/tasks/mimic_iii/*.yaml` 里的协议字段。
- **禁止多 seed 实验**：当前方法间主指标差距不足 0.01，多 seed 不提供信息。
  在方法间差距拉开超过 0.02 之前，正式对比一律单 seed 运行。
- 数据校验严格模式：缺失或形状不符的 HDF5 产物会让运行失败，这是刻意的，
  不要用合成数据兜底。
- `experiments/`、`dataset/` 下的生成物不进 Git。

## 性能相关现状（A800 80G，FP32）

- MIMIC-III 正式训练配置统一使用 **FP32** 精度计算（`training.device: cuda`、
  `training.precision: fp32`）；`allow_tf32: false` 与 `cudnn_benchmark: false` 由
  `configure_runtime` 在每次运行时应用。
- FP32 是当前正式实验的既定计算精度，不要擅自切换到 FP16、BF16 或开启 TF32：
  精度和运行时设置属于训练协议字段，改动会影响与已完成实验的可比性。
- A800 仅作为正式实验硬件，macOS 本地只用于代码和单元测试；正式任务运行必须具备 CUDA。
