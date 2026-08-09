# MoSAIC：以模态协同信息为采集目标的多模态主动学习范式

> **Mo**dality **S**ynergy **A**cquisition via **I**nterventional **C**ounterfactuals
>
> 定位：本文档是 `fisher_design_data_selection_agent.md`（RND-Agent）在多模态场景下的**继任者**，
> 不是它的重写。RND-Agent 的 c-最优实验设计目标（式 7）被完整保留，作为本文的**标量价值函数**；
> 本文新增的是一个此前不存在的层：**把该价值函数沿模态联盟格分解，并证明其高阶项恰好等于
> mixup 闭包无法合成的那部分设计价值**。这是本方案唯一的单点创新，其余部分是有意的继承。

---

## 0. 一句话主张

**在采用 mixup 的多模态训练中，样本价值中"可归因于单个模态"的部分可以被 mixup 闭包近似合成，
而"跨模态高阶绑定"的部分不能。因此标注预算应当优先购买后者。**

由此得到的可计算量 —— **模态信息增益格（Modality Information Gain lattice, MIG）** ——
就是本文要求的"多个模态的信息增益衡量算法"。

---

## 1. 为什么 CoMAL 在本仓库上不行：具体证据，而非泛泛而论

`experiments/mimic_iii_full_dryrun/final_metrics.json`（$C=50$，initial=500，query=100）：

| 指标 | test |
|---|---|
| `auroc_micro` | 0.601 |
| `auprc_micro` | 0.198 |
| `precision_micro` | 0.154 |
| `recall_micro` | 0.688 |
| `per_label_auprc` 尾部 | 0.054 ~ 0.09（约 15 个标签） |
| `per_label_auprc[46]` | **0.995**（近乎恒正标签，独自抬高 macro） |

三个结构性失效，逐条对应代码位置：

**(F1) 采集函数与模态无关。** `model.py:351` 的 `paper_comal_acquisition_scores` 计算

$$
\mathrm{score}=\sqrt{\Big(\textstyle\sum_{c:\,p_c\ge 0.5}\tfrac{\mathrm{sim}_c+1}{2}\Big)^{-1}}\cdot\sqrt{\big|\#\{\mathrm{sim}_c>\tau_c\}-\bar k\big|}
$$

两个因子都只依赖 `probabilities` 与融合后 latent 的 `prototype_similarities`。融合发生在
`model.py:150-151`（三 token 进 `fusion_encoder` 后 `mean(dim=1)`），此后模态身份被不可逆地抹掉。
**在多模态任务上使用一个模态不可见的采集函数，等于假设"哪个模态带来信息"对选样无影响**——
这个假设在本数据集上明显不成立（见 F3）。

**(F2) 原型几何在冷启动下无信号。** `configs/mimic_a800_144c.yaml` 设 `initialization: random`、
`inherit_across_rounds: false`，即每轮从零重训。`CoMALModule.prototypes` 由该轮弱编码器刷新，
早期轮次的 `own_prototype_similarity` 接近随机。更严重的是 `positive_similarity_thresholds`
（`model.py:325`）取**已标注正例相似度的 min/max 中点**：$C=50$、初始标注 3000 时，长尾标签在
$L$ 中往往只有个位数正例，min 与 max 由 1~2 个点决定，$\tau_c$ 实质是任意值。于是
`cardinality_mismatch` 计数的是一个阈值随机的指示函数。

**(F3) 模态信息稠密度是逐样本变化的，而 CoMAL 无从表达。** 即便在完备模态队列内（§7），
`measurement_observed_fraction` 在样本间差异很大：`build_structured_modalities` 对未观测分箱写
`normalized[~observed]=0.0`，稀疏样本的时序 token 接近常量。因此"该样本的增益究竟由文本还是
由时序驱动"逐样本不同。一个模态盲的打分函数无法表达这一点，只能整体性地偏向某一模态，
且偏向的方向不可控、不可诊断。

**结论：** 问题不在 CoMAL 的某个超参，而在于它的打分函数**定义域**里没有模态这个变量。
补一个模态加权项无法解决，因为"该给哪个模态多大权重"本身就是逐样本、逐轮次变化的未知量——
它必须被**测量**，不能被调参。这就是本文要解决的问题。

---

## 2. 设定与符号

模态集合 $\mathcal M=\{\mathrm{txt},\mathrm{ts},\mathrm{dem}\}$，$M=3$，对应
`features.py:110-124` 的 `clinical_note` / `icu_measurements` / `demographics`。

- 逐模态编码器 $e_m$，产生**模态 token** $t_m(x)\in\mathbb R^{h}$，$h=$ `model.fusion_dim` $=256$。
  对应 `model.py:141`（`text_encoder`）、`model.py:148`（`measurement_encoder[:,0]`）、
  `model.py:149`（`static_encoder`）。
- 融合映射 $\phi:(t_1,\dots,t_M)\mapsto z\in\mathbb R^h$，即 `model.py:151` 的
  `output_norm(fusion_encoder(tokens + modality_embedding).mean(dim=1))`。
- 增广特征 $\zeta(x)=[z(x),1]\in\mathbb R^{d}$，$d=h+1=257$（bias 增广，沿用 RND-Agent §2）。
- 头部 $W\in\mathbb R^{C\times d}$，$p_c=\sigma(\langle W_c,\zeta\rangle)$，$s_c^2=p_c(1-p_c)$。
- 队列门控（§7）保证所有样本三模态齐备；观测稠密度 $\mathrm{obs}(x)\in[0,1]$ 仅作诊断量，不进打分。
- $L$ 已标注集，$U$ 未标注池，$V$ 参考集（只用于设计方向与超参，test 全程封闭）。

**继承自 RND-Agent 的价值函数。** 逐标签分块 Fisher（原文式 14）

$$
A_c=\delta I_d+\sum_{x\in L}s_c(x)^2\,\zeta(x)\zeta(x)^\top,
\qquad
R_c=A_c^{-1}(g_V)_c,
$$

以及精确边际增益（原文式 14'）

$$
\Delta(\zeta)=\sum_{c=1}^{C}\frac{v_c^2}{1/s_c^2+m_c},
\qquad
v_c=\langle R_c,\zeta\rangle,\quad m_c=\zeta^\top A_c^{-1}\zeta .
\tag{1}
$$

**本文不修改式 (1)**。$\Delta$ 是一个把特征 $\zeta$ 映到标量的函数；下面所有内容都是关于
"$\zeta$ 是怎么由模态 token 组装出来的"，与 $\Delta$ 的具体形式正交（见 §12 的模块化说明）。

---

## 3. 干预算子：为什么必须是 mixup，而不是置零或 mask

要衡量"模态 $m$ 对该样本信息增益的贡献"，必须构造一个反事实：**如果模态 $m$ 不携带该样本特有的
信息，增益会变成多少**。反事实的构造方式决定了整个度量的合法性。

**（错误做法 A）置零：** $t_m\leftarrow 0$。`fusion_encoder` 从未在训练中见过零 token，
$\phi$ 在该点的输出是外推值。测得的差值混杂了"信息损失"与"分布外响应"，二者不可分离。

**（错误做法 B）掩码/丢弃 token：** 改变了 `fusion_encoder` 的序列长度与
`modality_embedding` 的配置，同样是模型未见过的输入构型。

**（本文做法）模态级 mixup 干预。** 从池中独立抽取伙伴 $x_j$，令

$$
\boxed{\;
t_m^{(i\to j,\lambda)}=(1-\lambda)\,t_m(x_i)+\lambda\,t_m(x_j),
\qquad \lambda\sim\mathrm{Beta}(\alpha,\alpha)\ \text{或}\ \lambda\equiv1
\;}
\tag{2}
$$

其余模态 token 保持 $x_i$ 的。$\lambda=1$ 是**完全置换**，$\lambda\to0$ 是**局部敏感度**。

**命题 1（在流形性）.** 对任意 $\lambda\in[0,1]$ 与任意 $j$，$t_m^{(i\to j,\lambda)}$ 落在
模态 $m$ 的 token 边缘分布的凸包内；当 $\lambda=1$ 时它就是一个真实样本的真实 token。
因此 $\phi$ 在该点上的求值不涉及分布外外推，**测得的差值只反映跨模态绑定的丧失**。

*说明.* 这是"为什么是 mixup"的**唯一**理由，也是充分理由：mixup 是把一个模态改造成
"边缘分布正确、但与其余模态的联合结构被破坏"的最简单算子。这恰好就是我们要做的干预。

**推论 1.1（干预在 token 空间而非输入空间）.** 干预必须施加在 $t_m$ 上而不是原始特征切片上。
原因：`measurement_encoder` 是一个 2 层 Transformer（`model.py:98`），在其输入端做 mixup 会
同时扰动"时序内部结构"与"跨模态绑定"，两者混淆。在 $t_m$ 上干预使被破坏的对象**恰好且仅仅**是
$\phi$ 所建模的跨模态关系。

---

## 4. 模态联盟值与 MIG 格

**定义（联盟值）.** 对 $S\subseteq\mathcal M$，令 $S$ 内模态取自 $x$ 本身，$S$ 外每个模态
**各自独立**抽取伙伴：

$$
v_x(S)\;=\;\mathbb E_{\{j_m\}_{m\notin S}\ \mathrm{i.i.d.},\,\{\lambda_m\}}
\Big[\;\Delta\big(\zeta\circ\phi\big(\{t_m(x)\}_{m\in S},\{t_m^{(x\to j_m,\lambda_m)}\}_{m\notin S}\big)\big)\Big].
\tag{3}
$$

两端点：$v_x(\mathcal M)=\Delta(\zeta(x))$ 是原始增益；$v_x(\varnothing)$ 是**全随机重组嵌合体**
的期望增益——注意它与 $x$ 无关，是一个池级常数 $v_\varnothing$。

> **关键设计选择：$S$ 外的伙伴必须相互独立。** 若所有被替换模态共用同一个伙伴 $j$，
> 则 $v_x(\varnothing)=\Delta(\zeta(x_j))$，测的是另一个真实样本，不是"无联合信息"的零点。
> 独立抽取给出的是**边缘分布的乘积** $\bigotimes_m\mu_m$ 的样本，这正是我们需要的零假设。

**定义（MIG，Möbius 系数）.** 对 $S\subseteq\mathcal M$，

$$
\boxed{\;
I_x(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}\,v_x(T)
\;}
\tag{4}
$$

$M=3$ 时全格只有 $2^3=8$ 个联盟，可**精确枚举**，无需 Shapley 采样。语义：

| 项 | 含义 |
|---|---|
| $I_x(\{m\})=v_x(\{m\})-v_\varnothing$ | 模态 $m$ 的**独有**贡献 |
| $I_x(\{m,m'\})$ | 模态对的**二阶交互**：$>0$ 协同（互补），$<0$ 冗余（可互相替代） |
| $I_x(\mathcal M)$ | 三模态的三阶交互 |

**定义（可加分量与协同分量）.**

$$
U^{\mathrm{add}}(x)=\sum_{m}I_x(\{m\}),
\qquad
\boxed{\;U^{\mathrm{syn}}(x)=\big[v_x(\mathcal M)-v_\varnothing\big]-U^{\mathrm{add}}(x)=\!\!\sum_{|S|\ge2}\!\!I_x(S)\;}
\tag{5}
$$

$U^{\mathrm{syn}}$ 就是该样本的设计价值中**任何单模态都无法解释**的部分。

**命题 2（协同为零刻画了融合的线性性）.** 若 $\phi$ 是仿射映射
$\phi(t_1,\dots,t_M)=\sum_m B_mt_m+b$，则 $z$ 在模态 token 上可加分离，
$\{$真实样本可达的 $z\}$ 与 $\{$跨样本重组可达的 $z\}$ **张成同一集合**（Minkowski 和），
从而 $v_x(S)$ 完全由各模态边缘决定，$I_x(S)=0\ \forall|S|\ge2$。

*证明.* 仿射性给出
$\phi(\{t_m(x)\}_{m\in S},\{t_m(x_{j_m})\}_{m\notin S})=\sum_{m\in S}B_mt_m(x)+\sum_{m\notin S}B_mt_m(x_{j_m})+b$，
第二项的分布不依赖 $x$ 也不依赖 $S$ 的内部构成，仅依赖 $\mathcal M\setminus S$ 这个集合。
代入式 (4) 的交错和，任何 $|S|\ge2$ 的项逐项相消。$\square$

**推论 2.1（$U^{\mathrm{syn}}\ne0$ 是可证伪的架构断言）.** 测得 $U^{\mathrm{syn}}\equiv0$ 意味着
`fusion_encoder` 在当前权重下退化为逐模态线性组合——这是一个**可直接检验的诊断**，
且它同时判定：此时本方法退化为 RND-Agent，不应宣称多模态贡献。**这条必须进日志。**

---

## 5. 核心命题：mixup 闭包能合成什么，不能合成什么

这是本文档全部主张的支点。

记真实联合分布 $\mu_{\mathrm{joint}}$ 为 $(t_1,\dots,t_M)$ 在真实数据上的律，
$\mu_\otimes=\bigotimes_m\mu_m$ 为其边缘乘积。训练中使用的**模态级 mixup 增广**
（式 (2)，伙伴独立抽取）所生成的样本，其 token 组合恰好服从（$\lambda$ 平滑后的）$\mu_\otimes$。

对一个 token 分布 $\nu$，定义它能供给的设计信息

$$
\mathcal F[\nu]=\mathbb E_{t\sim\nu}\big[s_c^2(\phi(t))\,\zeta\zeta^\top\big],\qquad \zeta=\zeta\circ\phi(t).
\tag{6}
$$

**命题 3（可加-协同分离定理）.** 由式 (3)(4) 的构造，对任意 $|S|\ge2$，$I_x(S)$ 的取值
**只依赖** $x$ 在 $\phi$ 下的联合构型相对于 $\mu_\otimes$ 重组构型的差异；
一切能由 $\mu_\otimes$ 支撑（即由已标注集经模态级 mixup 生成）的设计信息，
其对应的 Möbius 高阶项恒为零。等价地：

$$
\boxed{\;
U^{\mathrm{syn}}(x)\;=\;\big[\text{$x$ 的设计价值}\big]\;-\;\big[\text{仅凭 $\mu_\otimes$ 重组可复现的那部分设计价值}\big].
\;}
\tag{7}
$$

*证明要点.* $v_x(S)$ 对 $S$ 外模态取 $\mu_\otimes$ 边缘期望，故 $\{v_x(S)\}_{|S|<M}$ 张成的
是"至多 $M-1$ 个模态保持联合、其余按乘积边缘替换"的全部可复现价值；式 (4) 的交错和在
$|S|\ge2$ 上正是该子空间的**正交补上的读数**。$\square$

**这条命题不是启发式类比，而是构造性的恒等**：MIG 的高阶项之所以是"mixup 无法合成的价值"，
是因为 $v_x(S)$ 的定义本身就用 mixup 分布做替换。**这是本文选择这一分解（而非任意其他模态归因）
的唯一理由，也是它相对 Shapley/LIME/Ablation 类归因的实质区别**：后者的零点（置零、均值填充）
与训练中实际使用的增广分布无关，因而它们的分解结果与"标注买什么才划算"没有可推导的联系。

**必须明确不能由此推出的三件事：**

1. **不能推出 $U^{\mathrm{add}}$ 无价值。** 命题 3 只说它**可被 mixup 闭包供给**，
   前提是已标注集在该模态方向上已有覆盖。若某模态方向在 $L$ 中根本没有支撑，
   mixup 也造不出来——此时 $U^{\mathrm{add}}$ 同样必须购买。式 (8) 的 $\eta$ 项正是为此而设，
   **不得设为 0 而不做消融**。
2. **不能推出"选协同高的样本必然涨点"。** 命题 3 是关于**设计目标 $\Phi$** 的陈述；
   $\Phi$ 与真实泛化增益之间隔着 Laplace 近似与 head-local 假设（继承自 RND-Agent §11 的全部
   局限）。二者的相关性是 H2 的经验问题。
3. **不能推出本方法优于任何具体基线。** 命题 3 给出的是"该量测的是什么"，不是"该量测得更好"。

---

## 6. 采集准则

$$
\boxed{\;
\mathrm{Score}(x)\;=\;\big[U^{\mathrm{syn}}(x)\big]_+\;+\;\eta\cdot U^{\mathrm{add}}(x),
\qquad \eta\in[0,1)
\;}
\tag{8}
$$

- $[\cdot]_+$：负协同（= 模态冗余）不是"负价值"，只是"该样本的价值可由单模态解释"，
  截断到 0 后由 $\eta U^{\mathrm{add}}$ 项接管，不会把冗余样本推到排序末尾以下。
- $\eta$ 是**本方法唯一的新增权重超参**，语义明确：**你相信 mixup 闭包能替代多少可加信息**。
  $\eta=1$ 退化为模态盲的 $v_x(\mathcal M)-v_\varnothing$（≈ RND-Agent），
  $\eta=0$ 是纯协同。必须单独消融（§13）。

**这条式子就是本文要求的"衡量标准"**：它对每个候选样本给出各模态信息增益的完整分解
$\{I_x(S)\}_{S\subseteq\mathcal M}$，并据此排序。

---

## 7. 队列口径：完备模态队列（cohort gate）

**本实验只研究多模态数据。** $a_{\mathrm{ts}}(x)=0$ 的入院（无 ICU stay，其测量张量由
`multimodal.py` 的 `normalized[~observed]=0.0` 产出为全零）**在数据准备阶段即被剔除**，
不进入 train/validation/test 任何一侧，也不进入候选池。

**理由（两条，都是硬理由）：**

1. **干预不可定义。** 把别的病人的生命体征 mixup 进一个没有 ICU 记录的病人，是**捏造数据**，
   不是干预。式 (2) 的在流形性（命题 1）在该子群上不成立。
2. **协同不可定义。** $|\mathcal M_x|<M$ 时高阶 Möbius 项的支撑集不同，
   $U^{\mathrm{syn}}$ 跨子群不可比；强行分位数对齐是在比较两个不同的量。

**实现（`data.py` 的 `prepare_mimic`）：** 候选入院除现有的"有 note 且有 ICD 诊断"外，
追加第三个条件——**在 `ICUSTAYS` 中存在首个 ICU stay，且其 48h 窗内至少有
$n_{\min}$ 个有效测量分箱**。$n_{\min}$ 是队列定义超参（建议 $n_{\min}\ge1$，
即至少一个非空时间箱），必须写进 `manifest.json` 与 `audit.json`。

**必须报告的口径影响（诚实性要求，不可省）：**

- 剔除前/后的 records 数、subjects 数、各 split 规模；
- 剔除前/后的 `positive_counts` 与 `cardinality_mean`——**ICU 队列的疾病谱与全院队列不同**
  （ICU 病人更重、共病更多），这是一个**已知的选择偏倚**；
- 因此本实验的结论只对 **"有 ICU 记录的入院"** 这一总体成立，
  **不得**表述为"在 MIMIC-III 上"，须表述为"在 MIMIC-III 的完备模态 ICU 队列上"。

**后果（简化）：** 队列门控之后，$\mathcal M_x\equiv\mathcal M$ 对所有 $x$ 成立，于是
§4 的归因门控、分层排序、层间配额（原规则 A/B/C）**全部删除**，$\gamma$ 超参随之删除。
式 (8) 直接在全池上排序。这使方法显著简化，代价是适用范围收窄——这个取舍必须在论文中明写。

**保留的诊断.** 仍需报告池内 `measurement_observed_fraction` 的分布：即使都有 ICU stay，
观测稠密度差异很大，极稀疏样本的时序 token 接近常量，其 $I_x(S)$ 会系统性偏低。
若该效应显著（可由 observed_fraction 与 $U^{\mathrm{syn}}$ 的相关性检出），
应作为**已知局限**报告，而不是再加一层加权补偿。

---

## 8. 批内去冗余：继承，不新增

沿用 RND-Agent 式 (11)/(12) 的精确 deflation：选中 $x_b$ 后 Sherman–Morrison 更新
$A_c^{-1}$，重算 $R$。**不新增任何批内多样性项。**

理由：$\Delta$ 的饱和项 $m_c$ 已经在特征层面表达了冗余，而模态层面的冗余会**自动**传导——
定义模态回拉需求

$$
\rho_{m,c}(x)=J_m(x)^\top R_c\in\mathbb R^{h},
\qquad J_m=\partial z/\partial t_m ,
\tag{9}
$$

则 $R$ 一旦被 deflate，$\rho_{m,c}$ 随之收缩。**若某模态的需求已被本批次满足，
该模态方向上的后续候选自动降分**，无需外挂模态配额。

**猜想 8.1（模态轮转）.** 贪心过程会在模态之间自发轮转：本批次前几个样本若都由文本驱动，
$\rho_{\mathrm{txt}}$ 收缩，后续 argmax 转向时序驱动的样本。
**这是猜想，不是式 (9) 的推论**——可验证形式：记录每步选中样本的 $\arg\max_m I_x(\{m\})$
序列与 $\|\rho_{m,c}\|$ 轨迹，检验是否确实轮转。列为独立消融条目。

---

## 9. 复杂度与两段式实现

本仓库规模：$C=50$，$d=257$，$|U|\le49152$，$B=1000$，$M=3$。

**关键的架构红利：$\phi$ 只作用在 3 个 token 上。** `fusion_encoder` 是 2 层、3-token、$h=256$
的 Transformer，单次前向约 1.6 MFLOP。全格需 $2^3=8$ 个联盟 × $K$ 个伙伴样本的 $\phi$ 求值。
$K=8$ 时每候选 64 次 $\phi$——这在晚融合架构下是**可以承受的**，在早融合架构下则不可行
（需重跑整个编码器）。这一点应在论文中明确为方法的适用边界。

**两段式（沿用 RND-Agent §9.4 的纪律）：**

| 阶段 | 代价 | 手段 |
|---|---|---|
| 模态 token 缓存 $t_m(x)$ | $|U|$ 次编码器前向 | 本来就要做（predict 路径） |
| 全池粗筛 | $O(|U|Cd)$ | 式 (16) 的可容许上界 $\Delta^{\mathrm{ub}}$ + **跨模态注意力质量**（见下） |
| 工作集 $S$ 上的全格 | $|S|\cdot 8K$ 次 $\phi$ + $O(|S|\,8K\,Cd)$ | $|S|=5000$ 时约 $3\times10^5$ 次 3-token 前向 |
| deflation | $O(Cd^2)$/步 | Sherman–Morrison，继承 |

**筛选的诚实性问题（必须写进日志）。** $\Delta^{\mathrm{ub}}$ 对**总增益**是可容许上界，
对 $U^{\mathrm{syn}}$ **不是**。高协同但低总增益的样本可能被粗筛丢弃。缓解：工作集取
**并集**——top-$M$ by $\Delta^{\mathrm{ub}}$ $\cup$ top-$M'$ by 协同代理。协同代理取
`fusion_encoder` 第一层的**跨模态注意力质量** $\sum_{m\ne m'}\mathrm{attn}_{m\to m'}$，
它免费可得且与"融合是否真的在做跨模态绑定"直接同向。
**日志必须打印：最终选中样本中各来自哪个筛选分支、以及各自的配额截断量。**

---

## 10. 查询算法

```text
输入：L, U, V, 预算 B, 工作集大小 M_ws, 伙伴数 K, 权重 eta
输出：查询批次 Q

 0. 前置：U 已由 §7 的队列门控过滤，所有样本三模态齐备（M_x ≡ M）。
 1. 前向全池，缓存模态 token t_m(x) 与融合 zeta(x)。
    自检：max|Z W^T - logits| ~ 0（特征必须与 logits 同坐标系，继承 RND-Agent 易错点 1）。
 2. A_c <- delta*I + sum_{x in L} s_c^2 zeta zeta^T，并**叠加 mixup 闭包项**：
    额外采样 n_mix 组来自 L 的独立模态重组样本，把它们的 Fisher 计入 A_c。
    理由：训练实际消费的是 mixup 闭包，冗余应当相对闭包度量，而非相对 L 本身。
    消融项：n_mix = 0（不含闭包）。
 3. R_c <- A_c^{-1} (g_V)_c；施加稀有标签行下界（继承 RND-Agent 式 15，kappa 仍需消融，
    且 kappa>0 时目标变为 GV_eff，不得声称是原 c-最优目标）。
 4. 估计池级常数 v_empty：抽 n_0 个全随机嵌合体，取 Delta 均值。打印其标准误。
 5. 全池粗筛 -> 工作集 S = top-M_ws(Delta_ub) ∪ top-M'(跨模态注意力质量)。
 6. 对 S 中每个 x，枚举全部 2^M = 8 个联盟 S'：
    6.1 抽 K 组独立伙伴，按式 (2) 构造反事实 token，过 phi 得 zeta'，
        按式 (1) 算 Delta，平均得 v_x(S')。
    6.2 Möbius 反演式 (4) 得 {I_x(S')}；式 (5) 得 U_add, U_syn。
    6.3 Score(x) 按式 (8)。
    **诊断：若 max_{|S'|>=2} |I_x(S')| 的池均值 ~ 0，说明融合已退化为线性，
      本方法无多模态贡献（推论 2.1），必须在日志中显式告警。**
 7. 重复 B 次：
    7.1 取 argmax Score(x)（全池单一排序，无分层）；
    7.2 Sherman–Morrison deflation 更新 A_c^{-1}、R；
    7.3 重算受影响候选的 v_c、m_c（不重算 phi：token 未变，反事实 zeta 可缓存复用）；
    7.4 记录 Phi_b 单调性与预测/实际 Delta 一致性。
 8. 返回 Q。

关键实现注记：第 7.3 步的 zeta 缓存是本算法可行性的核心——反事实融合特征 zeta' 只依赖
模态 token 与 phi，与 A_c/R 无关，故**整轮只需计算一次**，deflation 循环中只做
O(Cd) 的重打分。若实现时每步重跑 phi，代价会放大 B=1000 倍。
```

---

## 11. 前提假设与验证方案

按重要性排序。**H0 是一票否决项，必须先做。**

- **H0（几何跨轮可迁移性）— 必须最先验证，且当前配置下很可能不成立。**
  本仓库 `inherit_across_rounds: false`，每轮从零重训。第 $t$ 轮模型的 Fisher 几何是否能预测
  第 $t{+}1$ 轮**从头训练**的收益？若不能，则**一切基于模型几何的 AL 方法在此配置下都失效**，
  包括 CoMAL、RND-Agent 和本方法。
  *检验：* 固定 $L$，用第 $t$ 轮几何给候选打分，随机抽若干子集实际加入并从头重训，
  测打分与参考集损失下降的 Spearman $\rho$。同时跑 `inherit_across_rounds: true` 对照。
  *若冷启动下 $\rho\approx0$ 而继承下 $\rho>0$：应将主实验切换为继承模式，并在论文中
  明确声明这是方法的前提，而不是掩盖它。*

- **H1（协同信号非平凡）.** $U^{\mathrm{syn}}$ 在池上的分布应显著偏离 0，且
  $\mathrm{Var}(U^{\mathrm{syn}})$ 与 $\mathrm{Var}(U^{\mathrm{add}})$ 同阶。
  *若 $U^{\mathrm{syn}}\approx0$：由推论 2.1，融合已线性化，方法无贡献——
  此时的正确反应是先修训练（见 H1'），而不是加权重。*

- **H1'（融合未坍缩的训练前提）.** 弱编码器下 `fusion_encoder` 极易坍缩到只用文本
  （文本 256 维 SVD 稠密，测量 336 维中一半是 mask 且 `observed_fraction` 偏低）。
  归因测的是**当前模型在用什么**，不是**什么本来可用**——这是所有模型依赖型 AL 的共同盲区。
  *缓解（训练侧前提，必须实施）：* 训练中加入**模态 dropout**，强制 $\phi$ 不能忽略任一模态。
  *诊断：* 报告"模型已利用的协同"与"单模态探针给出的可利用上界"之差。

- **H2（协同优先确实优于总增益优先）— 本文的核心卖点。**
  相同预算下，$\eta$ 小的配置应在 macro-AUPRC 与长尾标签 AUPRC 上不劣于 $\eta=1$。
  *这是命题 3 的经验推论，不是它的逻辑推论。命题 3 只保证测的是"mixup 造不出的价值"，
  不保证该价值兑换成泛化增益的效率更高。*

- **H3（队列门控的代价）.** 报告完备模态队列相对全院队列的规模损失与疾病谱漂移（§7）。
  这不是"验证方法有效"，而是**界定结论的适用总体**——必须做，且结果不影响方法取舍。
  附加检验：`measurement_observed_fraction` 与 $U^{\mathrm{syn}}$ 的相关性；
  若强相关，说明协同度量部分地只是在测"时序有多稠密"，须作为局限报告。

- **H4（mixup 闭包项的作用）.** $n_{\mathrm{mix}}>0$ 应降低所选批次与 $L$ 的 mixup 闭包的
  重合度。若无差异，说明闭包在 $d=257$ 下已被 $L$ 张满，该项可删。

---

## 12. 与已有工作的关系（按证明强度分级，沿用 RND-Agent §8 的纪律）

**A 档：代数恒等或构造性事实。**

| 已有工作 | 关系 |
|---|---|
| RND-Agent（式 14'） | 本文 $v_x(\mathcal M)$ 即其 $\Delta_b(x)$；$\eta=1$ 时本方法**恒等于**它 |
| Möbius / Harsanyi 分解 | 式 (4) 就是标准 Möbius 反演，$M=3$ 时精确枚举。**分解本身不是创新点** |
| 仿射融合 ⇒ 零协同 | 命题 2，逐项相消 |

**B 档：附加假设下成立的特例。**

| 已有工作 | 关系 | 所需假设 |
|---|---|---|
| CoMAL（本仓库） | $M=1$ 或 $\phi$ 仿射时，模态维度退化，本文归于单模态原型方法 | 融合线性化 |
| 模态 Shapley 值 | $\eta=1/M$ 且用均匀权重时，式 (8) 与模态 Shapley 排序一致 | 权重取 Shapley 核 |

**C 档：结构类比，不得宣称归约。**

| 已有工作 | 可准确陈述的 | 不能宣称的 |
|---|---|---|
| PID（Partial Information Decomposition） | $U^{\mathrm{syn}}$ 与 PID 的 synergy 项在**语义上**同向：都刻画"联合独有" | **不能**说二者相等。PID 定义在互信息上且一般不可计算；本文定义在设计目标 $\Delta$ 上且可精确枚举。这是两个不同的量 |
| BADGE / BAIT | 同属 Fisher 型采集 | 二者均无模态分解，不存在归约关系 |
| Mixup 正则化文献 | 共用同一算子 | 那里 mixup 是**训练增广**，这里是**归因干预**。相同算子、不同用途，不构成继承 |
| 模态重要性归因（Grad-CAM 类） | 同为逐模态归因 | 那些方法的零点是置零/均值填充，与训练分布无关；命题 3 的联系在它们上**不成立** |

**关于"多模态主动学习"这个方向本身。** 已有工作（模态不确定性加权、缺失模态补全、
逐模态委员会）**不是**本文的归约对象，也不构成本文的创新点。本文的差异点只有一个：
**归因的零点被选为训练实际使用的增广分布，从而使分解结果与"标注买什么划算"之间存在
可推导的联系（命题 3）**。这是唯一应当声明的新意。

---

## 13. 消融矩阵

| 消融 | 要回答的问题 | 对应命题/节 |
|---|---|---|
| $\eta=1$（模态盲，≡ RND-Agent） | 模态分解是否有净增益？**这是最重要的一条** | 式 (8)、H2 |
| $\eta=0$（纯协同） | 可加分量是否完全可弃？ | 命题 3 的注 1 |
| 干预算子：mixup vs **置零** vs 均值填充 | 在流形反事实是否必要？ | 命题 1 |
| 伙伴独立 vs 共用同一伙伴 | 零点构造是否正确？ | §4 的关键设计选择 |
| $\lambda\equiv1$ vs $\lambda\sim\mathrm{Beta}$ | 软干预是否降方差？ | 式 (2) |
| 伙伴数 $K\in\{1,4,8,16\}$ | 归因的蒙特卡洛噪声底在哪里？ | §9 |
| 全格 vs 仅一阶边际（Shapley 采样） | 高阶项是否携带独立信息？ | 式 (4) |
| 队列门控 $n_{\min}$ 阈值 | 队列定义对结论的敏感度？（不是效果消融，是口径敏感度） | §7、H3 |
| $n_{\mathrm{mix}}=0$ | mixup 闭包 Fisher 项是否有用？ | H4 |
| 训练侧关闭模态 dropout | 融合是否坍缩、$U^{\mathrm{syn}}$ 是否塌到 0？ | H1' |
| `inherit_across_rounds` true/false | 冷启动是否摧毁几何有效性？ | **H0** |
| 粗筛并集 vs 仅 $\Delta^{\mathrm{ub}}$ | 高协同低总增益样本是否被丢弃？ | §9 |
| 基线组 | random / CoMAL-paper / CoMAL-weighted / BADGE / CoreSet / BALD | — |

---

## 14. 仓库落点

**新增模块（建议 `mosaic/` 与 `mimic_comal/` 并列，不改动既有分支）：**

| 文件 | 职责 |
|---|---|
| `mosaic/tokens.py` | 从 `MultimodalFusionClassifier` 抽取 $t_m(x)$ 与 $\phi$ 的可复用调用；需在 `model.py` 的 `forward` 中额外返回 `modality_tokens`（当前 `model.py:150` 的 `tokens` 已就绪，只需 expose） |
| `mosaic/intervene.py` | 式 (2) 的 mixup 干预算子、伙伴采样、$\lambda$ 调度 |
| `mosaic/lattice.py` | 式 (3) 的 $v_x(S)$、式 (4) Möbius 反演、式 (5) 分解 |
| `mosaic/design.py` | 继承式 (1)：$A_c$（含 mixup 闭包项）、$R_c$、Sherman–Morrison deflation、稀有标签下界 |
| `mosaic/acquire.py` | 式 (8) 打分、§9 两段式筛选与证书日志 |
| `mosaic/_check_*.py` | 命题 2 的数值自检（仿射融合下 $I(S)\equiv0$）、Möbius 反演一致性、$\Phi$ 单调性 |

**改动点（最小化）：**

- `mimic_comal/model.py:139-152`：`forward` 增加 `return_tokens` 参数，返回融合前的
  `tokens`（3 个模态 token）。**不改变默认返回值**，保证既有基线逐位可复现。
- `mimic_comal/model.py`：新增 `fuse_from_tokens(tokens)`，供反事实复用 $\phi$，
  避免重跑逐模态编码器。
- `mimic_comal/runner.py:321-411`：`strategy == "mosaic"` 走新分支，与既有
  `comal` / `random` 并列。**不改动既有分支**。
- `mimic_comal/training.py`：新增模态 dropout（H1' 的前提），受配置开关控制，默认关闭以保基线。
- `mimic_comal/data.py`：`prepare_mimic` 的候选筛选追加**完备模态条件**（§7），
  并在 `audit.json` / `manifest.json` 记录剔除前后的队列口径。**这一步会使既有
  `prepared/mimic_iii` 缓存失效，需重跑 `prepare` 与 `features`。**
- `configs/`：新增 `mimic_mosaic.yaml`，含 `eta`、`K`、`n_mix`、`M_ws`、`lambda_alpha`、`min_observed_bins`。
- 诊断写入既有 `diagnostics/round_*.json`：每轮记录
  $\{I_x(S)\}$ 的池级分布、$U^{\mathrm{syn}}/U^{\mathrm{add}}$ 直方图、
  observed_fraction 与 $U^{\mathrm{syn}}$ 的相关性、粗筛证书、$\Phi_b$ 轨迹、
  以及**推论 2.1 的线性化告警**。

**实施顺序：**

1. **H0**（几何跨轮有效性）——若不成立，先解决配置问题，否则后续全部无意义；
2. `mosaic/lattice.py` + `_check_*` 的离线验证：在固定 $L$ 上验证命题 2 的数值恒等
   与 $U^{\mathrm{syn}}$ 的非平凡性（H1）；
3. H1'（模态 dropout）——保证融合不坍缩；
4. 接入 AL 主循环跑 H2 主曲线与 §13 消融。

---

## 15. 可声明的贡献

不可把 mixup、Möbius/Shapley 分解、Fisher 实验设计、多模态融合、主动学习任一单点作为创新点。
可声明的表述为：

> 在多模态主动学习中，样本的信息增益此前只能在**融合后**的表示上度量，因而"哪个模态、
> 以及模态的哪种组合带来了增益"不可分辨。本文把 c-最优实验设计的边际增益作为集合函数，
> 沿模态联盟格做 Möbius 分解，并把分解的**零点选取为训练实际使用的模态级 mixup 增广分布**
> （伙伴独立抽取，式 2–3）。由此得到的高阶项 $U^{\mathrm{syn}}$ **按构造**等于
> "已标注集经 mixup 闭包无法合成的那部分设计价值"（命题 3）——这使模态归因结果与
> "标注预算应当购买什么"之间第一次存在可推导的联系，而非启发式加权。
> 命题 2 进一步给出该量的可证伪性：融合映射仿射时高阶项恒为零，故 $U^{\mathrm{syn}}\ne0$
> 是对融合层非线性绑定的一个可检验断言。方法在晚融合架构下可精确枚举全部 $2^M$ 个联盟
> （本仓库 $M=3$、融合仅作用于 3 个 token），无需 Shapley 采样近似。
> 可精确枚举全部 $2^M$ 个联盟（本仓库 $M=3$、融合仅作用于 3 个 token），无需 Shapley 采样近似。
>
> **不宣称的部分：** 协同优先是否带来更高的标注效率（H2）是经验问题，命题 3 不蕴含它；
> 与 PID 的 synergy 项只有语义同向，不存在等式关系；本文的全部设计目标结论继承
> RND-Agent 的 head-local 与 Laplace 近似局限；实验限定在 **MIMIC-III 的完备模态 ICU 队列**
> 上（§7 的队列门控），该队列的疾病谱与全院队列不同，结论不得外推到"MIMIC-III 上"；
> 在 `inherit_across_rounds: false` 的冷启动配置下，几何跨轮有效性（H0）尚未验证，
> 若其不成立，本方法与所有模型依赖型主动学习方法一并失效。
