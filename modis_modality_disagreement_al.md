# MoDIS：以模态分歧、模态不稳定性与单模态充分性为采集判据的多模态主动学习

> 方法名 **MoDIS** = **Mo**dality **D**isagreement · **I**nstability · **S**ufficiency-penalty。
> 本文档把"模态差异大、模态信息不稳定、且非单模态可解释的样本最值得标注"这一直觉，
> 整理为可实现、可消融、可证伪的采集准则，并给出与本仓库既有 CoMAL / MM-CoMAL / MoSAIC
> 三臂完全共享数据、划分、种子与预算的对照口径。

---

## 0. 一句话主张

在标注极度受限（全量训练队列的 20%）的 MIMIC-III 多标签 ICD-9 分组预测任务上，
**一个样本的标注价值取决于它的三个模态"说的话有多不一致、各自说得有多不牢靠、以及是否已经能被
单一模态独自讲完"**。前两者是信息量，第三者是一个**负号项**：单模态即可完整复现融合决策的样本，
标注它主要是在强化分类头对强模态（出院小结文本）的依赖，对融合模块的训练贡献很小。

### 0.1 主线：三个判据度量的是同一个对象

全部三项判据都围绕**同一个对象**——该样本的**融合硬判决集** $\hat P(x)$——展开，只是从三个不同的
方向去问"这个决策是怎么来的"：

| 判据 | 问题 | 读出方式 | 想要的方向 |
|---|---|---|---|
| $\mathcal D$ 分歧 | 三个模态**单独**回答时，答案一致吗？ | 逐模态探针 $q_m$ 与融合 $p$ 的**概率**对比 | 越不一致越好 |
| $\mathcal I$ 不稳定性 | 融合决策**依赖**每个模态多少样本特异证据？ | 只扰动一个模态 token 后重跑融合，看 $\hat P$ 何时改变 | 越易改变越好 |
| $\Gamma$ 单模态充分性 | 融合决策能不能被**某一个**模态独自复现？ | 探针硬判决集与 $\hat P$ 的重合度 | 越不能越好 |

三者的合取语义是：**这个样本的融合决策是"有争议的、边缘的、且无法归因于单一模态的"**，
因此它的真实标签对融合模块的信息量最大。这一句就是全部方法的主线；
后续每一节只做一件事——把上表的一行写成一个无歧义、可计算、可证伪的量。

三者的非冗余性不是假设，是必须实测的命题（§11 的 F2）。

---

## 1. 设定、符号与预算口径

### 1.1 任务与数据

沿用仓库现状（`configs/mimic_a800_144c.yaml`、`mimic_comal/multimodal.py`）：

- 队列：完备模态 ICU 队列。首个 ICU stay 的 48 小时窗口内至少 6 个有效测量分箱
  （`dataset.min_observed_bins: 6`）。这是**队列门控**，不是输入特征。
- 标签：`DIAGNOSES_ICD` 三位 ICD-9 组，取训练集频次前 $C=50$ 个
  （`code_prefix_length: 3`、`label_top_k: 50`、`min_label_frequency: 100`），admission 级多标签。
- 划分：按 `SUBJECT_ID` 确定性 80/10/10，患者不跨 split。文本 SVD、生理量归一化参数只由 train 估计。

### 1.2 模态与模型

模态集合 $\mathcal M=\{\mathrm{txt},\mathrm{ts},\mathrm{dem}\}$，$M=3$，与 `model.py:146-157`
的 `encode_modalities` 一一对应：

| $m$ | 编码器 | token $t_m(x)\in\mathbb R^{h}$ |
|---|---|---|
| $\mathrm{txt}$ | `text_encoder`（TF-IDF+SVD 特征上的 MLP） | `model.py:148` |
| $\mathrm{ts}$ | `measurement_encoder`（2 层 Transformer，取 CLS） | `model.py:155` |
| $\mathrm{dem}$ | `static_encoder`（8 维人口统计/住院信息） | `model.py:156` |

$h=$ `model.fusion_dim` $=256$。融合 $\phi:(t_1,\dots,t_M)\mapsto z\in\mathbb R^h$ 即
`fuse_from_tokens`（`model.py:170-176`）；融合头 $W\in\mathbb R^{C\times h}$，
$p_c(x)=\sigma(\langle W_c,z(x)\rangle+b_c)$。

$L$ 为已标注集 $D_l$，$U$ 为未标注池 $D_u$，$V$ 为参考集（validation split）。
**$V$ 不参与任何采集打分，也不参与阈值估计**（见 §7），只用于模型选择意义上的诊断与 test 前的封闭检查。

### 1.3 预算口径（"只能用 20% 数据"的形式化）

标注预算是**真实标注的调用次数**，与伪标签无关：

$$
b_0+R\cdot q\;\le\;\beta\,|D_{\text{train}}|,\qquad \beta=0.2 .
\tag{0}
$$

$b_0=$ `active_learning.initial_labeled`，$q=$ `query_size`，$R=$ `rounds`$-1$（末轮只训练不查询，
见 `runner.py:222`）。仓库默认 $b_0=3000,q=1000,R=5$，合计 8000 次标注；式 (0) 要求
$|D_{\text{train}}|\ge 40000$，否则必须按比例下调 $b_0,q$。**该预算表在四臂之间逐轮相同**，
初始标注集由同一 `_initial_indices(seed)` 生成（`runner.py:74-96`），否则曲线不可比。

"读取人工标注"由**揭示 MIMIC 真实标签**模拟：被选中的 index 才允许把 `labels[index]` 加入训练；
未被选中的样本在任何损失中都不得出现其真实标签。

---

## 2. 三个读出量

整个方法只需要三种前向输出。它们全部建立在**缓存的模态 token** 上，因此每轮只需一次编码器前向。

### 2.1 融合读出 $p$

$p_c(x)=\sigma(\langle W_c,\phi(t_1,t_2,t_3)\rangle+b_c)$，由
`classifier_outputs_tensors(..., return_tokens=True)`（`training.py:1019-1070`）一次给出
$\{t_m(x)\}$、$z(x)$ 与 $p(x)$。这是 $\hat P(x)$ 的来源，也是三项判据共同的参照对象。

### 2.2 单模态探针 $q_m$

$$
q_{m,c}(x)=\sigma\big(\langle w_{m,c},\,\mathrm{sg}[t_m(x)]\rangle+b_{m,c}\big),
\qquad g_m\in\mathbb R^{C\times h}.
\tag{1}
$$

$\mathrm{sg}[\cdot]$ 为 stop-gradient。探针只在 $L$ 上用与融合头相同的加权 BCE
（`_pos_weight`，`maximum_pos_weight: 20`）训练，与融合头同轮优化。

**（a）为什么 stop-gradient。** 若探针梯度回传进编码器，会发生两件坏事：
其一，采集模块反过来改变了被训练的模型，四臂之间不再只差一个采集函数，
主动学习曲线的差异将混杂"辅助损失的正则效应"与"采集准则的效应"，实验不可归因；
其二，探针目标本身就是在鼓励**每个模态单独可判别**，这与 §5 的单模态充分性惩罚在目标上直接冲突。
stop-gradient 让探针成为纯读出器：`MultimodalFusionClassifier` 的参数轨迹与 CoMAL 臂完全一致
（同种子、同 RNG 消耗顺序下），采集是唯一变量。

**（b）为什么线性——容量对齐，而不是为了省算力。** $\Gamma$（§5）要回答"单个模态能否复现融合决策"，
这个问题只有在**探针与融合头的读出容量相同**时才有意义：融合头是 $z$ 上的一个线性层
（`model.py:126`），因此模态侧的读出也必须是 token 上的一个线性层。
若探针用 MLP，则 $\Gamma=1$ 可能仅仅因为探针比融合头更强，与"该模态信息是否充分"无关，
判据三整个失效。线性还顺带消除了"探针结构"这一自由度：探针没有任何需要调的超参。

**（c）必须声明的局限。** 探针度量的是"当前表征中模态 $m$ 的**线性可读**信息"，不是"模态 $m$ 的原始信息量"。
探针在 $L$ 上的折外技巧分数必须逐轮记录（§6、§11 的 F5）：若某模态探针接近基率基线，
则该模态在 $\mathcal D,\Gamma$ 中的贡献不可解释，必须在结论中显式排除，而不是继续按数值汇报。

### 2.3 干预读出 $p^{(m)}(\cdot;\alpha)$

只替换模态 $m$ 的 token，其余保持真实值，重跑融合与融合头：

$$
p^{(m)}_c(x;\alpha)=\sigma\Big(\big\langle W_c,\;\phi\big(t_1(x),\dots,\underbrace{t_m(x;\alpha)}_{\text{只动这一项}},\dots,t_M(x)\big)\big\rangle+b_c\Big).
\tag{2}
$$

实现上直接复用 `mosaic/tokens.py` 的 `fuse_token_batches` 与 `probabilities_from_fused`，
不需要重跑任何编码器。$t_m(x;\alpha)$ 的定义与零点选择见 §4.1。
按定义 $p^{(m)}(x;1)=p(x)$ 对每个 $m$ 都成立，因此三个模态的干预共享同一个参照决策集 $\hat P(x)$。

---

## 3. 判据一：模态分歧

多标签头输出的是 $C$ 个独立 Bernoulli 而非单纯形上的一点，所以"KL 两个分类头的输出"必须逐标签写；
而逐标签 KL 既不对称又在探针输出饱和时发散，且三模态没有自然的成对推广。
下面的广义 JS 散度同时解决这三点。

### 3.1 定义

设模态可靠度权重 $\pi_{m,c}\ge 0,\ \sum_m\pi_{m,c}=1$（§6 给出估计），混合预测
$\bar q_c(x)=\sum_{m}\pi_{m,c}\,q_{m,c}(x)$，

$$
\boxed{\;
\mathrm{JS}_c(x)=\sum_{m\in\mathcal M}\pi_{m,c}\,
\mathrm{KL}_{\mathrm B}\big(q_{m,c}(x)\,\big\|\,\bar q_c(x)\big)
= H_{\mathrm B}\big(\bar q_c\big)-\sum_m \pi_{m,c}H_{\mathrm B}\big(q_{m,c}\big)
\;}
\tag{3}
$$

其中 $\mathrm{KL}_{\mathrm B}(a\|b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}$，$H_{\mathrm B}$ 为二元熵。

**命题 1（式 (3) 的三条性质）.**
(i) $0\le \mathrm{JS}_c\le H(\pi_{\cdot,c})\le\log M$，恒有限，即使某个探针输出 $0$ 或 $1$；
(ii) 对模态置换对称，不需要指定"哪个是参照分布"；
(iii) 令 $\mathsf M\sim\pi_{\cdot,c}$ 为随机抽取的模态编号、
$\mathsf Y_c\mid \mathsf M=m\sim\mathrm{Bern}(q_{m,c})$，则 $\mathrm{JS}_c=I(\mathsf M;\mathsf Y_c)$。

*证明.* (i)(ii) 由式 (3) 的熵差形式直接可得，且
$H_{\mathrm B}(\bar q)-\sum_m\pi_m H_{\mathrm B}(q_m)\le H(\pi)$ 是广义 JSD 的标准上界。
(iii) 展开 $I(\mathsf M;\mathsf Y_c)=H(\mathsf Y_c)-H(\mathsf Y_c\mid\mathsf M)$ 即为熵差形式。$\square$

性质 (iii) 给了这一项**唯一正确的读法**：$\mathrm{JS}_c$ 就是"答案取决于你问哪个模态"的信息量（单位 nat），
这正是原始直觉"模态信息差异性"的严格表述，且天然处理 $M=3$，不需要成对加和。
它同时给出了对 $\pi$ 的硬性要求：$\pi_{\cdot,c}$ 是"问哪个模态"的**先验**，
必须表达"该模态关于标签 $c$ 值得被听取的程度"——这条要求直接决定了 §6 的估计量形式。

### 3.2 标签支撑集与聚合

只在"至少有一方主张为正"的标签上聚合。设融合伪正集与探针伪正集

$$
\hat P(x)=\{c:p_c(x)\ge\tau_c\},\qquad
\hat P_m(x)=\{c:q_{m,c}(x)\ge\tau_{m,c}\},\qquad
S(x)=\hat P(x)\cup\bigcup_m \hat P_m(x),
$$

若 $S(x)=\varnothing$ 则回退为 $\{\arg\max_c p_c(x)\}$（与 `model.py:684` 的 `fallback` 约定一致）。

$$
\boxed{\;\mathcal D(x)=\frac{1}{|S(x)|}\sum_{c\in S(x)}\mathrm{JS}_c(x)\;}
\tag{4}
$$

**为什么是并集而不是 $\hat P$。** 若只在融合伪正集上聚合，"某个模态坚持为正、融合判负"这一类样本的
分歧会被整体丢弃——而这恰恰是本方法最想要的样本。
**为什么取均值而不是求和。** 求和等价于额外乘了一个标签基数因子，与 CoMAL 已有的
`cardinality_mismatch` 项重复；求和形式列为消融。

---

## 4. 判据二：融合决策对单模态的不稳定性

本项**不使用探针**。它问的是融合模型自己的问题：把某个模态的样本特异证据一点点抽走，
融合决策什么时候改变。

### 4.1 干预路径与零点的选择

模态零点取已标注集上的 token 均值（**模态原型**）

$$
h_{p,m}=\frac1{|L|}\sum_{x\in L}t_m(x),
\qquad
t_m(x;\alpha)=\alpha\,t_m(x)+(1-\alpha)\,h_{p,m},\qquad \alpha\in[0,1].
\tag{5}
$$

$\alpha=1$ 是原样本，$\alpha=0$ 是"该模态只剩群体共性、没有本样本特有信息"。

**（a）在流形性。** $t_m(x;\alpha)$ 是真实 token 与真实 token 均值的凸组合，落在模态 $m$ 的 token
边缘分布凸包内，$\phi$ 在该点的取值不涉及分布外外推。这与
`mosaic_multimodal_synergy_al.md` §3 命题 1 是同一条理由；置零或 mask token 会同时引入
"信息损失"与"分布外响应"，二者不可分离，故不采用。

**（b）与 MoSAIC 随机 mixup 的关系，以及此处的 Jensen gap。** 由于
$h_{p,m}=\mathbb E_j[t_m(x_j)]$，式 (5) 是 `mosaic/intervene.py:19-57` 中 $\lambda$-mixup 干预的
**平均场代理**：把对伙伴的期望移进了非线性 $\phi$ 内部。$\phi$ 是 Transformer，
$\phi(\mathbb E[\cdot])\ne\mathbb E[\phi(\cdot)]$，**这里的 Jensen gap 不为零**。
采用确定性路径的理由不是无偏，而是：$\alpha^\ast$ 是一个**阈值穿越点**，
随机伙伴会让它变成随机变量，需要按伙伴数重复采样才能稳定估计，代价按 `partners` 倍增；
确定性路径给出零方差的估计量。随机伙伴版本列为消融 A8，用于直接测量该 gap 的影响。

**（c）质心是否典型：必须记录的诊断。** 在 $h=256$ 维上，均值向量的范数系统性小于个体 token
（$\|\bar t\|\le\overline{\|t\|}$，且高维下差距可观），因此 $h_{p,m}$ 可能是一个"没有真实病人长这样"的点。
两点缓解与一条诊断：
其一，融合层是 `norm_first=True`（`model.py:99,119`），进入注意力前对每个 token 做 LayerNorm，
范数收缩被部分抵消——但残差支路仍携带未归一化的输入，抵消并不完全；
其二，提供 **medoid 变体**（取 $L$ 中距 $h_{p,m}$ 最近的真实 token）作为配置项与消融；
其三，逐轮记录 $\|h_{p,m}\|\big/\overline{\|t_m\|}$ 与 $h_{p,m}$ 到最近真实 token 的距离（相对平均两两距离）。
该比值显著偏离 1 时，主臂结果必须与 medoid 变体并列汇报。

### 4.2 临界混合系数与网格估计量

沿路径观察**融合**硬判决集 $\hat A^{(m)}(x;\alpha)=\{c:p^{(m)}_c(x;\alpha)\ge\tau_c\}$，
注意 $\hat A^{(m)}(x;1)=\hat P(x)$。定义

$$
\boxed{\;
\alpha_m^\ast(x)=\sup\{\alpha\in[0,1):\hat A^{(m)}(x;\alpha)\neq \hat P(x)\},
\qquad \sup\varnothing:=0 .
\;}
\tag{6}
$$

$\alpha^\ast_m$ 越接近 1，说明只需把模态 $m$ 向群体中心挪动极小一步，融合决策就会改变，
即该样本的融合决策**吊在模态 $m$ 的样本特异证据上、且没有余量**。
$\alpha^\ast_m=0$ 表示沿整条路径决策都不变：模态 $m$ 的样本特异证据对该决策不起作用。

**估计量（这是实际被计算的量，必须按它汇报）.** $\phi$ 非线性，式 (6) 的 $\sup$ 无闭式解。
固定网格 $G_K=\{1-k/K:\,k=1,\dots,K\}$，从 $\alpha=1$ 向下扫描，取**首个**判决集改变的网格点：

$$
\hat\alpha^\ast_m(x)=\max\big\{\alpha\in G_K:\hat A^{(m)}(x;\alpha)\ne\hat P(x)\big\},
\qquad \max\varnothing:=0 .
\tag{7}
$$

**命题 2（估计量的性质）.** (i) $\hat\alpha^\ast_m\le\alpha^\ast_m$，即式 (7) 是保守估计；
(ii) 若对每个 $c$，$\alpha\mapsto p^{(m)}_c(x;\alpha)$ 在 $[0,1]$ 上单调（等价地：每个标签沿路径至多穿越
$\tau_c$ 一次），则 $\alpha^\ast_m-\hat\alpha^\ast_m<1/K$。

*证明.* (i) 由 $G_K\subset[0,1)$ 与 $\max\subseteq\sup$ 得。
(ii) 单调性下"判决集改变"的 $\alpha$ 构成一个以 $\alpha^\ast_m$ 为上端的区间，
该区间长度 $\ge$ 相邻网格间距时必含一个网格点，故首个命中点与 $\alpha^\ast_m$ 相距小于 $1/K$。$\square$

命题 2(ii) 的前提**不被 Transformer 保证**，因此它是一条待检验条件而非假设：
逐轮记录**单调性违反率**——网格上判决改变指示序列不呈 `0…01…1` 形态的候选比例。
该比例高时，$\hat\alpha^\ast$ 只能读作"网格分辨率下的保守脆弱度"，不得声称逼近式 (6)。
可选的二分细化：仅对已命中的样本在 $[\hat\alpha^\ast,\hat\alpha^\ast+1/K]$ 内做 $J$ 步二分，
分辨率降到 $2^{-J}/K$，代价只加在命中子集上；默认 $J=0$。

**阈值必须沿路径固定。** $\hat A^{(m)}(x;\alpha)$ 与 $\hat P(x)$ 用同一组 $\tau_c$，
否则"决策改变"会与"阈值改变"混淆。由此带来一个已知混淆：干预把整个池的预测都向原型方向平移，
在某个 $\alpha$ 上可能出现**全池共同翻转**，这属于群体级平移而非样本特异脆弱性。
因此必须逐轮记录**翻转曲线** $\{\Pr_x[\hat A^{(m)}(x;\alpha)\ne\hat P(x)]\}_{\alpha\in G_K}$；
若某个 $\alpha$ 上翻转率接近 1，该网格点之下的信息已被平移主导。
消融 A14 给出"沿路径按分位数重定阈值"的秩保持变体，用于把平移与重排分离。

### 4.3 跨模态聚合

$$
\mathcal I(x)=\sum_{m}\pi_m\,\hat\alpha^\ast_m(x),
\qquad \pi_m=\frac{\sum_c \pi_{m,c}\,n_c}{\sum_c n_c},
\tag{8}
$$

$n_c$ 为 $L$ 上标签 $c$ 的正例数。

**为什么是加权均值，而不是 $\max_m$ 或 $\min_m$。**
$\max_m$ 表达"至少有一个模态是关键的"——但在本任务上文本模态压倒性强，$\max$ 几乎恒等于
$\hat\alpha^\ast_{\mathrm{txt}}$，判据退化为单模态间隔采样；
$\min_m$ 表达"每个模态都关键"，但 $\hat\alpha^\ast_{\mathrm{dem}}$ 在大多数样本上为 0
（人口统计模态对决策贡献小），$\min$ 会几乎恒为 0 而退化为常数。
加权均值是唯一不在本数据的模态强度分布下退化的聚合方式，
而"决策是否集中在单一模态"这件事由 $\Gamma$ 单独承担，不需要 $\mathcal I$ 兼任。
$\max$、$\min$、等权均值全部列为消融，并强制记录 $\mathcal I$ 与 $\hat\alpha^\ast_{\mathrm{txt}}$ 的
Spearman $\rho$：若 $\rho>0.95$，本项已退化为文本模态的间隔采样，不得作为多模态贡献汇报。

---

## 5. 判据三：单模态充分性惩罚

### 5.1 定义

原始表述"$h_i$ 经分类头后覆盖了多少正样本预测"若写成 $|\hat P_m\cap\hat P|/|\hat P|$，
则一个把所有标签都判正的退化探针会得到覆盖率 1，从而被判为"单模态主导"并被惩罚——方向完全错误。
改用对称的 Jaccard 一致度：

$$
J_m(x)=\frac{|\hat P_m(x)\cap \hat P(x)|}{|\hat P_m(x)\cup \hat P(x)|}\in[0,1],
\qquad
\boxed{\;\Gamma(x)=\max_{m\in\mathcal M} J_m(x)\;}
\tag{9}
$$

（$\hat P_m\cup\hat P=\varnothing$ 时取 $J_m=0$。）$\Gamma(x)=1$ 表示存在某个模态，其硬判决集与融合
硬判决集完全相同：融合模块在这个样本上没有做任何单模态做不到的事。

§7 的分位数阈值还提供了第二重保护：每个模态与融合头在池上的**逐标签正例率被约束为相同**，
因此 $|\hat P_m|$ 与 $|\hat P|$ 期望同量级，$J_m$ 不会被某个模态"多判正"人为抬高或压低。

这里用 $\max$ 而非加权和是有意的：**只要存在一个模态足以复现融合决策，该样本就是单模态可解释的**，
与该模态是否"可靠"无关。这也是三项判据中唯一不使用 $\pi$ 的一项。

### 5.2 惩罚的动机与它的可证伪形式

标注一个 $\Gamma\approx1$ 的样本，其监督信号可以被 $\phi$ 的任一单模态旁路完全吸收，
这会加剧分类头对强模态的依赖（在本任务中就是出院小结文本）。惩罚因子取 $1-\Gamma(x)$。

**这不是一个可以只靠直觉成立的断言。** 它等价于宣称：在固定预算下，被 $(1-\Gamma)$ 上调权重的样本
能带来更好的**融合**性能。可证伪检验见 §11 的 F3/F4。

---

## 6. 可靠度权重 $\pi_{m,c}$



### 6.1 折外技巧分数 + 经验贝叶斯收缩

把 $L$ 划为 $K=5$ 折（按 `SUBJECT_ID` 分组，避免同一患者跨折）。对每折训练一份**折外探针副本**，
得到每个 $x\in L$ 的折外预测 $q^{\text{oof}}_{m,c}(x)$。逐样本对数损失与基率基线：

$$
\ell_{m,c}(x)=-\big[y_c\log q^{\text{oof}}_{m,c}(x)+(1-y_c)\log(1-q^{\text{oof}}_{m,c}(x))\big],
\qquad
\ell^0_c(x)=-\big[y_c\log\hat\pi_c+(1-y_c)\log(1-\hat\pi_c)\big],
$$

$\hat\pi_c=n_c/|L|$。定义**信息增益**与**技巧分数**

$$
G_{m,c}=\overline{\ell^0_c-\ell_{m,c}}\ \ (\text{nat}),
\qquad
R_{m,c}=\frac{G_{m,c}}{H_{\mathrm B}(\hat\pi_c)}\in(-\infty,1].
\tag{10}
$$

$R_{m,c}$ 是"模态 $m$ 消除了标签 $c$ 多少比例的先验不确定性"，分子分母同为 nat，
与式 (3) 的散度同尺度，直接满足 §3.1 对 $\pi$ 提出的要求。

**收缩强度由数据定，不引入超参。** $G_{m,c}$ 是 $|L|$ 个逐样本项的均值，
其标准误 $\mathrm{se}_{m,c}$ 由这些项的样本方差给出，$\mathrm{se}_{R}=\mathrm{se}_{G}/H_{\mathrm B}(\hat\pi_c)$。
以矩估计得到模态 $m$ 的标签间真实方差
$s_m^2=\big[\mathrm{Var}_c(R_{m,c})-\overline{\mathrm{se}^2_{R,m,c}}\big]_+$，
再做精度加权收缩：

$$
\tilde R_{m,c}=
\frac{R_{m,c}/\mathrm{se}^2_{R,m,c}+\bar R_m/s^2_m}
     {1/\mathrm{se}^2_{R,m,c}+1/s^2_m},
\qquad
\pi_{m,c}=\frac{[\tilde R_{m,c}]_+}{\sum_{m'}[\tilde R_{m',c}]_+},
\tag{11}
$$

$\bar R_m$ 为按 $n_c$ 加权的池化技巧分数。$s_m^2=0$ 时式 (11) 退化为完全池化 $\tilde R_{m,c}=\bar R_m$；
分母 $\sum_{m'}[\tilde R_{m'c}]_+<\varepsilon$ 时退化为等权 $1/M$。
稀有标签（$n_c$ 小）的 $\mathrm{se}$ 大，自动被拉向该模态的池化水平——这正是 $\kappa$ 想做但需要人工指定的事，
现在由估计量自身的方差决定。

**代价。** 折外副本只是缓存 token 上的 $K\times M$ 个线性回归（$|L|\le 8000$，$h=256$，$C=50$），
不重训任何编码器或融合模块。用于**给候选打分**的探针仍是在全部 $L$ 上训练的那一份；
折外副本只用来估计 $\pi$。

---

## 7. 阈值与伪标签：分位数匹配

三项判据全部依赖硬判决集，阈值因此是能整体改变采集行为的隐藏自由度。这里不用"在某个集合上最大化 F1"，
理由是：融合模型训练于全部 $L$，任何在 $L$ 上定的阈值都带样本内乐观；而借用 $V$ 又等于偷用预算外的标签，
与式 (0) 的口径冲突。

**定义（分位数阈值）.** 设 $\hat\pi_c=n_c/|L|$ 为标签 $c$ 的已标注流行度。在**当轮候选池**上取

$$
\tau_c=\mathrm{Quantile}_{1-\hat\pi_c}\big(\{p_c(x)\}_{x\in \text{pool}}\big),
\qquad
\tau_{m,c}=\mathrm{Quantile}_{1-\hat\pi_c}\big(\{q_{m,c}(x)\}_{x\in \text{pool}}\big).
\tag{12}
$$

它不使用任何池外标签，不需要折外融合模型（后者每轮要重训 $K$ 次，不可承受），
并且把"融合与三个探针在池上的逐标签正例率"约束为同一个 $\hat\pi_c$。

**关键论点：阈值不需要准，只需要共同。** $\hat\pi_c$ 一定是有偏的——`_initial_indices` 会给每个标签
播种正例，采集本身又会让 $L$ 的标签分布偏离池。但 $\mathcal D$ 与 $\Gamma$ 都是**跨模态对比量**，
$\hat\pi_c$ 的偏差对融合与三个探针是同一个平移，在对比中大部分抵消；
$\mathcal I$ 只用到沿路径固定的 $\tau_c$，偏差同样是共同项。
这与 §6 形成对照：可靠度的样本内偏差**按模态不同**，不抵消，所以那里必须折外；
阈值的偏差是共同的，所以这里不必折外。这是本方法里唯一需要区分对待的两处估计。

**伪标签的使用边界。** 伪标签只用于构造 $\hat P,\hat P_m,S$ 与 $\hat A^{(m)}$，
绝不进入任何训练损失；本方法不含自训练成分，预算式 (0) 只统计真实标注。

**必记诊断。** $\mathbb E_x|\hat P(x)|$ 与 $L$ 上真实平均标签基数之比（式 (12) 下应接近 1，
偏离即说明池与 $L$ 的分布差异已经大到不可忽略）；以及 $\hat\pi_c$ 与初始随机子集上估计值的比值。

---

## 8. 打分合成与批选择

三项判据量纲互不可比（$\mathcal D$ 以 nat 计，$\mathcal I$ 是网格分辨率下的位移，$\Gamma$ 是集合重合度），
因此在**当轮候选池内**做秩归一化：对任一分量 $s$，$u_s(x)=\mathrm{rank}(s(x))/N_{\text{pool}}\in(0,1]$。

$$
\boxed{\;
a(x)=u_{\mathcal D}(x)^{\beta_1}\cdot u_{\mathcal I}(x)^{\beta_2}\cdot u_{1-\Gamma}(x)^{\beta_3}
\;}
\tag{13}
$$

**为什么是乘性。** $\log a=\sum_k\beta_k\log u_k$ 是秩的加权几何平均，实现的是**合取**语义
（三条都要高才入选），与 §0.1 主线一致；加性形式允许单项极端值主导，语义上是析取，作为消融。
乘性形式也与仓库 paper-CoMAL 分数
`sqrt(inverse_positive_evidence) * sqrt(cardinality_mismatch)`（`model.py:654`）同族。

**可辨识性。** $a^k$ 与 $a$ 给出相同排序，故只有比值 $\beta_1:\beta_2:\beta_3$ 可辨识；固定
$\beta_1=1$，搜 $\beta_2,\beta_3\in\{0,\tfrac12,1,2\}$。$\beta_3=0$ 即为"去掉充分性惩罚"的消融臂。
超参搜索只允许用**已标注集上的折外指标**或独立的种子重复，不得用 test。

**批选择。** 主臂用与 CoMAL / MM-CoMAL 完全相同的 top-$q$，不引入额外多样性机制——
否则四臂之间又多了一个变量。批内冗余作为独立消融（在 top-$2q$ 上按融合特征 $z$ 做
$k$-center 贪心取 $q$ 个），且必须与"给 CoMAL 臂也加同样多样性"的对照同时汇报。

---

## 9. 查询算法、两段式筛选与复杂度

### 9.1 算法

```text
输入: 当前轮训练好的 (encoders, φ, W, {g_m}), L, U, q, 网格 K, 工作集 N_work
 1  候选池 P ← 从 U 中无放回抽 candidate_size 个                      (runner.py:239-247)
 2  一次编码前向: tokens {t_m(x)}, 融合 z(x), 概率 p(x)               (classifier_outputs_tensors, return_tokens=True)
 3  π_{m,c} ← 折外技巧分数 + 经验贝叶斯收缩, 式(10)(11)               (K×M 个线性拟合, 只在 L 上)
 4  τ_c, τ_{m,c} ← 池上分位数阈值, 式(12)
 5  P̂, P̂_m, S ← 硬判决集
 6  D(x) ← 式(3)(4);   Γ(x) ← 式(9)                                   # 全池, 逐元素, 代价可忽略
 7  W_set ← 按 u_D·u_{1-Γ} 取前 N_work 个候选                          # 两段式筛选, 见 9.2
 8  for m in 模态, for α in G_K:                                       # 只在 W_set 上
 9      p^{(m)}(·;α) ← probabilities_from_fused(fuse_token_batches(干预 token))
10  α̂*_m ← 式(7);  I(x) ← 式(8)                                       # W_set 外置 0
11  a(x) ← 式(13) 秩归一化后取加权几何平均
12  返回 top-q, 揭示真实标签, L ← L ∪ Q, U ← U \ Q
```

### 9.2 两段式筛选是一个有偏近似，必须按近似汇报

步 7 使得 $\mathcal D$ 低或 $\Gamma$ 高的样本无论 $\mathcal I$ 多大都不可能入选。
这是为控制步 8–9 的成本而**故意引入的偏差**，不是等价变换。
沿用 MoSAIC 的做法（`mosaic/acquire.py:30-51` 的 `_rank_union` 证书），逐轮记录：
最终入选样本中落在工作集秩后 10% 的比例。该比例高说明 $N_{\text{work}}$ 偏紧，需要放大。
$N_{\text{work}}=N_{\text{pool}}$ 即关闭筛选，是精确但更贵的配置。

### 9.3 复杂度（按融合前向次数计，不做有利于本方法的取整）

一次融合前向 = 2 层 Transformer、序列长 3、$h=256$、FFN $4h$，约 $4.7$ M MAC $\approx 9.4$ MFLOP，
加上 $256\times50$ 的头可忽略。

| 项 | 融合前向次数 | 估算 FLOP |
|---|---|---|
| 全池 $\mathcal D,\Gamma$ | 0（复用步 2） | 可忽略 |
| $\mathcal I$，$N_{\text{work}}=5000$，$M=3$，$K=8$ | $1.2\times10^5$ | $\approx1.1$ TFLOP |
| $\mathcal I$，关闭筛选（$N=49152$） | $1.18\times10^6$ | $\approx11$ TFLOP |
| 参考：一轮训练（20 epoch，$|L|=8000$，含 CoMAL 对比损失） | — | $\approx4\text{–}5\times10^1$ TFLOP |
| 参考：MoSAIC 每轮（工作集 1500，8 联盟 ×4 伙伴，含 Fisher 设计） | $4.8\times10^4$ | $\approx0.7$ TFLOP |

必须如实写明的两点：
其一，**融合路径版本的 MoDIS 并不比 MoSAIC 便宜**。关闭筛选时约为 MoSAIC 的 15 倍、
约为一轮训练的四分之一；开启 $N_{\text{work}}=5000$ 的筛选后与 MoSAIC 同量级。
早先"低两个数量级"的说法只对探针路径（$\alpha^\ast$ 有闭式解）成立，选用融合路径后不再成立。
其二，MoDIS 相对 MoSAIC 的实际工程差别是**不需要 Fisher 设计**：
没有 $C=50$ 个 $257\times257$ 矩阵的构建、求逆与 deflation 重算，采集代价是一个固定、
可预测的融合前向倍数，不随标签数二次增长。

显存：不要一次性物化 $[N,M,K,3,h]$ 的干预 token（$49152\times3\times8\times3\times256\times4\,\mathrm B\approx3.6$ GB），
按 $(m,\alpha)$ 流式生成，峰值与步 2 的 token 缓存同量级（约 150 MB）。

---

## 10. 与既有工作与仓库三臂的关系

分歧项（式 (3)(4)）属于 query-by-committee / multi-view co-testing 谱系，
只是把"视图"取为模态、把分歧度量换成多标签下有界对称的广义 JSD，**不作为新贡献主张**。

**可以主张为新的（但需 §11、§12 通过后才能写进结论）：**

- 式 (9) 的**单模态充分性惩罚**：把"该样本是否已能被单一模态讲完"作为**负向**采集信号。
  多模态学习中关于模态偏置/模态懒惰的既有工作（如梯度调制类方法）都是**训练期**干预，
  据检索范围所见没有把它作为**采集期**判据的先例；本文档不对"没有先例"作绝对断言。
- 式 (6)(7) 把"融合决策对单模态样本特异证据的依赖"定义为一条**在流形路径上的阈值穿越点**，
  并给出保守性与网格误差的命题 2 及其前提的可检验形式。
- 式 (13) 把三项组织为合取式秩几何平均，并给出可辨识性与消融协议。

**仓库内的强制基线（不是"相关工作"，是同一实验里的对照臂）：**

| 臂 | 采集信号 | 与 MoDIS 的关键差别 |
|---|---|---|
| `comal` | 正原型证据 $\times$ 基数失配 | 完全不看模态结构 |
| `mm_comal` | 逐视图可靠度 + 证据**离散度** | 离散度是原型证据空间的分歧，MoDIS 是**预测分布**空间的分歧；且 MM-CoMAL 无稳定性项、无充分性惩罚 |
| `mosaic` | Fisher c-最优价值的 Möbius 协同分解 | MoSAIC 度量"融合能提供多少单模态无法解释的**设计信息**"，MoDIS 度量"模态之间**说法**有多不一致 + 决策有多脆弱"。前者是价值分解，后者是预测分歧与间隔，二者可能高度相关——必须实测，见 F2 |
| `random` | 均匀 | 下界 |

MM-CoMAL 的 dispersion 与 MoDIS 的 $\mathcal D$ 在动机上最接近，是最重要的对照；
若二者排序 Spearman $\rho>0.9$，则 $\mathcal D$ 项不构成新贡献，只能靠 $\mathcal I$ 与 $\Gamma$ 立论。

---

## 11. 可证伪断言与必记诊断

每轮写入 `experiments/<name>/diagnostics/round_*.json`：

**F1（判据非退化）.** $\hat\alpha^\ast=0$ 的候选比例、$\Gamma=1$ 的比例、$\mathcal D$ 的分位数。
任一项在 $>0.9$ 的候选上取同一个值，即宣告对应判据在本数据上退化为常数，对应 $\beta$ 无意义。

**F2（不是换皮的既有分数，且三项彼此非冗余）.**
(a) $a(x)$ 与融合预测熵、paper-CoMAL 分数、MM-CoMAL dispersion、MoSAIC 协同分数的 Spearman $\rho$；
任一 $>0.9$ 即表明未引入新信息。
(b) $\mathcal D,\mathcal I,1-\Gamma$ 三者两两的 $\rho$ 与 $(\mathcal D,\Gamma)$ 二维直方图；
任一对 $|\rho|>0.9$ 则该对中的一项应当删除而不是保留。

**F3（充分性惩罚的方向性检验）.** 单独运行 $\beta_3<0$（即**偏好**单模态主导样本）的臂。
若与 $\beta_3>0$ 的最终 F1 差异落在种子噪声内，则式 (9) 无效，必须从方法中删除。
这是本方法最容易证伪的一条，必须做。

**F4（惩罚项不是"短文本探测器"）.** $\Gamma(x)$ 与出院小结字符数、与
`measurement_observed_bins_by_record`（`multimodal.py:311-312`）的相关性。
若被选样本仅由"文本短"解释，则本方法退化为与模态语义无关的启发式，
必须补充"按文本长度分层后 $\Gamma$ 仍有效"的证据。

**F5（探针有效性）.** 逐模态、逐标签的折外技巧分数 $R_{m,c}$ 与其池化值 $\bar R_m$，
以及最终 $\pi_{m,c}$ 矩阵。$\bar R_m\le0$ 的模态在 $\mathcal D,\Gamma$ 中的贡献不可解释，
必须标注并做去掉该模态的敏感性分析。

**F6（干预项的健康度）.** §4.2 的单调性违反率、翻转曲线
$\Pr_x[\hat A^{(m)}(x;\alpha)\ne\hat P(x)]$ 逐 $\alpha$ 值；§4.1(c) 的原型范数比与 medoid 距离；
$\mathcal I$ 与 $\hat\alpha^\ast_{\mathrm{txt}}$ 的 $\rho$。

**F7（阈值与筛选健康度）.** §7 的 $\mathbb E|\hat P|$ 与真实平均基数之比；
§9.2 的工作集边界证书。

所有主结论必须在 2 个种子上给出均值与标准差；单种子曲线不得作为结论依据。

---

## 12. 消融矩阵

| 编号 | 变动 | 检验的命题 |
|---|---|---|
| A0 | 完整 MoDIS | — |
| A1 | $\beta_1=0$（去分歧） | 分歧项是否必要 |
| A2 | $\beta_2=0$（去不稳定性） | 干预项是否必要 |
| A3 | $\beta_3=0$（去充分性惩罚） | 惩罚项是否必要 |
| A4 | $\beta_3<0$ | F3，惩罚方向是否正确 |
| A5 | 成对 KL 求和替代式 (3) | 广义 JSD 是否只是形式改写 |
| A6 | $\pi$ 等权 / 样本内技巧分数 / MM-CoMAL 均值差 | §6.1 三条理由是否真的影响结果 |
| A7 | 式 (8) 改用 $\max_m$、$\min_m$、等权 | §4.3 的聚合论断 |
| A8 | 随机伙伴 mixup 替代原型路径（`partners`$\ge4$） | §4.1(b) 的 Jensen gap 影响 |
| A9 | 原型改用 medoid | §4.1(c) 的质心典型性 |
| A10 | 探针路径 $\alpha^\ast$（扰动后只过 $g_m$，有闭式解） | 模内脆弱性 vs 融合决策脆弱性；同时给出便宜十倍的变体 |
| A11 | 网格 $K\in\{4,8,16\}$ 与二分细化 $J>0$ | 命题 2(ii) 的分辨率影响 |
| A12 | 加性合成替代式 (13) | 合取 vs 析取语义 |
| A13 | 与 paper-CoMAL 分数相乘的混合臂 | MoDIS 与原型证据是否互补 |
| A14 | 沿路径按分位数重定阈值（秩保持） | §4.2 的群体平移混淆 |
| A15 | $N_{\text{work}}\in\{2000,5000,N_{\text{pool}}\}$ | §9.2 筛选偏差 |
| A16 | 探针联合训练（去 stop-gradient） | §2(a) 的公平性论断；结果**不可**与 A0–A15 直接比较采集效果，只能单独解读 |
| A17 | top-$q$ vs $k$-center 批多样性（四臂同时开） | 批内冗余的影响 |

---

## 13. 仓库落点

| 文件 | 改动 |
|---|---|
| `modis/__init__.py`、`modis/probes.py` | `ModalityProbes`（$M$ 个 `nn.Linear(h, C)`，前向对输入 `detach()`）；折外副本训练与式 (10)(11) 的 $\pi$ 估计 |
| `modis/intervene.py` | 式 (5) 的确定性原型路径（含 medoid 选项），复用 `mosaic.tokens.fuse_token_batches` / `probabilities_from_fused` |
| `modis/acquire.py` | 式 (12) 阈值、式 (4)(7)(8)(9) 三项判据、式 (13) 合成、两段式筛选与证书，返回 `MoDISAcquisitionComponents(disagreement, instability, dominance, combined)` |
| `mimic_comal/training.py` | `train_round` 中在融合损失外并行优化探针（对 detached token）；轮末缓存 $h_{p,m}$、$\pi_{m,c}$、折外技巧分数 |
| `mimic_comal/runner.py` | `strategy` 白名单（`runner.py:182`）加入 `modis`；新增 `fuse_mode == "modis"`，候选池前向开 `return_tokens=True` |
| `mimic_comal/diagnostics.py` | 落盘 §11 的 F1–F7 全部指标 |
| `configs/mimic_modis.yaml` | `extends: mimic_comal.yaml`；`active_learning.strategy: modis`；`modis: {beta: [1,1,1], grid_k: 8, bisect_steps: 0, workset_size: 5000, prototype: mean, oof_folds: 5, fusion_batch_size: 4096}` |
| `configs/mimic_modis_smoke.yaml` | CPU 结构检查 |
| `tests/test_modis.py` | 式 (3) 的 $[0,\log M]$ 界与置换对称；命题 2(i) 的保守性（网格 vs 稠密扫描）；线性 $\phi$ 下 A10 闭式解与网格一致；$\Gamma$ 在退化全正探针下不为 1（分位数阈值保护）；$S(x)=\varnothing$ 回退；式 (11) 在 $s_m^2=0$ 时退化为完全池化 |
| `scripts/run_three_methods.sh` | 扩为四臂，共享 `prepared/` 与特征缓存 |

模型架构、`model.initialization: random`、从头训练保证与队列门控全部不变；探针是新增参数，
必须在 checkpoint 中与融合模型分开保存，并在 `source_integrity.json` 中登记。
