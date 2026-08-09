# MM-CoMAL：CoMAL 的模态分辨式适配（Modality-Resolved CoMAL）

> 定位：本文档是**第二套方案**，与 `mosaic_multimodal_synergy_al.md`（MoSAIC）并列但目标不同。
>
> - **MoSAIC** 是创新投注：新增一层此前不存在的量（模态联盟格 + 协同信息），代价是 $2^M$ 次干预前向。
> - **MM-CoMAL**（本文）是**低风险适配**：**不改动 CoMAL 的主题思想**，只把它的三个组件
>   （标签级隐分解、每标签一个原型 + 共享背景原型、原型相似度×基数失配的采集式）
>   从"融合后单一视图"扩展到"模态分辨的多视图"。额外算力约为 CoMAL 模块的 $(M{+}1)$ 倍，
>   相对分类器可忽略。
>
> **两者不是竞争关系。** MM-CoMAL 同时是 MoSAIC 的**正确对照组**：
> 没有它，MoSAIC 的增益无法排除"任何形式的模态感知都能带来同样收益"这一解释，
> 从而无法归因到"协同信息"本身。见 §11。

---

## 0. 一句话主张

**CoMAL 的采集分数是融合表示的函数，而融合映射不可逆；因此 CoMAL 在原理上无法区分
"三个模态都弱证据"与"两个模态强证据、一个模态强反证"的样本 —— 而这两类样本的标注价值截然不同。**

MM-CoMAL 保留 CoMAL 的全部结构，只做一件事：**把"原型证据"从一个数变成一个按模态分辨的向量**，
再用**逐 (模态, 标签) 的可靠度权重**把它聚合回一个数。
聚合前的离散度就是采集式中新增的唯一乘性因子。

不宣称这是全新范式。见 §12 的诚实定级。

---

## 1. 要修的是什么：CoMAL 在本仓库上的三处失效

沿用 MoSAIC §1 的证据（`experiments/mimic_iii_full_dryrun/final_metrics.json`：
test `auprc_micro` 0.198，`precision_micro` 0.154，约 15 个尾标签 AUPRC 在 0.054~0.09）。
本方案对三处失效的处置**不同于** MoSAIC：

| 失效 | 代码位置 | MoSAIC 的处置 | MM-CoMAL 的处置 |
|---|---|---|---|
| **F1** 采集函数与模态无关 | `model.py:351` | 换掉整个价值函数（Fisher c-最优 + 联盟格） | **保留采集式的代数形状**，把其中的证据项与基数项按模态分辨 |
| **F2** 冷启动下阈值 $\tau_c$ 不稳 | `model.py:325` | 不涉及（不使用 $\tau_c$） | **正面修**：极值统计量 → 均值统计量 + 收缩 |
| **F3** 模态信息稠密度逐样本变化 | 融合处 `model.py:151` | 用 mixup 干预 + Möbius 反演显式度量 | 用**逐样本的模态间证据离散度**近似度量 |

F1 的确切含义值得写清楚。`paper_comal_acquisition_scores` 的全部输入是
`probabilities` 与 `own_similarity`，二者都只经由

```python
fused = self.output_norm(self.fusion_encoder(tokens + self.modality_embedding).mean(dim=1))
```

（`model.py:151`）得到。该映射在 token 三元组上**不是单射**：存在 $x\neq x'$ 使
$z_{\mathrm{fus}}(x)=z_{\mathrm{fus}}(x')$ 而 token 三元组不同。
对这类样本 CoMAL 必然给出相同分数 —— 这不是调参问题，是定义域问题。见命题 2。

---

## 2. 符号

沿用 CoMAL 原有记号，只新增视图下标。

| 符号 | 含义 |
|---|---|
| $\mathcal{M}=\{\mathrm{txt},\mathrm{meas},\mathrm{stat}\}$，$M=3$ | 模态视图 |
| $\mathcal{V}=\mathcal{M}\cup\{\mathrm{fus}\}$，$V=M{+}1=4$ | 全部视图（含融合视图） |
| $z_v(x)\in\mathbb{R}^{D}$ | 视图 $v$ 的表示；$D=$ `fusion_dim` $=256$。$z_{\mathrm{txt}},z_{\mathrm{meas}},z_{\mathrm{stat}}$ 即 `model.py:150` 的 `tokens` 三列，$z_{\mathrm{fus}}$ 即 `fused` |
| $C$ | 标签数（配置 `label_top_k: 50`） |
| $h_{v,c}(x)\in\mathbb{R}^{P}$ | 视图 $v$、标签 $c$ 的隐向量，$P=$ `prototype_dim` |
| $p_{v,c}$，$p_{v,\mathrm{bg}}$ | 视图 $v$ 的标签 $c$ 原型 / 共享背景原型 |
| $s_{v,c}(x)\in[-1,1]$，$e_{v,c}=\tfrac{s_{v,c}+1}{2}\in[0,1]$ | 自原型余弦相似度及其 CoMAL 式 $[0,1]$ 缩放 |
| $r_{v,c}$，$w_{v,c}$ | 逐 (视图, 标签) 可靠度 / 归一化权重 |
| $\tau_{v,c}$ | 逐 (视图, 标签) 正例阈值 |
| $\hat k$ | 已标注集平均基数（CoMAL 原量，不变） |

**关键的量纲事实：$z_{\mathrm{txt}},z_{\mathrm{meas}},z_{\mathrm{stat}},z_{\mathrm{fus}}$ 全部落在同一个 $\mathbb{R}^{256}$ 中**
（`model.py:88-122`：三个模态编码器的输出维度与 `fusion_dim` 相同）。
这不是巧合而是本仓库架构的既有性质，**正是它让"用同一套 CoMAL 参数处理四个视图"成为可能** ——
无需新增任何编码器，这是本方案"最小改动"的技术前提。

**队列口径**：沿用 MoSAIC §7 的完备模态队列门控（排除无 ICU 时序的纯文本病人），
故 $\mathcal{M}$ 对全池样本恒定，无需可用性指示变量。

---

## 3. 改动一：模态分辨的标签级原型（结构）

CoMAL 的主题是"每个标签在隐空间中有自己的正原型，全部负例共享一个背景原型"
（`model.py:159-160`，对应发布代码 `cl_neg_mode=1`）。**这一条不动**，只是对每个视图各做一份。

**共享参数、分视图应用。** 对 $v\in\mathcal{V}$：

$$
h_{v,c}(x)\;=\;W_{\mathrm{lat}}\Big(\big[\,W_{\mathrm{lab}}\big(z_v(x)+u_v\big)\,\big]_c\Big),
\qquad \hat h_{v,c}=\frac{h_{v,c}}{\lVert h_{v,c}\rVert}
\tag{1}
$$

其中 $W_{\mathrm{lab}}=$ `to_label`、$W_{\mathrm{lat}}=$ `to_latent`，**与 CoMAL 完全相同的两层**，
$u_v\in\mathbb{R}^{D}$ 是零初始化的视图码（新增参数，共 $4\times256$ 个标量）。

> **为什么共享 $W_{\mathrm{lab}},W_{\mathrm{lat}}$ 而不是每视图一套？**
> 两个理由。(i) 若每视图独立参数，四套原型将落在四个互不可比的空间，
> $e_{v,c}$ 之间的差值失去意义，§5 的离散度项就成了噪声。
> (ii) 参数共享把"标签 $c$ 的语义"约束为跨模态一致的，这**正是 CoMAL 原论文对原型的解释**
> —— 原型是标签的语义中心，而不是某个特征通道的统计量。视图码 $u_v$ 提供了必要的视图偏置自由度。

**原型刷新**：对每个视图独立执行 CoMAL 原有的规则（`training.py:591-636` 的算术逐视图重复）：

$$
p_{v,c}=\mathcal{N}\!\Big(\sum_{i\in\mathcal{L}} y_{ic}\,\hat h_{v,c}(x_i)\Big),\qquad
p_{v,\mathrm{bg}}=\mathcal{N}\!\Big(\sum_{i\in\mathcal{L}}\sum_{c}(1-y_{ic})\,\hat h_{v,c}(x_i)\Big)
\tag{2}
$$

$\mathcal{N}$ 为 $\ell_2$ 归一化。缓冲区从 $[C{+}1,P]$ 变为 $[V,C{+}1,P]$，即 $4\times51\times64$ 个浮点数，可忽略。

**重构分支不变。** `from_latent`/`aggregate`/`reconstruction_classifier` 仅作用于融合视图
（重构目标仍是 `fused`，与 `training.py:493-494` 一致）。
不为模态视图新增解码器 —— 那会改变 CoMAL 的结构，超出"只做适配"的边界。

---

## 4. 改动二：跨视图对比项（训练）

若只做 §3，四套原型虽在同一参数空间中却未被显式对齐，$e_{v,c}$ 的跨视图比较仍缺乏依据。
CoMAL 的监督对比损失（`model.py:262`）本身就是对齐工具，最小的适配是**扩大它的正样本集**。

记 $\mathrm{SupCon}(\cdot)$ 为 CoMAL 原损失（锚点为 $(i,c)$，类 id 为 $y_{ic}=1\,?\,c:C$）。定义

$$
\mathcal{L}_{\mathrm{con}}
=\underbrace{\frac{1}{V}\sum_{v\in\mathcal{V}}\mathrm{SupCon}\big(\{\hat h_{v,c}(x_i)\},Y\big)}_{\text{视图内：与原 CoMAL 逐字相同}}
\;+\;\beta\cdot\underbrace{\mathrm{SupCon}^{\times}\big(\{\hat h_{v,c}(x_i)\}_{v\in\mathcal{V}},Y\big)}_{\text{跨视图：正负对均要求视图不同}}
\tag{3}
$$

$\mathrm{SupCon}^{\times}$ 与原损失唯一的差别是在正例掩码与 softmax 分母上都追加了
$\mathbb{1}[v\neq v']$，即**只统计跨视图对**。实现上是给
`supervised_contrastive_loss` 增加一个可选的 `view_ids` 参数并追加一个掩码，
不触碰已有的分块与数值稳定逻辑（`model.py:224-259`）。

**$\beta$ 的两端都是失效模式，必须消融。**

- $\beta=0$：视图未对齐，$e_{v,c}$ 不可比，§5 的离散度项退化为噪声。
- $\beta\to\infty$：模态坍缩，$\hat h_{v,c}\approx\hat h_{v',c}$，于是 $G_c\to 0$，
  采集式退回原 CoMAL（见命题 1）。**这不是崩溃，是无声退化** —— 因此必须把池上的平均离散度
  $\bar G$ 作为常规诊断量记录；$\bar G\to 0$ 意味着 $\alpha$ 项失效，而指标不会报错。

建议起点 $\beta\in[0.1,0.3]$。

---

## 5. 改动三：模态分辨的采集函数（核心）

CoMAL 的采集式（`model.py:351-374`）为

$$
\mathrm{score}_{\mathrm{CoMAL}}(x)=\Big(\underbrace{\textstyle\sum_{c\in C^{+}(x)}e_c(x)}_{E(x)}+\epsilon\Big)^{-1/2}\cdot
\Big(\underbrace{\big|\,\#\{c:s_c>\tau_c\}-\hat k\,\big|}_{D(x)}\Big)^{1/2},
\qquad C^{+}(x)=\{c: p_c(x)\ge 0.5\}
\tag{4}
$$

**这个代数形状原样保留。** 只把 $E$、$D$ 两项按模态分辨，并追加一个乘性调制因子。

### 5.1 逐 (视图, 标签) 可靠度

在已标注集 $\mathcal{L}$ 上，对每个 $(v,c)$：

$$
\mu^{+}_{v,c}=\underset{i:\,y_{ic}=1}{\mathrm{mean}}\;e_{v,c}(x_i),\qquad
\mu^{-}_{v,c}=\underset{i:\,y_{ic}=0}{\mathrm{mean}}\;e_{v,c}(x_i),\qquad
r_{v,c}=\big[\mu^{+}_{v,c}-\mu^{-}_{v,c}\big]_{+}
\tag{5}
$$

$r_{v,c}$ 读作"视图 $v$ 的原型几何对标签 $c$ 有多少分辨力"。
尾标签的 $n^{+}_c$ 很小，需向视图级池化值收缩：

$$
\tilde r_{v,c}=\frac{n^{+}_{c}\,r_{v,c}+\lambda_r\,\bar r_{v}}{n^{+}_{c}+\lambda_r},
\qquad \bar r_{v}=\frac{\sum_{c}n^{+}_{c}r_{v,c}}{\sum_{c}n^{+}_{c}}
\tag{6}
$$

权重**只在 $M$ 个模态视图上归一化，不含融合视图**：

$$
w_{v,c}=\frac{\tilde r_{v,c}}{\sum_{v'\in\mathcal{M}}\tilde r_{v',c}},\qquad v\in\mathcal{M}
\tag{7}
$$

> **为什么排除融合视图？** 融合视图正是分类器实际使用的表示，其 $r_{\mathrm{fus},c}$ 通常最大，
> 若纳入归一化会长期主导权重，使 $\bar e_c\approx e_{\mathrm{fus},c}$，方案静默退回 CoMAL。
> 融合视图仍然保留，用于重构分支、$\alpha=0$ 的归约通路与诊断，但不参与证据聚合。
> 这是一个**设计选择而非定理**，配置项 `include_fused_in_weights` 保留以便消融。

**式 (7) 的可辩护性（命题 3，B 级）**：设在标签 $c$ 上各视图证据 $e_{v,c}$ 条件独立、
正负两类同方差 $\sigma_v^2$，则最大化线性组合 $\sum_v w_v e_{v,c}$ 正负均值间隔与标准差之比的权重为
$w_v\propto r_{v,c}/\sigma_v^2$；在等方差假设下即 $w_v\propto r_{v,c}$，恰为式 (7)。
即：**可靠度权重是独立同方差假设下的 Fisher 线性判别方向**。
必须同时申明这两个假设在本数据上都不成立（视图间显然相关），
相关情形的最优权重为 $\Sigma^{-1}r$；在 $n^{+}_c$ 小到个位数时估计 $\Sigma$ 不稳，故本方案有意不采用。

**零零碎碎但重要**：式 (5)(6) 所需的全部统计量都能从**已有的**已标注自相似缓存
（`training.py:508` 的 `labeled_own_cache`，`runner.py:340-355`）按视图维度扩展后直接归约得到，
**不引入任何额外前向**。

### 5.2 阈值：F2 的正面修复

CoMAL 原阈值（`model.py:337-346`）为已标注正例相似度的 **min/max 中点**。
极值统计量在 $n^{+}_c$ 为 1~2 时方差无界 —— 这正是失效 F2。
在**保持"阈值分隔正原型区与背景区"这一原意**的前提下换成均值统计量并收缩：

$$
\tau_{v,c}=\tfrac{1}{2}\big(\mu^{+}_{v,c}+\mu^{-}_{v,c}\big),\qquad
\tilde\tau_{v,c}=\frac{n^{+}_{c}\,\tau_{v,c}+\lambda_\tau\,\bar\tau_{v}}{n^{+}_{c}+\lambda_\tau}
\tag{8}
$$

这是**对 CoMAL 发布代码的一处实质偏离**，必须单独消融（见 §11 假设 H_D），
否则无法区分"多模态适配带来的增益"与"仅仅修了一个估计量的增益"。

### 5.3 聚合与最终分数

$$
\bar e_c(x)=\sum_{v\in\mathcal{M}}w_{v,c}\,e_{v,c}(x),\qquad
\bar\tau_c=\sum_{v\in\mathcal{M}}w_{v,c}\,\tilde\tau_{v,c}
\tag{9}
$$

$$
\bar E(x)=\sum_{c\in C^{+}(x)}\bar e_c(x),\qquad
\bar D(x)=\big|\,\#\{c:\bar e_c(x)>\bar\tau_c\}-\hat k\,\big|
\tag{10}
$$

**模态间证据离散度**（加权平均绝对偏差）：

$$
G_c(x)=\sum_{v\in\mathcal{M}}w_{v,c}\,\big|e_{v,c}(x)-\bar e_c(x)\big|\;\in[0,1],
\qquad
\bar G(x)=\frac{1}{|C^{+}(x)|}\sum_{c\in C^{+}(x)}G_c(x)
\tag{11}
$$

（$C^{+}(x)=\varnothing$ 时定义 $\bar G=0$。）
用加权 MAD 而非极差，是因为极差忽略 $w$：一个 $w_{v,c}\approx0$ 的无关视图不应制造离散度。
当权重集中于单一视图时 $G_c\to0$ —— 这正是所需的退化行为。

**最终采集分数：**

$$
\boxed{\;
\mathrm{score}_{\mathrm{MM}}(x)=\big(\bar E(x)+\epsilon\big)^{-1/2}\cdot\bar D(x)^{1/2}\cdot\big(1+\alpha\,\bar G(x)\big)
\;}
\tag{12}
$$

批选择仍为全池 top-$B$，与 CoMAL 发布实现一致（不引入批内多样性机制，那属于改动主题思想）。

---

## 6. 三条命题

**命题 1（归约 / 严格推广）。** 若 $M=1$，或对每个 $c$ 权重 $w_{\cdot,c}$ 为 one-hot，
则 $\bar G\equiv 0$，式 (12) 退化为式 (4) 在该单一视图上的实例。特别地，
在 `cached_text_mlp3`（纯文本 tf-idf 配置）下 MM-CoMAL 与 CoMAL 逐点相同，
**仅有的差别是 §5.2 的阈值估计量**；取 `threshold_estimator: midpoint` 时二者完全一致。

*证明.* 权重为 one-hot 时式 (11) 中每项 $|e_{v,c}-\bar e_c|$ 在权重非零处为 $0$，故 $G_c=0$，
$\bar G=0$，乘性因子为 $1$；同时式 (9) 给出 $\bar e_c=e_{v^\star,c}$、$\bar\tau_c=\tilde\tau_{v^\star,c}$，
代回式 (10) 即式 (4)。$\square$

**意义**：$\alpha$ 不是一个"加了就有效"的超参，而是一个**带零点的**超参。
$\alpha=0$ 给出一个精确的、可运行的对照臂，这是消融矩阵的锚。

**命题 2（可分性）。** $\mathrm{score}_{\mathrm{CoMAL}}$ 经由映射
$x\mapsto(p(x),z_{\mathrm{fus}}(x))$ 分解；由于
$\mathrm{tokens}\mapsto\mathrm{LayerNorm}(\mathrm{mean}(\mathrm{FusionEnc}(\cdot)))$
（`model.py:151`）在 $\mathbb{R}^{3\times D}$ 上不是单射，
存在 $x\neq x'$ 满足 $z_{\mathrm{fus}}(x)=z_{\mathrm{fus}}(x')$ 且 $p(x)=p(x')$，
此时 $\mathrm{score}_{\mathrm{CoMAL}}(x)=\mathrm{score}_{\mathrm{CoMAL}}(x')$ 恒成立；
而 $\mathrm{score}_{\mathrm{MM}}(x)\neq\mathrm{score}_{\mathrm{MM}}(x')$ 当且仅当二者的
逐视图证据向量 $\{e_{v,c}\}$ 不同。

*证明.* 非单射性：`mean(dim=1)` 已把 $\mathbb{R}^{3D}$ 映到 $\mathbb{R}^{D}$，
纤维维度至少 $2D$；`fusion_encoder` 为连续映射，复合后原像非平凡。其余由式 (12) 的定义直接得到。$\square$

**诚实说明**：这个命题是**平凡的**（本质上是"$\mathrm{mean}$ 会丢信息"），
其价值不在难度而在**它把 F1 从一句抱怨变成一个定义域陈述**。定级见 §12。

**命题 3（可靠度权重的判别式解释）。** 见 §5.1，B 级，假设已完整披露。

---

## 7. 训练侧的前提条件：模态 dropout

这是**必要条件而非可选项**，理由是本仓库特有的：
文本视图是 256 维稠密 SVD，测量视图是 336 维半掩码时序（`multimodal.py:234` 把未观测位置零填充）。
`fusion_encoder` 有很强的动机忽略测量 token。一旦忽略，
`measurement_encoder` 就**收不到任何梯度**（它没有别的下游），
其输出退化为随机投影，于是 $r_{\mathrm{meas},c}\to0$、$w_{\mathrm{meas},c}\to0$，
整个模态适配无声地退回文本单模态。

处置：训练时以概率 $p_{\mathrm{drop}}$ 独立地把某个模态 token 替换为可学习的 `[MISSING]` 嵌入
（不得三个同时丢弃）。建议 $p_{\mathrm{drop}}=0.15$。

**必须常规记录的诊断量**：每轮的 $\bar w_v=\frac{1}{C}\sum_c w_{v,c}$。
若 $\bar w_{\mathrm{meas}}$ 或 $\bar w_{\mathrm{stat}}$ 接近 0，
方案并非失败，而是**在如实报告该模态对当前模型不携带标签信息** —— 结论应据此叙述，
不得掩盖为"多模态方法有效"。

---

## 8. 复杂度与显存

| 项 | 代价 |
|---|---|
| CoMAL 模块前向 | $\times V=4$（`to_label` 是 $256\to C\cdot 64=3200$ 的线性层，相对分类器可忽略） |
| 对比损失 | 锚点数 $\times V$；成对 GEMM 为 $\times V^2=16$。`model.py:279-284` 已有分块保护，需按 $V$ 调低 `anchor_chunk_size` |
| 原型缓冲区 | $4\times51\times64$ 浮点 |
| 已标注 token 缓存 | $3000\times4\times256\times$fp16 $\approx 6$ MB |
| 候选池 token 缓存 | $49152\times4\times256\times$fp16 $\approx 100$ MB |
| 可靠度 / 阈值估计 | 仅在已有 `labeled_own` 缓存上归约，**零额外前向** |

对比 MoSAIC 每样本 $2^M=8$ 次带 mixup 伙伴的融合前向 —— MM-CoMAL 的增量成本约低两个数量级。
**这是本方案存在的主要理由之一。**

---

## 9. 查询算法

```
输入：已标注集 L，候选池 U（已过完备模态队列门控），预算 B，α，β，λ_r，λ_τ
0. 训练分类器（含模态 dropout）；冻结
1. 缓存 L 与 U 的视图表示 z_v(·) ∈ R^{N×4×D}          # 分类器一次前向，附带产出
2. 在 L 上训练 CoMAL 模块，损失 = 式(3) + 重构项(仅融合视图) + 重构 BCE
3. 按式 (2) 逐视图刷新原型 p_{v,c}, p_{v,bg}
4. 在 L 上按式 (5)(6)(7) 估计 r̃_{v,c}, w_{v,c}；按式 (8) 估计 τ̃_{v,c}   # 复用步 3 的缓存
5. 对 U 计算 e_{v,c}，按式 (9)(10)(11) 得 Ē, D̄, Ḡ
6. 按式 (12) 打分，取 top-B
7. 记录诊断：w̄_v（每模态）、Ḡ 的池上分布、n⁺_c 分布、α=0 时的排序 Kendall-τ
```

步 7 的最后一项尤为重要：它直接量化"多模态项到底改变了多少选择"。
若与 $\alpha=0$ 的排序 Kendall-$\tau>0.95$，则本方案在该轮实质上没有生效，
任何指标差异都应归因于噪声而非方法。

---

## 10. 仓库落点

改动原则：**所有已有分支（`random`、`comal/paper`、`comal/weighted`）必须保持逐位不变**，
新增功能一律走新分支。

| 文件:行 | 改动 | 性质 |
|---|---|---|
| `model.py:139-152` | `forward` 返回值追加 `"modality_tokens": tokens`（`tokens` 已在第 150 行物化） | 纯增量，无行为变化 |
| `model.py:155` `CoMALModule` | 新增 `view_code` 参数 $[V,D]$（零初始化）；`prototypes` / `prototype_counts` 缓冲区加视图维；`forward` 接受 $[N,D]$（旧路，数值不变）或 $[N,V,D]$ | 向后兼容 |
| `model.py:262` `supervised_contrastive_loss` | 可选参数 `view_ids`，用于式 (3) 的跨视图掩码 | 默认关闭时逐位不变 |
| `model.py:325` | **不修改**。新增 `shrunk_thresholds(...)`，同时返回 $\tilde\tau_{v,c}$ 与 $\tilde r_{v,c}$ | 基线安全 |
| `model.py:351` | **不修改**。新增 `mm_comal_acquisition_scores(...)` | 基线安全 |
| `training.py:466` `_cache_classifier_features` | 缓存改为 $[N,V,D]$（仅多模态架构） | 内存见 §8 |
| `training.py:591-636` `_refresh_prototypes_from_cached` | 视图维度上加一层循环，逐视图执行既有算术 | 结构不变 |
| `training.py:482-495` | 训练损失换为式 (3)；重构与重构 BCE 仍只作用于融合视图 | |
| `runner.py:328` | 新增 `formula == "paper_mm"` 分支；`paper` / `weighted` 分支不动 | |
| `data.py:194-197` | 完备模态队列门控（与 MoSAIC §7 共用同一实现） | **会使 `prepared/mimic_iii` 现有缓存失效，须重跑 `prepare` + `features`** |

**新增配置键：**

```yaml
model:
  modality_dropout: 0.15          # §7，多模态架构下的前提条件
comal:
  cross_modal_weight: 0.15        # β，式 (3)
acquisition:
  strategy: comal
  formula: paper_mm
  mm:
    alpha: 1.0                    # α，式 (12)；0 = 精确归约到 CoMAL
    reliability_shrinkage: 10.0   # λ_r，式 (6)
    threshold_shrinkage: 10.0     # λ_τ，式 (8)
    threshold_estimator: shrunk   # shrunk | midpoint（midpoint = 复现发布代码）
    include_fused_in_weights: false
    dispersion: weighted_mad      # weighted_mad | range | std
data:
  min_observed_bins: 6            # 队列门控，与 MoSAIC 共用
```

仓库既有的从零训练约束（无 `from_pretrained`、无检查点加载、`model.initialization: random`）
与 `prepared/` 不得入库的规定，本方案全部沿用，无一处放宽。

---

## 11. 待验证假设

按"先证伪成本最低的"排序。

**H_A（实现正确性，非科学假设）。** $\alpha=0$、$\beta=0$、`threshold_estimator: midpoint` 时，
MM-CoMAL 的每轮选择集与 `comal/paper` 基线**完全一致**。
这是命题 1 的直接推论，必须作为单元测试而非实验来验证。不通过则实现有误。

**H_B（主张，实证）。** $\alpha>0$ 相对 $\alpha=0$ 提升尾标签 AUPRC。
主指标：`per_label_auprc` 中 $n^{+}_c$ 最低三分位标签的均值。
副指标：`auprc_micro`、`precision_micro`（现值 0.154，是 CoMAL 最难看的一项）。
**须排除 `per_label_auprc[46]`（0.995 的近恒正标签）后再报 macro**，否则该标签独自主导。

**H_C（模态分辨是真实的，而非记账）。** 检验 $\{w_{v,c}\}_{v}$ 在标签间是否存在非退化差异：
若对几乎所有 $c$ 都有同一个 $v$ 占据全部权重，则式 (11) 恒为 0，命题 1 的退化条件成立，
方案实质未生效。附加检验：$e_{\mathrm{txt},c}$ 与 $e_{\mathrm{meas},c}$ 在池上的 Spearman 相关；
若 $>0.9$ 则视图不携带互补信息。

**H_D（把两处改动分开，最关键的对照）。** $\alpha=0$ 下，`threshold_estimator` 取
`shrunk` vs `midpoint`。这一臂单独测量 §5.2 的阈值修复带来多少增益。
**若 H_D 的增益与 H_B 相当，则本方案的实际贡献是修了一个估计量，而不是多模态适配** ——
这个结论必须如实报告，不得合并叙述。

**H_E（$\beta$ 的两端）。** $\beta\in\{0,0.05,0.15,0.3,1.0\}$，同时记录池上 $\bar G$ 的均值。
预期 $\bar G$ 随 $\beta$ 单调下降；若 $\bar G$ 在 $\beta=0$ 时也接近 0，说明视图本就高度冗余（回到 H_C）。

**H_F（继承自 MoSAIC 的否决项）。** 配置 `inherit_across_rounds: false` 意味着每轮从零重训。
若第 $t$ 轮的原型几何对第 $t{+}1$ 轮从零重训的收益没有预测力，
则**所有依赖模型状态的主动学习方法**（CoMAL、MoSAIC、本方案）在本设置下都不成立。
这不是本方案的缺陷，但它是先决条件，应在投入完整实验前先测：
先跑 `inherit_across_rounds: true` 的对照，确认排序具有跨轮稳定性。

---

## 12. 消融矩阵

| 编号 | 配置 | 隔离的因素 |
|---|---|---|
| A0 | `comal/paper` 原样 | 基线 |
| A1 | `paper_mm`，$\alpha=0,\beta=0$，midpoint 阈值 | **实现正确性**（须与 A0 逐点相同） |
| A2 | $\alpha=0$，shrunk 阈值 | §5.2 阈值修复的单独贡献（H_D） |
| A3 | $\alpha=1$，$\beta=0$ | 未对齐视图下离散度项的贡献 |
| A4 | $\alpha=1$，$\beta=0.15$ | **完整方案** |
| A5 | A4 但 `include_fused_in_weights: true` | §5.1 排除融合视图这一设计选择 |
| A6 | A4 但 $w_{v,c}\equiv1/M$（等权） | 可靠度加权的贡献（对照命题 3） |
| A7 | A4 但 `dispersion: range` | 加权 MAD vs 极差 |
| A8 | A4 但 `modality_dropout: 0` | §7 前提条件的必要性（预期 $\bar w_{\mathrm{meas}}\to0$） |
| A9 | A4，$\alpha\in\{0.5,1,2,4\}$ | 乘性调制强度的敏感性 |
| S1 | A4，`min_observed_bins` $\in\{3,6,12\}$ | 队列门控口径敏感性（**范围检验，非效应消融**） |
| R  | `random` | 采集下界 |

A1 不是实验而是回归测试。S1 改变的是结论适用的人群，不能与 A 系列并列解读。

---

## 13. 与已有工作的关系（分级）

沿用 RND-Agent §8 / MoSAIC §12 的纪律：**A = 可证明；B = 有推导但假设未验证；C = 类比或启发式**。

**A 级。**
- 命题 1 的归约性：$\alpha=0$ 时严格退化为 CoMAL。可验证（A1）。
- 命题 2 的不可分性：CoMAL 分数经融合表示分解，融合非单射。平凡但确切。

**B 级。**
- 命题 3：可靠度权重 = 独立同方差假设下的 Fisher 判别方向。假设在本数据上不成立，已披露。
- 式 (6)(8) 的收缩：形式上是 James–Stein 式收缩，但本文**未证明**其在本问题上的风险占优；
  $\lambda_r,\lambda_\tau$ 是超参而非由理论定出。

**C 级（必须显式承认的重叠）。**
- **式 (11) 的"视图间不一致 ⇒ 值得标注"就是 co-testing / query-by-committee 的思想**
  （Seung 等 1992 的委员会查询；Muslea 等的多视图主动学习 contention points）。
  **本文不宣称这一思想是新的。** 把模态当作委员会成员是自然且已有的做法。
  本方案相对该线的差别仅在三点，且都是适配性的：
  (i) 不一致在 CoMAL 的**标签级原型空间**中度量，因而是**逐标签**一个值而非逐样本一个值 ——
      在平均基数 6.5 的多标签设定下这一区分是实质的；
  (ii) 逐 (模态, 标签) 的可靠度加权，使得对标签 $c$ 无分辨力的视图不制造伪争议；
  (iii) 与 CoMAL 原有的证据项、基数项以乘性方式复合，保证归约性（命题 1）。
- 多模态表示对齐的跨视图对比项（式 3）与 CLIP 式对比对齐、多视图 SupCon 同源，**不新**。

**明确不宣称的：**
- 不宣称本方案度量了"模态的信息增益"这一因果量。$\bar G$ 是**证据离散度**，
  不是任何互信息分解项，也不等价于 PID 的任何分量。要做因果口径的度量请用 MoSAIC。
- 不宣称视图不一致必然意味着信息量大。视图也可能因为噪声而不一致；
  可靠度加权只在**标签级**缓解此问题，对**样本级**噪声无能为力
  （队列门控的 `min_observed_bins` 是一个粗糙的补丁，不是解决方案）。
- 不宣称在完备模态队列之外的人群上成立。结论范围是 **MIMIC-III 的完备模态 ICU 队列**。

---

## 14. 可声明的贡献

**可以说：** 在不改变 CoMAL 主题思想（标签级隐分解、每标签正原型 + 共享背景原型、
监督对比训练、证据×基数失配的采集式）的前提下，给出一个模态分辨的适配：
四个视图共享同一套标签级参数因而原型可比；逐 (模态, 标签) 可靠度权重按 Fisher 判别的形式聚合证据；
采集式以乘性因子引入模态间加权证据离散度，并在 $\alpha=0$ 处**精确**退化为原 CoMAL。
同时正面修复了原实现中基于极值统计量的阈值在小 $n^{+}_c$ 下的不稳定性。
增量算力约为 CoMAL 模块的 4 倍，相对分类器可忽略。

**不能说：** 这是一个新范式，或它度量了模态的信息增益。
它的选择准则在思想上属于多视图主动学习 / 委员会查询这一既有脉络（§13 C 级），
本方案的贡献是把该思想与 CoMAL 的标签级原型几何**正确地**耦合起来，
并给出可归约、可消融、可证伪的形式。这是工程与方法学上的贡献，不是概念上的。

**它在整体研究中的位置：** MM-CoMAL 是 MoSAIC 的对照臂。
若 MoSAIC 相对 A0（原 CoMAL）有增益但相对 A4（本方案）无增益，
则"协同信息"这一概念在本任务上没有超出"朴素模态感知"的价值 —— 这是必须能被观测到的结论，
而只有同时实现两套方案才能观测到它。
