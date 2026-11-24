# 私有主成分分析中的中心化：均值与协方差链路的隐私责任划分

> 分析对象：本仓库 `diffprivlib` 0.6.6（`diffprivlib/models/pca.py`、`diffprivlib/models/utils.py`、`diffprivlib/tools/utils.py`）。
> 运行环境：Python 3.12.3 / numpy 2.5.5 / scikit-learn 1.9.1。
> 本文所有数值均来自对当前工作树的实际运行，复现代码见附录 A。

---

## 0. 结论摘要

1. **预算账本只有一笔，但内部有两类支出。** `PCA.fit()` 对用户的 accountant 只做一次 `check(epsilon)` 和一次 `spend(epsilon)`（`pca.py:204`、`pca.py:273`）。当 `centered=False` 时，这个 epsilon 在内部被切成均值 `epsilon/2` 与协方差 `epsilon/2`（`pca.py:223`、`pca.py:236`）；当 `centered=True` 时，整个 epsilon 归协方差，均值支出为 0。**"centered=True 省预算"是一种会计错觉**：中心要么是真正的公开先验（此时确实不该花预算），要么是在别处花过预算（例如 DP `StandardScaler`），要么就是一笔未受保护的样本统计量（此时不是省预算，而是漏记预算）。

2. **两种中心对"可公开的主成分结论"的影响不对称。** 私有估计的中心带有噪声 `δ = μ̂ − μ`，它把经验散布矩阵从 `S` 变成 `S + δδᵀ` 附近的秩一扰动：主方向会向 `δ` 旋转，`explained_variance_[0]` 被抬高约 `‖δ‖²`。公开中心没有这一项，但中心本身必须以非样本的方式被证明为公开。因此**同一份 PCA 输出中，`components_` 的可信度和 `explained_variance_ratio_` 的可信度取决于中心来源不同而不同**，不能在报告里写成同一个"eps-DP 结论"。

3. **特征界（`bounds` / `data_norm`）的来源直接决定隐私声明是否成立。** 界进入三个位置：均值的截断域与 Laplace 尺度（`tools/utils.py:288,293`）、协方差的行范数归一因子与截断（`pca.py:234`、`utils.py:92`）。若界由数据派生，机制的敏感度本身依赖数据，输出不再是对任何**事先固定**的 epsilon 的差分隐私；而且库会把派生出的界**原样存回模型对象**（`clf.bounds`、`clf.data_norm`），经 `repr` / `get_params` / `clone` / pickle 直接外泄样本的最小值、最大值与最大行范数。**这条路径上的量绝不能被事后追认成"公开先验"。**

4. **两条落地缺陷与中心来源直接相关**（详见第 2.4、3.4 节）：
   - 均值先于协方差计算，且用的是**内部新建的空 accountant**（`pca.py:224`）。若后续步骤抛异常，DP 均值已经算出并留在 `clf.mean_` 上，而用户 accountant 记录为 0 —— 已支出的预算被漏记。
   - 界退化为单点（`min == max`）时敏感度为 0，机制返回**精确样本均值**，但预算照常扣除。这等于用"公开先验"的外衣发布一个样本统计量。

---

## 1. 训练链路：从 `fit()` 到可公开属性的完整数据流

### 1.1 调用顺序

```
PCA.fit(X)
 └─ sklearn PCA._fit(X)                      # 校验、复制语义、n_components 归一
     └─ PCA._fit_full(X, n_components)       # pca.py:203  —— 本库重写
         ├─ self.accountant.check(epsilon, 0)                       # pca.py:204
         ├─ [centered=True]  mean_ = zeros                          # pca.py:210-211
         ├─ [centered=False] bounds 兜底 = (min(X), max(X))         # pca.py:213-220  ← 数据派生
         │                   mean_ = dp_mean(X, eps/2, bounds)      # pca.py:223-224  ← 独立 accountant
         ├─ X -= mean_                                              # pca.py:226      ← 原地修改
         ├─ data_norm 兜底 = max ‖x − mean_‖₂                       # pca.py:228-232  ← 数据派生、且依赖噪声均值
         ├─ X = clip_to_norm(X, data_norm)                          # pca.py:234
         ├─ covariance_eig(X, eps 或 eps/2, norm=data_norm)         # pca.py:236-238
         │    ├─ cov = XᵀX / norm²                                  # utils.py:92
         │    ├─ 特征值：LaplaceBoundedDomain(0, ∞, sens=2) ×d       # utils.py:96-98
         │    └─ 特征向量：Bingham ×k（逐维投影）                    # utils.py:109-122
         ├─ explained_variance_ = sort(λ)/(n−1)                     # pca.py:245
         ├─ ratio_ = ev / ev.sum()                                  # pca.py:246-247
         ├─ n_components 后处理（'mle' / 0<r<1）                     # pca.py:251-257
         ├─ 写入 components_/ev/ratio_/singular_values_（截断到 k）  # pca.py:266-271
         └─ self.accountant.spend(epsilon, 0)                       # pca.py:273
```

### 1.2 预算流的真实形态

| 步骤 | 机制 | 名义 epsilon | 记账位置 |
|---|---|---|---|
| 均值（仅 `centered=False`） | `LaplaceTruncated`，逐特征 | `epsilon/2`，再按特征数均分（`tools/utils.py:89`） | **内部 `BudgetAccountant()`**（`pca.py:224`），不进入用户账本 |
| 特征值（两模式均执行） | `LaplaceBoundedDomain(lower=0, upper=∞, sensitivity=2)` | `epsilon_0 = ε_cov / (k + [k≠d])`（`utils.py:94`） | 无 accountant，不单独记账 |
| 特征向量（两模式均执行） | `Bingham` | `epsilon_i = ε_cov / (k + [k≠d])`（`utils.py:104`），最后一个特征向量不再实例化 | 无 accountant，不单独记账 |
| 汇总 | — | 用户只见一次 `epsilon` | `pca.py:273` |

实测"一次 `fit` 到底实例化了几次机制"（d=10）：

| 场景 | 特征值机制数 | Bingham 数 | 每次名义 ε | 名义总和 | accountant 记录 |
|---|---|---|---|---|---|
| `centered=True, n_components=2` | 10 | 2 | 1/3 | **4.000·ε** | 1.0·ε |
| `centered=True, n_components=10` | 10 | 9 | 0.1 | **1.900·ε** | 1.0·ε |
| `centered=False, n_components=2` | 10 | 2 | 1/6 | 2.000·ε（协方差）**＋ 0.5·ε（均值，10 次调用）＝ 2.500·ε** | 1.0·ε |

（`centered=False` 时均值先吃掉一半预算，协方差层只剩 `ε/2`，所以每次机制的 epsilon 是 `centered=True` 时的一半；名义总和仍大于账本记录值。）

**这里有一个必须写进隐私声明的差别**：账本记的是 `ε`，而基础组合（basic composition）给出的是 `(d + k − [k=d]) · ε_cov / (k + [k≠d])`（最后一个特征向量不花预算，见 `utils.py:103`），当 `k ≪ d` 时显著大于 `ε_cov`。库依赖所引论文（[IS16b] / Biswas et al. 2019）的按投影子空间分配预算的论证来声称总预算是 `ε`；这一论证**没有在本仓库内被验证**（`tests/models/test_covariance_eig.py` 只测 epsilon 无穷大时的正确性和参数校验，不测预算）。因此对外声明时应写成"按文献 [X] 的组合论证为 `ε`-DP"，而不是无条件地写"消耗 `ε`"。

另有一个方向相反的观察：`LaplaceBoundedDomain(lower=0, upper=∞)` 为了满足"有界域"的更强保证，实际加噪比同名义 epsilon 的纯 Laplace 更大。实测 `effective_epsilon()`：名义 0.1667 → 有效 0.0868；名义 0.5 → 0.281；名义 1.0 → 0.620。也就是说**隐私上更保守、效用上更差**：对真实值 0 的归一化特征值，名义 1/6 时 4000 次采样的均值约为 23.1。这解释了为什么在中等 epsilon 下特征谱严重上偏（第 2.3 节）。

### 1.3 中心在哪里被"记住"

- `centered=False`：`self.mean_` 是 DP 均值，属于受保护的发布物。
- `centered=True`：`self.mean_ = zeros`（`pca.py:211`）。模型**不记录**真实使用的中心；真实中心由上游（`StandardScaler` 等）持有并发布。`transform()`/`score()` 因此要求传入**已中心化**的数据。
- 两种模式下 `self.bounds`、`self.data_norm` 都会被保留（`centered=True` 时 `bounds` 保持用户原值或 `None`），它们是参数而非 DP 输出（见第 3 节）。

---

## 2. 两种中心来源对预算支出与可公开结论的影响

### 2.1 预算支出：不是"省不省"，而是"谁承担、剩多少给协方差"

固定总预算 `epsilon=2.0`，四种中心来源的账本实测：

| 中心来源 | `centered` | accountant 记录 | 均值实际支出 | 协方差实际获得 | 中心噪声 |
|---|---|---|---|---|---|
| 库内 DP 均值 | `False` | 2.0 | 1.0（未入账本） | 1.0 | 有 |
| 外部 DP 均值（另一步，如 DP `StandardScaler`） | `True` | 2.0 | 0（在上游账本） | 2.0 | 有（上游） |
| 精确样本均值（sklearn `StandardScaler` / `X.mean(0)`） | `True` | 2.0 | **0，且无人支付** | 2.0 | 有（未受保护） |
| 真正的公开先验（领域常数） | `True` | 2.0 | 0 | 2.0 | 无 |

关键点：

- **第三行是危险行**：`centered=True` 的文档只说"数据假定已中心化（例如用 `StandardScaler`）"（`pca.py:101-106`），**没有要求那个 `StandardScaler` 必须是差分隐私版本**。若用户用 sklearn 的 `StandardScaler`，PCA 会以完整 eps 处理协方差，而中心 `x̄` 是同一批样本的未加噪统计量——整条链路没有 epsilon 可以覆盖它。
- **第四行才是"零成本"成立的条件**：中心与训练样本在分布意义上独立（例如传感器零点、法令基准值）。这时"公开中心 + 公开 `data_norm`"才是一个可以写进声明的前置条件。
- **第二行不是免费的**：预算转移到了上游，管线的总预算相加。`centered=True` 只改变"协方差步骤还剩多少预算"，不改变总账。

**账本可读性的坑**：`BudgetAccountant.load_default(None)` 返回的是一个进程级单例（`accountant.py:417-437`）。实测两个 PCA 模型共用同一个默认 accountant，`a1.accountant.total()` 与 `a2.accountant.total()` 都显示累计值（0.3），而不是各自 0.1/0.2；在管线里若各组件都不传 accountant，每步的 `total()` 都是全局累计，而"每步 epsilon 相加"又会重复计数。**要给出管线级声明，必须显式共享一个 `BudgetAccountant`，并让中心化步骤与 PCA 步骤都向它支出。**

### 2.2 中心噪声如何进入协方差：一个秩一污染项

设 `x̄` 为样本均值（`x̄ = (1/n)Σxᵢ`），PCA 使用 `μ̂`，记 `δ = μ̂ − x̄`。因为 `Σᵢ(xᵢ − x̄) = 0`，交叉项为零，展开恒等式为

```
Σᵢ (xᵢ − μ̂)(xᵢ − μ̂)ᵀ = Σᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ + n·δδᵀ
```

除以 `(n−1)` 后即 `explained_variance_` 所度量的散布矩阵。若参照的是总体中心 `μ` 而非 `x̄`，上式多出一个 `O(‖x̄−μ‖·‖δ‖)` 的交叉项，量级通常远小于 `n‖δ‖²`，不改变结论。于是：**中心误差不是"平移一下"就消失的，它以秩一项的形式叠加到协方差上**。后果是

- 最大特征值上抬约 `n‖δ‖²/(n−1) ≈ ‖δ‖²`，第二、第三……个几乎不变 → 谱的**相对**形状被扭曲；
- 第一大主方向向 `δ` 旋转（`cos(PC1, δ)` 显著偏离 0）；
- 若 `‖δ‖` 与真实谱量级相当（小 epsilon、宽 `bounds`、小样本），PC1 会退化为"均值噪声方向"。

实测（`n=3000, d=4`，数据 `X = 0.7·N(0,I) + [50,−30,20,10]`，`bounds=(−60,80)`，`random_state=1`；真实散布谱 [0.5341, 0.4918, 0.4704, 0.4627]）：
为避免协方差层噪声混入，这里不用 `PCA` 的完整输出，而是直接把中心换成 DP 均值后计算 `(X − μ̂)ᵀ(X − μ̂)/(n−1)` 的谱，从而**只观察中心项**：

| 均值 eps | ‖δ‖² | 预测 `λ₁ + n‖δ‖²/(n−1)` | 实测 `λ₁` | top ratio（真实 0.273） | `cos(PC1, δ)` |
|---|---|---|---|---|---|
| 0.5 | 0.1705 | 0.7046 | **0.7208** | 0.334 | 0.989 |
| 1.0 | 0.0358 | 0.5699 | **0.5753** | 0.287 | 0.934 |
| 2.0 | 0.0062 | 0.5403 | **0.5425** | 0.275 | 0.813 |
| 5.0 | 0.0005 | 0.5346 | **0.5343** | 0.273 | 0.115 |
| 10.0 | 0.0006 | 0.5347 | 0.5332 | 0.272 | 0.690 |

预测与实测在 eps ≥ 1 时吻合到 1% 以内；eps=0.5 时偏差 2.3%，来自 `LaplaceTruncated` 的有界性（噪声不是无界的，`‖δ‖` 的尾部被截断）。eps=5 与 10 的 `cos` 不稳，是因为此时 `‖δ‖` 已极小、`δ` 的方向被数值噪声主导，不代表中心项消失（`‖δ‖²` 本身已降到 5e−4 量级，对谱无实质影响）。**关键是 top ratio：中心噪声把 0.273 抬到 0.334，相对夸大 22%。**

均值噪声的量级由 `(upper − lower)·d / (n·ε_mean)` 决定（`tools/utils.py:293` 的逐特征敏感度 + `tools/utils.py:89` 的按特征均分）。实测每坐标标准差与预测一致（乘 `√2` 的 Laplace 系数）：

| n | d | ε_mean | (hi−lo)d/(nε) | 实测 std |
|---|---|---|---|---|
| 1000 | 2 | 2.0 | 0.1000 | 0.2713 |
| 4000 | 2 | 2.0 | 0.0250 | 0.0678 |
| 4000 | 8 | 2.0 | 0.1000 | 0.2713 |
| 4000 | 2 | 8.0 | 0.0063 | 0.0170 |

因此**中心噪声按 O(d/(nε)) 衰减**：小样本、高维、宽特征界的场景，`centered=False` 的秩一污染可以完全压过真实谱。

### 2.3 对可公开结论的影响：方向、谱、解释比例、组件数

对同一份数据，比较"公开精确中心（`centered=True`）"与"库内 DP 中心（`centered=False`）"，各跑 10 个随机种子取均值。真实谱 `[1.0259, 0.8243, 0.1229, 0.0607]`，真实 top ratio 0.504，真实 `k@90% = 2`；子空间误差取 k=2 的模型，比例统计取 k=d 的模型（与 `'mle'`/小数阈值的决策路径一致）：

| 总 eps | 中心来源 | 子空间误差 `1−σ_min` | `ev1` (k=2) | `ratio1` (k=2) | `ratio1` (k=d) | `k@90%` (k=d) |
|---|---|---|---|---|---|---|
| 0.5 | 公开中心 | 0.062 | 1.109 | 0.502 | 0.497 | 3.0 |
| 0.5 | DP 中心 | **0.226** | 1.276 | 0.467 | 0.439 | 3.1 |
| 1.0 | 公开中心 | 0.025 | 1.053 | 0.509 | 0.505 | 2.7 |
| 1.0 | DP 中心 | 0.037 | 1.104 | 0.468 | 0.470 | 2.9 |
| 2.0 | 公开中心 | 0.011 | 1.038 | 0.506 | 0.507 | 2.6 |
| 2.0 | DP 中心 | 0.016 | 1.033 | 0.485 | 0.473 | 2.7 |
| 10.0 | 公开中心 | 0.002 | 1.028 | 0.505 | 0.506 | 2.0 |
| 10.0 | DP 中心 | 0.004 | 1.026 | 0.502 | 0.501 | 2.2 |

可公开结论的边界：

- **主方向（`components_`）**：两种中心下都是 DP 后处理产物，可以发布；但在 `eps ≲ 1` 且用 DP 中心时，子空间误差会因中心噪声而放大（0.062 → 0.226）。发布"前 k 个主方向"时应同时声明 epsilon 与中心来源。
- **解释方差（`explained_variance_`）**：显著**上偏**。原因有两个叠加：(a) 中心秩一项；(b) `LaplaceBoundedDomain` 在 `lower=0` 一侧的强截断带来的巨大正向偏差（名义 1/6 时对真值 0 的采样均值约 23.1，见第 1.2 节）。**`explained_variance_` 不能当作方差的无偏估计来引用**，只能当作同一 epsilon 下的相对量。
- **解释方差比例（`explained_variance_ratio_`）**：它是 DP 输出的比值，隐私上是后处理（安全），但效用上是"噪声谱的归一化"——小特征值的相对上偏远大于大特征值。实测（真实 top ratio 0.787）：`ratio1` 在 eps=10 时 0.782、eps=1 时 0.673、eps=0.5 时 0.567。**"PC1 解释 x% 方差"这句话在低 epsilon 下系统性低估主成分主导性**（因为噪声摊薄了比例），而单看 `ev1` 又系统性高估。
- **组件数决策（`n_components` 为 0~1 的小数或 `'mle'`）**：由噪声比例决定（`pca.py:251-257`），因此 `n_components_` 本身是 DP 后处理、可以发布，但**不稳定**：真实 `k@90%=2` 的数据上，eps=0.5 时 `k@90%` 平均为 3.0~3.1、eps=10 时才回到 2.0~2.2；且小数阈值选出的 `k` 会让谱偏差更大——同一份数据（真实 `ev1=0.0912`，真实 `k@90%=3`）上，`n_components=2` 的 `ev1` 偏差 +0.0016，而 `n_components=0.5` 的偏差 +0.0086，原因是小数阈值下 `dims=None` 让全部 d 个特征值都加噪、且 `k` 被噪声推大（平均 2.75 而非 2）。
- **`mean_`**：`centered=False` 时是 DP 均值，可发布；`centered=True` 时恒为 0，**不含样本信息，但也不代表"数据均值为 0"**——真实中心在别处被发布，必须在那里声明。

### 2.4 预算漏记：DP 均值算了、留下了、却记作 0

`pca.py:223-224` 用**新建的内部 accountant** 计算均值，而 `spend()` 在函数末尾（`pca.py:273`）。若在均值之后、`spend()` 之前抛异常，用户 accountant 记为 0，但 `clf.mean_` 已经是一个 DP 输出并留在对象上。实测（`data_norm=-1` 触发 `clip_to_norm` 的 `ValueError`，`pca.py:234`）：

```
raised: ValueError Clip value must be strictly positive, got -1.0
accountant spent: (epsilon=0, delta=0.0)
clf.mean_ retained: [2.09 1.79 4.57 2.04]     ← DP 均值已计算且可读
```

这不是"中心来源"本身的问题，而是**中心来源决定了这条失败路径的代价**：`centered=False` 在协方差之前就先花掉了 `eps/2`；`centered=True` 没有这一步，失败时确实没花预算。因此对 `centered=False`，**不能以"fit 抛异常所以没花钱"为前提复用数据**。

---

## 3. 特征界的来源如何约束隐私声明

### 3.1 界进入机制的三条路径

1. **均值**：`bounds=(lower, upper)` 同时决定截断区间（`tools/utils.py:288`）与 Laplace 敏感度 `(upper−lower)/n`（`tools/utils.py:293`）。
2. **协方差的行范数**：`data_norm` 决定归一因子（`utils.py:92`，`cov = XᵀX/norm²`）与截断（`pca.py:234`）。特征值机制的敏感度 2、Bingham 的敏感度 1 都建立在"每行范数 ≤ `data_norm`"这一前提上；`covariance_eig` 会**主动校验**并在越界时抛错（`utils.py:89-90`）。
3. **噪声放大**：`data_norm` 以平方进入噪声——归一化后加噪再乘回 `norm²`（`utils.py:98`）。若 `norm` 比真实行范数大 `c` 倍，噪声功率放大 `c²`。

实测（同一数据、同一 epsilon=1、同一随机种子，真实谱 `[1.63, 1.12, 0.87, 0.23]`、中心化后最大行范数 5.47，只改 `data_norm`）：

| `data_norm` | 发布 `ev1` | `ev2` | `ratio1` | 均值误差 |
|---|---|---|---|---|
| 40 | 60.4 | 24.8 | 0.668 | 0.35 |
| 400 | 5932.4 | 2329.9 | 0.684 | 3.48 |

`norm` 放大 10 倍，`ev1` 放大约 100 倍（`norm²` 定律），而真实 `ev1` 只有 1.63。`ratio1` 看似"稳定"（0.668 vs 0.684）只是因为噪声同时抬高了所有特征值——**它稳定在一个错误的数上**。（均值误差也随 `bounds` 变宽而增大，因为同一 epsilon 下 Laplace 尺度变大。）

**"界选得松一点更安全"是错的**：更松的界保持同样的 eps，但把结论变成噪声。反过来，"界选得紧到刚好包住数据"只有在界独立于数据时才合法。

### 3.2 数据派生的界 = 未保护的样本统计量（核心禁令）

`pca.py:213-220` 与 `pca.py:228-232` 在未提供界时**从数据反推**，仅发一条 `PrivacyLeakWarning` 然后照常使用：

```python
self.bounds = (np.min(X, axis=0), np.max(X, axis=0))          # pca.py:220
self.data_norm = np.linalg.norm(X, axis=1).max()              # pca.py:232（在中心化之后）
```

更严重的是这两个值**被写回模型参数**，因此会成为模型的公开表示的一部分：

```
stored bounds: [46.04 47.09 44.89] [54.54 53.9 53.82]   ← 逐特征精确 min/max
stored data_norm: 7.02                                 ← 精确最大行范数
clone 保留: True    pickle 保留: True    get_params 暴露: True    repr 打印: True
```

（数据为 `np.random.RandomState(0).randn(40,3)*2+50`。注意 `data_norm` 是在减去 DP 均值之后算的，因此它本身也携带均值噪声的信息。）

这意味着：

- 任何"模型发布"（`repr`、`get_params`、序列化、`clone`）都会把**样本的最小值、最大值、最大行范数**以明文送出；
- 这些量**不是 epsilon-DP 的**（它们是数据本身，不是机制的随机化输出）；
- 因此 **`bound`/`data_norm` 一旦来自数据，就不能在事后写成"公开先验"**，也不能把该次 `fit` 声明为 `ε`-DP——机制的实际敏感度是数据依赖的，存在一个与数据相关的 `ε'`，但输出不满足任何**事先选定**的 `ε`。

正确写法只有两种：

- **(A) 先验来源**：界来自领域知识、外部法令/规格、其它独立公开数据集，且在训练前固定。此时可以写"在域 `[L,U]×…`、行范数 ≤ `N` 的先验下为 `ε`-DP"，并把 `bounds`/`data_norm` 当作前提而不是结论。
- **(B) 数据来源**：那就必须**为界本身支付预算**（或使用不泄漏的界估计机制），并在声明中把这次支出计入总账；否则只能声明为"启发式去标识化"，不能声明 `ε`-DP。

一个反直觉但真实的例子：数据派生界会被**异常值/常量列**放大泄漏。实测（`np.random.RandomState(0).randn(100,3)*0.01+3`，把 `[0,0]` 改成 1000）一行离群点把第 0 个特征的数据派生 `bounds` 宽度从 0.046 拉到 997，该特征的均值误差从 `0.00` 变为 `43.78`——发布这个 `bounds` 等于发布了"该特征存在一个到 1000 量级的值"。

### 3.3 界与中心所在的域不一致

`bounds` 在**未中心化**的数据上派生（`pca.py:220`），而 `data_norm` 在**中心化后**的数据上派生（`pca.py:232`，`X -= self.mean_` 之后）。两者不是同一个域。由此产生两类错误：

- **用户提供"原始行的 `data_norm`"**（最常见），但 PCA 实际作用于中心化后的行。中心化可以**增大**行范数：实测 `X = [[100,0],[-100,0],[0,100]]` 的原始最大行范数为 100，精确中心化后为 105.41。此时 `data_norm` 被违反，而 `pca.py:234` 会**静默裁剪**（`covariance_eig` 内部的越界检查根本走不到，因为已经在 PCA 层裁过了）。这不是隐私问题（裁剪后的数据仍在声明域内），而是**域声明与声明者心里的域不一致**：作者以为约束的是原始行，实际约束的是中心化后的行。
- **用户提供"中心化后的 `data_norm`"却传原始数据（或 `centered=True` 却传入未中心化数据）**，则会大面积静默裁剪。实测（每行范数恰为 1 的数据、`centered=False`、`eps=2`、`norm=1`）：中心化后约 **48.8%** 的行被裁，且**没有任何警告**。该例中裁剪对谱的改变不大（`ev=[0.343, 0.341]`，与 `norm=1.15` 不裁剪时的 `[0.352, 0.346]` 相近），所以问题不是"必然算错"，而是**不可见的域偏移**：使用者无从知道声明中的 `N` 是否真的覆盖了实际送入的数据。在 `eps=0.1` 时差异放大到 `ev1` 0.74（裁剪 52.7%）vs 0.895（不裁剪），同时仍无警告。

结论：**界必须与"实际送入 `covariance_eig` 的数据"同一坐标系**。报告中应显式写出"本结论对 `{x : ‖x − c‖₂ ≤ N}` 成立，且 `{x}` 已按公开中心 `c` 中心化"，而不是笼统地给一个 `data_norm`。

### 3.4 退化界：用"公开先验"发布精确均值的漏洞

`bounds=(v, v)` 时敏感度为 0，`LaplaceTruncated` 返回精确均值，但预算照扣：

```
bounds=(5.0,5.0) → mean_ = [5. 5. 5.]（精确），spent = (epsilon=1.0, delta=0.0)
narrow bounds(4.999,5.001) → mean_ ≈ [4.99999868, 5.00012978, 4.99985679]
```

若 `(5,5)` 是真正的公开先验，这是合法的、甚至是正确的用法。但若用户是在数据上取 `bounds = (min, max)`，而某列恰好是常量，则派生的界退化为 `(v,v)`——**精确值被"合法的 0 敏感度"发布出去**。这与第 3.2 节是同一个问题：**决定合法性的是界的来源，不是 `bounds` 的取值形态**。

### 3.5 越界数据被静默钳制，公开界可能完全不覆盖数据

`clip_to_bounds`（`tools/utils.py:288`）会钳制到界内，越界不报警。实测数据 `[0, 100, 200]`、`bounds=(0,1)`、`eps=1`：

```
mean_ = [0.248]，真实均值 = 100.0，警告数 = 0，spent = (epsilon=1.0, delta=0.0)
```

输出对"被钳制后的域"仍是 `ε`-DP，但对真实数据毫无意义。**声明中的界必须至少覆盖数据**，否则"公开先验"是一个错误前提，结论不可用（但隐私声明本身在形式上仍然成立——这正是为什么错误前提最危险）。

### 3.6 声明模板

可以直接采用的措辞（三选一，按界的来源）：

- **可声明**："在事先固定的特征域 `[Lᵢ, Uᵢ]` 与行范数上界 `N`（来源：`<领域文档/规格书/独立公开数据集>`）下，`fit` 以 `ε = …` 满足 `ε`-DP；主方向 `components_`、噪声特征值 `explained_variance_`、比例 `explained_variance_ratio_`、以及 `mean_`（`centered=False` 时）均为该 DP 输出的后处理，可以公开发布。"
- **可声明（外部中心）**："中心 `c` 为公开先验 `<来源>`，与训练样本独立；`centered=True`，故 PCA 将全部 `ε` 分配给协方差估计。若 `c` 由上游 DP 步骤产生，则上游支出 `ε_up` 需与本节 `ε` 相加（`ε_total = ε_up + ε`）。"
- **不可声明**："`centered=True`，所以均值不消耗隐私预算"——除非同时给出中心的公开来源。数据派生 `bounds`/`data_norm`、sklearn `StandardScaler` 的 `mean_`、`n_samples`、样本 min/max，都属于第 3.2 节禁止的事后追认。**同时禁止发布** `clf.bounds`、`clf.data_norm`（当其来自数据时），即使只发布 `repr`。

---

## 4. 其他影响结论可信度的实现问题（与中心链路相关）

以下问题不改变隐私证明，但会让"公开的主成分结论"与实验设置脱节，报告中应避免依赖它们。

1. **特征值与特征向量未正确配对。** 特征值被排序（`pca.py:245`），特征向量按 Bingham 的逐维抽取顺序（`utils.py:111-122`），二者不保证同序。实测 `eps=1`、`d=5`：

   ```
   explained_variance_:        [0.10071 0.02938 0.01848 0.01830 0.01023]   ← sorted(singular_values_²/(n−1))，降序
   singular_values_²/(n−1):    [0.10071 0.02938 0.01023 0.01848 0.01830]   ← 与上一行不同序
   沿 components_[i] 的实际方差: [0.09218 0.01911 0.00417 0.00001 0.00412]   ← 又是一种顺序
   ```

   `get_covariance()` 用 `components_` 与排序后的 `explained_variance_` 重建：重建误差 0.0361，而按"实际方差配对"重建为 0.0235；迹从真实 0.1197 被抬到 0.1771。**因此"第 i 个主成分解释 `explained_variance_[i]`"这一常见表述在本实现下不严格成立**，尤其在低 epsilon 时（`eps=0.5` 的序相关性仅 0.82）。

2. **`fit_transform` 双重中心化。** `_fit_full` 原地执行 `X -= mean_`（`pca.py:226`），随后 `fit_transform` 对同一个 `X` 调用 `transform`，而 `transform` 又会减一次 `mean_`，于是输出等价于 `(X − 2·mean_) @ components_.T`。实测 `centered=False`、`mean_≈[10.13, 10.14, 9.75]`：`fit_transform` 输出均值为 `[-15.53, -7.36]`（恰等于双重中心化），而同一模型 `transform(原始 X)` 的均值为 `[0, 0]`，两者不相等。管线里下游看到的是"被中心化两次"的数据，而同一模型在推理时只中心化一次——**训练/推理分布不一致**。这会让"PCA 降维后分类精度"类结论虚高或虚低。

3. **`copy=True` 不生效。** `pca.py:226` 无拷贝分支；实测 `copy=True` 时传入数组仍被原地修改（改前改后均值从约 10 变为约 0）。因此 `fit(X); transform(X)` 会在"已被中心化的 X"上再投影，与 `transform(原始X)` 结果不同（实测投影均值 `[-8.44, -15.06]` vs `[-0.21, 0.05]`）。基准测试若复用同一数组会得到错误数字。

4. **refit 沿用陈旧的 `bounds`/`data_norm`。** 二者在首次 `fit` 后写回 `self`（`pca.py:220,232`），第二次 `fit` 不再重新派生。实测第一次 `data_norm` 派生为 1.139，第二次数据的最大中心化行范数为 3449.6，仍然使用 1.139，全部静默裁剪且无警告。

5. **`n_components` 边界不一致。** `n_components=10` 而 `d=6` 时，`n_components_=10` 但 `components_.shape==(6,6)`；`n_components='mle'` 在 `n_samples < n_features` 时不报错（sklearn 会报错），实测算出 `n_components_=5`。若报告引用 `n_components_`，会与实际返回的维度不符。

6. **Bingham 拒绝采样可能不终止，而触发条件正是"中心噪声 + 过小的界"。** `bingham.py:145-151` 是无上界 `while True` 的拒绝采样；实测 `d=30`、`n=200`、`centered=False`、`eps=1`、`data_norm=20`、`bounds=(−5,5)` 时 `fit` 在 `bingham.py:146` 内长时间不返回（faulthandler 确认栈），同样的数据在 `centered=True`、`data_norm=1` 下 0.01 s 完成。

   已定位的机理：DP 均值把数据平移到离原点约 `‖μ̂−x̄‖=17.6` 处，中心化后的行范数最大到 21.2，而用户给的 `data_norm=20` 是最小的一侧约束。`covariance_eig` 拿到的归一化矩阵 `XᵀX/norm²` 的迹为 `168.6`（应约为 `n` 的量级，但被中心平移项 `n‖δ‖²` 顶大）。Bingham 的斜坡参数是 `ε_vec·λ_max/4`，其接受概率正比于 `exp(−ε_vec·λ_max/4)`；固定 `eps_vec=1/6` 时，`ε_vec·trace/4` 从 0.44 涨到 7。实测扫描：

   | `data_norm` | trace(cov) | `ε_vec·trace/4` | `fit` 耗时 |
   |---|---|---|---|
   | 80 | 10.5 | 0.44 | 0.01 s |
   | 40 | 43.0 | 1.79 | 0.01 s |
   | 26 | 101.5 | 4.23 | 0.78 s |
   | 20 | 168.6 | 7.03 | **不返回（>5 s）** |

   直接对 Bingham 的合成矩阵扫描也复现同一阈值（`trace≈175` 时挂起，`trace≈51` 时 0.02 s）。**这不是纯粹的数值巧合**：`centered=False` 会产生"中心平移 + 过紧 `data_norm`"的组合，所以它比 `centered=True` 更容易踩到；报告与测试若在低 epsilon 下跑 `centered=False`，应设超时。修复方向：给拒绝循环加迭代上限并回退到"最接近可行"的向量，或在 `data_norm` 明显小于中心化后行范数时直接报错（见 5.1-6）。

---

## 5. 建议

### 5.1 代码层（按优先级）

1. **把 `mean_` 的支出并入用户账本，并在抛异常时也记录。** 将 `pca.py:224` 的 `accountant=BudgetAccountant()` 改为传入用户的 accountant，并接受 `epsilon/2` 的支出；或至少在 `finally` 中按"已计算的机制数"记账。否则第 2.4 节的漏记无法消除。
2. **禁止把数据派生的 `bounds`/`data_norm` 写回模型参数。** 保留在局部变量中，并在 `repr`/`get_params` 中屏蔽；或在 `fit` 后清空。这是第 3.2 节泄漏的唯一直接出口。
3. **区分"公开中心"与"未受保护中心"。** 为 `centered=True` 增加显式参数（如 `center_source="public"`），并在文档/警告中要求上游中心必须是 DP 版本或公开先验；对 sklearn `StandardScaler` 的典型用法给出 `PrivacyLeakWarning`。
4. **修复排序配对**：把 `explained_variance_` 与 `components_` 按同一次排序对齐（或对 `u_mtx` 按 `sigma_vec` 排序后再截断），使"第 i 个主成分"语义成立。
5. **修复 `fit_transform` 双重中心化与 `copy` 语义**，避免训练/推理分布不一致。
6. **refit 时重置数据派生的 `bounds`/`data_norm`**；在界与中心化后数据不一致时**抛错而不是静默裁剪**（至少发 `PrivacyLeakWarning`）。
7. **为 Bingham 拒绝采样加迭代上限**并回退到可行解；在 `centered=False` 且 `data_norm` 小于"中心化后行范数"时**直接报错而不是静默裁剪**——第 4.6 节的挂起正是这两种情况叠加的结果。

### 5.2 报告/声明层

- 每个 PCA 结论旁边标注三件事：`ε`、**中心来源**（DP 均值 / 外部 DP 中心 / 公开先验 / 未受保护均值）、以及**界的来源**（先验 / 数据派生）。
- `explained_variance_` 与 `explained_variance_ratio_` 只作相对量使用；不要写成方差的无偏估计，也不要在 `eps ≲ 1` 时根据比例断言"需要几个主成分"。
- 管线中所有 DP 步骤共享一个 `BudgetAccountant`，并给出**相加后的**总 `ε`；不要引用单个组件的 `.accountant.total()`（默认 accountant 是进程级单例）。
- 不发布 `clf.bounds` / `clf.data_norm`（数据派生时），包括 `repr` 与序列化产物。

---

## 附录 A：复现实验

以下片段可直接在仓库根目录运行（`python3`），均使用仓库内实现，未修改源码。

```python
# A1. 预算账本与中心来源
import numpy as np, warnings; warnings.simplefilter('ignore')
from diffprivlib.models import PCA
from diffprivlib.accountant import BudgetAccountant
rng = np.random.RandomState(0)
X = rng.randn(2000, 4) @ np.diag([2., 1.5, 1., .5]) + np.array([30., -20., 15., 5.])
for centered in (False, True):
    acc = BudgetAccountant(); kw = dict(data_norm=60.0)
    if not centered: kw['bounds'] = (-60, 90)
    PCA(2, epsilon=2.0, centered=centered, accountant=acc, random_state=0, **kw) \
        .fit((X if centered else X.copy()))
    print(centered, acc.total())        # 两者都是 (2.0, 0)

# A2. 中心噪声的秩一污染：λ1 ≈ 真实 λ1 + n‖δ‖²/(n−1)
#     这里绕过协方差机制，只看中心项对散布矩阵的影响
from diffprivlib.tools import mean as dpmean
rs = np.random.RandomState(1)
n, d = 3000, 4; xbar = np.array([50., -30., 20., 10.])
Y = rs.randn(n, d) * .7 + xbar
S0 = (Y - xbar).T @ (Y - xbar) / (n - 1)
w = np.sort(np.linalg.eigvalsh(S0))[::-1]
for eps_mean in (0.5, 1.0, 2.0, 5.0, 10.0):
    mu_hat = dpmean(Y, epsilon=eps_mean/2, bounds=(-60, 80), axis=0, random_state=1)
    delta = mu_hat - xbar
    M = (Y - mu_hat).T @ (Y - mu_hat) / (n - 1)
    wm, V = np.linalg.eigh(M); o = np.argsort(wm)[::-1]; wm = wm[o]; V = V[:, o]
    cos = abs(V[:, 0] @ (delta / np.linalg.norm(delta)))
    print(f"eps={eps_mean}: ||d||^2={delta@delta:.4f} pred={w[0]+n/(n-1)*delta@delta:.4f} "
          f"actual={wm[0]:.4f} ratio={wm[0]/wm.sum():.3f} cos={cos:.3f}")
# true top ratio = 0.273；eps=0.5 时 ratio 被抬到 0.334，pred≈actual

# A3. 界的来源直接外泄：拟合对象携带精确 min/max 与最大行范数
import pickle
from sklearn.base import clone
Z = np.random.RandomState(0).randn(40, 3) * 2 + 50
c = PCA(2, epsilon=1.0, centered=False, random_state=0).fit(Z)
print(np.asarray(c.bounds[0]), np.asarray(c.bounds[1]), float(c.data_norm))   # 精确样本统计量
print(np.allclose(np.asarray(clone(c).bounds[0]), np.asarray(c.bounds[0])))  # True
print(pickle.loads(pickle.dumps(c)).data_norm)                               # 同样保留
# bounds [46.04 47.09 44.89] [54.54 53.90 53.82]，data_norm 7.02

# A4. 退化界 → 精确均值，预算照扣
Z2 = rng.randn(200, 3) * 2 + 5
acc = BudgetAccountant()
c = PCA(2, epsilon=1.0, centered=False, data_norm=20, bounds=(5., 5.), accountant=acc, random_state=0).fit(Z2)
print(c.mean_, acc.total())              # [5. 5. 5.] 与 (1.0, 0)

# A5. 中心化可增大行范数（原始 100 → 中心化后 105.41）
X5 = np.array([[100., 0.], [-100., 0.], [0., 100.]])
print(np.linalg.norm(X5, axis=1).max(), np.linalg.norm(X5 - X5.mean(0), axis=1).max())

# A6. 机制实例数与名义预算总和
import diffprivlib.mechanisms as M
from diffprivlib.models.utils import covariance_eig
rec = []
_o1, _o2 = M.LaplaceBoundedDomain.randomise, M.Bingham.randomise
M.LaplaceBoundedDomain.randomise = lambda s, v: (rec.append(('eig', s.epsilon)), _o1(s, v))[1]
M.Bingham.randomise               = lambda s, v: (rec.append(('vec', s.epsilon)), _o2(s, v))[1]
for dims in (None, 2):
    rec.clear(); X6 = rng.randn(500, 10); X6 -= X6.mean(0); X6 /= np.linalg.norm(X6, axis=1).max()
    covariance_eig(X6, epsilon=1.0, norm=1.0, dims=dims)
    print(dims, sum(1 for k, _ in rec if k == 'eig'), sum(1 for k, _ in rec if k == 'vec'), round(sum(e for _, e in rec), 3))
M.LaplaceBoundedDomain.randomise, M.Bingham.randomise = _o1, _o2
# None -> 10 9 1.900 ；2 -> 10 2 4.000

# A7. 特征值与特征向量不同序
Y7 = rng.randn(2000, 5) @ np.diag([6., 3., 1., .4, .05]); Y7 -= Y7.mean(0); Y7 /= np.linalg.norm(Y7, axis=1).max()
c = PCA(5, epsilon=1.0, centered=True, data_norm=1.0, random_state=0).fit(Y7)
print(np.round(c.explained_variance_, 5))                          # 降序
print(np.round(c.singular_values_ ** 2 / (len(Y7) - 1), 5))        # 不同序
print(np.round([np.var(Y7 @ ci) for ci in c.components_], 5))      # 沿各主成分的实际方差

# A8. fit_transform 双重中心化
Y8 = rng.randn(30, 3) + 10
c = PCA(2, epsilon=1e6, centered=False, data_norm=40, bounds=(-10, 30), random_state=0)
Xt = c.fit_transform(Y8.copy())
print(np.round(c.mean_, 3))                                  # 约 [10.13, 10.14, 9.75]
print(np.allclose(Xt, (Y8 - 2 * c.mean_) @ c.components_.T)) # True：即双重中心化
c2 = PCA(2, epsilon=1e6, centered=False, data_norm=40, bounds=(-10, 30), random_state=0).fit(Y8.copy())
print(np.round(c2.transform(Y8.copy()).mean(0), 3))          # [0, 0]，与 Xt 不同
```

## 附录 B：关键位置索引

| 主题 | 位置 |
|---|---|
| 单次 check / spend | `diffprivlib/models/pca.py:204`、`:273` |
| 均值 epsilon 切分与内部 accountant | `diffprivlib/models/pca.py:223-224` |
| `bounds` 数据派生 | `diffprivlib/models/pca.py:213-220` |
| `data_norm` 数据派生（中心化之后） | `diffprivlib/models/pca.py:228-232` |
| 行范数裁剪（静默） | `diffprivlib/models/pca.py:234`、`diffprivlib/validation.py:135` |
| 协方差预算切分 | `diffprivlib/models/pca.py:236-238` |
| 特征值排序 / 比例 | `diffprivlib/models/pca.py:245-247` |
| `n_components` 后处理 | `diffprivlib/models/pca.py:251-257` |
| 协方差归一化与核验 | `diffprivlib/models/utils.py:83-92` |
| 特征值预算 `epsilon_0` | `diffprivlib/models/utils.py:94-98` |
| Bingham 预算与逐维投影 | `diffprivlib/models/utils.py:104-122` |
| 均值敏感度与截断 | `diffprivlib/tools/utils.py:288,293` |
| 按特征均分 epsilon | `diffprivlib/tools/utils.py:89` |
| 默认 accountant 单例 | `diffprivlib/accountant.py:434-440` |
| Bingham 无上界拒绝采样 | `diffprivlib/mechanisms/bingham.py:145-151` |
