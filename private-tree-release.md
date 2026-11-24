# 私有决策树/随机森林的发布边界与隐私责任分析

分析对象：`diffprivlib/models/forest.py` 中的 `RandomForestClassifier`、`DecisionTreeClassifier`、
`_FittingTree`，配合 `diffprivlib/mechanisms/exponential.py` 的 `PermuteAndFlip` 与
`diffprivlib/accountant.py` 的 `BudgetAccountant`。所有结论均基于当前分支 `task/327` 的实际代码，
关键位置以 `文件:行号` 标注。

---

## 1. 训练链路事实：每一步依赖什么数据

### 1.1 输入预处理（`RandomForestClassifier.fit`，forest.py:166–286；`DecisionTreeClassifier.fit`，361–426）

| 步骤 | 代码位置 | 数据依赖 | 隐私责任 |
|---|---|---|---|
| 特征界 `bounds` | forest.py:195–201, 395–401 | 若调用方传入：仅公开信息；若为 `None`，用 `np.min/np.max(X, axis=0)` **从私有特征计算** | 数据派生时发出 `PrivacyLeakWarning`，**不消耗任何预算、不受 DP 保护**，端到端 DP 声明要求调用方提供公开界 |
| 按界裁剪 `_clip_to_bounds` | forest.py:201, 401 | 私有 X，但输出被限制在公开界内 | 裁剪本身不发布任何内容 |
| 类别域 `classes` | forest.py:214–220, 403–409 | 若调用方传入：公开；若为 `None`，用 `np.unique(y)` **从私有标签计算** | 同样是带警告的预算外泄漏；类别域必须视为公开参数才能宣称 ε-DP |
| 标签整数化 `np.searchsorted(classes_, y)` | forest.py:223 | 私有 y | 仅内部表示，不发布 |

结论：**只有在 `bounds` 与 `classes` 由外部公开给定的前提下，后续 ε 预算才覆盖整个训练过程。**
数据派生的界和类别域是库明确警告的预算外通道。

### 1.2 结构选择：先建树、后碰数据（`_FittingTree.build`，forest.py:493–536）

调用顺序在 `DecisionTreeClassifier.fit` 中是先 `fitting_tree.build()` 再 `fitting_tree.fit(X, y)`
（forest.py:415–416），即**整棵树的结构在接触任何训练样本之前已经确定**：

- 每个内部节点的划分特征：`random_state.randint(n_features)`（forest.py:520）；
- 划分阈值：`random_state.uniform(bounds_lower[feature], bounds_upper[feature])`（forest.py:521），
  均匀采样的支撑区间只来自公开 `bounds` 与父节点继承下来的公开子区间；
- 树总是完整生长到 `max_depth`，叶节点数恒为 `2^max_depth`（深度截断在 forest.py:510–517）。

因此树的拓扑、每个节点的特征和阈值**仅仅是公开界、`max_depth`、`n_features` 与随机数种子的函数，
与 X、y 完全无关**。本实现没有基于数据的贪心分裂（没有 Gini/熵/信息增益），结构选择不承担任何隐私责任。

### 1.3 样本落叶（`_FittingTree.apply`，forest.py:578–595；森林分区 forest.py:256–259）

- 森林层面 `bootstrap=False`，样本按 `random_state.permutation(n_samples)`（或顺序 `arange`）后做整数除法
  均匀分到各树：`tree_idxs = ... // (n_samples / n_more_estimators)`（forest.py:258–259）。
  分区只依赖样本数与 RNG，**不依赖任何样本的取值或标签**，每个个体恰好落入一棵树。
- 树内 `apply` 按已建好的公开阈值逐级比较 X，把每个样本路由到唯一叶节点。路由结果是私有信息，
  但它只用于内部计数，**不会逐样本写进发布对象**（见 §3）。

### 1.4 叶标签发布：唯一的预算化私有计算（`_FittingTree.fit`，forest.py:538–576）

对每个**有样本落入的叶**：

```python
counts = [np.sum(leaf_y == cls) for cls in self.classes]
mech = PermuteAndFlip(epsilon=self.epsilon, sensitivity=1, monotonic=True,
                      utility=counts, random_state=self.random_state)
values[leaf, 0, mech.randomise()] = 1
```

（forest.py:562–565）

- 效用 `counts` 是该叶内私有标签的各类计数；改变一个个体的标签（或增删一个个体）对任一类别计数的
  L1 影响至多为 1，故 `sensitivity=1` 正确。
- `monotonic=True` 使指数机制的尺度为 `ε / Δ / (2−1) = ε`（exponential.py:131, 249），
  即该机制是 **ε-DP（δ=0）**，而不是 ε/2。
- 发布的只是一个 one-hot 类标签（argmax 式选择），**原始计数不发布**。

---

## 2. 空叶的输出来源：随机标签，不是数据证据

### 2.1 代码事实

空叶（没有任何样本落入的叶）由单独的循环填充（forest.py:567–572）：

```python
for node in self.nodes:
    if values[node.node_id].sum() or node.left_child != self._TREE_LEAF:
        continue
    values[node.node_id, 0, self.random_state.randint(len(self.classes))] = 1
```

空叶标签是对类别索引 `{0, …, n_classes−1}` 的**均匀随机抽取**：

1. 该分支不读取 `X`、不读取 `y`、不构造计数、不调用任何 DP 机制，因此**不消耗隐私预算**；
2. 它的分布是类别域上的均匀分布，与数据集在统计上独立。

实测验证（400 个随机种子、深度 3 的树、每树 7 个空叶、二分类）：空叶标签经验频率约为
`[0.511, 0.489]`，与均匀分布一致。

### 2.2 一个必须澄清的种子层面耦合

固定同一个整数种子、更换数据集时，空叶标签**可能逐实例不同**。这不是数据依赖：有样本叶上的
Permute-and-Flip 随机化（`randomise` 中的随机排列与伯努利尝试，exponential.py:278–287）消耗的随机数
个数本身依赖计数，会平移 RNG 流上后续空叶 `randint` 的位置。机制随机性不属于 DP 对抗者的观测视图
（相邻数据集比较时机制硬币按惯例不固定），因此：

- **分布层面**：空叶输出对任意数据集都是同一均匀分布，后验不包含任何数据集信息，DP 论证不受影响；
- **工程层面**：不要因为"同种子可复现"就把空叶标签解读为训练数据的函数——它只是 RNG 流的确定性产物。

### 2.3 发布对象中无法区分空叶与实叶

写入 sklearn `Tree` 的节点元数据是硬编码的（`_Node.__iter__`，forest.py:607–619）：
`impurity=0.0`、`n_node_samples=0`、`weighted_n_node_samples=0` 对所有节点一律相同；
`values` 数组对每个叶只保存一个 one-hot 标签。**叶占用情况、样本数、标签计数都不在发布物中**，
外部观察者无法从模型对象识别哪些叶是空叶。

因此对空叶的正确表述是：

> 空叶的预测是一个与数据独立的均匀随机猜测，属于先验填充而非观测结论；
> **没有样本落入的叶不能解释为观测标签**，对落入该叶的查询点，模型输出不构成任何数据证据。

---

## 3. 可公开发布的边界

### 3.1 结构（零预算可发布）

在 `bounds`、`classes` 公开且随机种子不被视为秘密的条件下，以下内容只是公开参数与本地随机性的函数，
不含任何训练数据信息，可完整公开：

- 树的拓扑与深度、叶的总数（恒为 `2^max_depth`）；
- 每个内部节点选用的特征索引与阈值；
- 森林规模 `n_estimators`、`max_depth`、特征维数、类别域本身（须公开给定）；
- 节点上硬编码的不纯度/样本数占位字段（恒为 0，无信息量）。

注意区分：**结构公开 ≠ 路由公开**。任意具体样本落在哪片叶是私有计算，仅内部使用，不在发布物中；
若额外发布"哪些叶非空"或逐样本的 `apply` 结果，就超出了本模型的发布边界，需要另行论证。

### 3.2 预测结论（ε-DP 保护下发布）

以下输出全部派生自叶节点上的 Permute-and-Flip 结果，整体享有下节给出的 ε-DP 保证：

- 每个叶发布的 one-hot 类标签（`Tree.values_`），包括实叶的噪声标签与空叶的均匀随机标签；
- 由此得到的 `predict`、`predict_proba`（单树为 one-hot 概率向量）、`apply` 对新查询点的预测；
- 森林的多数投票聚合结果（投票只处理上述 DP 标签，不引入新的私有数据访问）。

边界之外、**不得计入 ε 保护**的内容：

- 数据派生的 `bounds`（特征极值）与数据派生的 `classes`（`np.unique(y)`，即标签域本身）；
- 任何内部中间量：各叶原始标签计数、样本路由、训练分区排列；
- 对同一数据的重复 `fit` 或 `warm_start` 增量建树——这是对同一数据集的新一轮发布，
  按顺序组合另行累计预算（见 §4.4）。

---

## 4. 隐私预算如何组合

### 4.1 单棵树内：不相交叶 → 并行组合，整树成本 ε

- 每片叶的机制输入是该叶样本的标签计数；一个个体经 `apply` 只落在一片叶上，
  因此任意两个叶机制的输入数据集互不相交，满足并行组合（parallel composition）条件。
- 无论树有多少片叶（深度 5 时即 32 片）、其中多少片非空，**整棵树的隐私成本就是一次 ε**，
  空叶机制免费。实测：深度 4、ε=2 的树，会计只记录一笔 `(2, 0)`。

### 4.2 森林层面：不相交样本分区 → 并行组合，整片森林成本 ε

- forest.py:258–259 的分区与数据取值无关，每个个体恰好进入一棵树；
- 因而各树的训练集互不相交，K 棵树之间同样是并行组合而非顺序组合，
  **整片森林对任意个体的总成本仍为 ε，不随 `n_estimators` 放大**；
- `RandomForestClassifier.fit` 也只在森林层级调用一次 `accountant.spend(self.epsilon, 0)`
  （forest.py:284），实测会计记录为单笔 `(ε, 0)`，与这一分析一致。

这一点与普通（非分区、bootstrap 重采样）随机森林不同：后者每棵树都看到全部样本，必须按顺序组合
累计 Kε；本实现靠"每个体只喂一棵树"的设计换取了 K 倍的预算节省。

### 4.3 库会计的宽松上界与紧凑分析的差异

需要注意 `BudgetAccountant` 的记账方式（accountant.py:223–273）：

- 每棵子树在自己的 `fit` 中各自调用一次 `spend(ε, 0)`（forest.py:424）。子树通过
  `BudgetAccountant.load_default(None)` 共享进程级默认会计（forest.py:160, 354），
  故默认会计上会留下 K 笔 ε 记录；
- 当 `slack=0`（默认）时，`total()` 直接做朴素加法（accountant.py:266–267），
  即默认会计报告 Kε 的**宽松上界**。它没有利用"各树数据集不相交"的结构信息；
- 森林自身传入的会计只收一笔 ε，对应 §4.2 的紧凑并行组合结论；
- 只有设置 `slack > 0`（引入 δ 预算）时，会计才会启用 KOV17/DRV 高级组合公式
  （accountant.py:269–273）。本模型 δ 恒为 0，纯 ε-DP 场景下紧凑结论就是并行组合的 ε。

换言之：**模型的真实保证是 ε-DP；若只看默认会计的朴素累计，会得到 Kε 的保守估计，
那是记账粒度造成的过估计，不是额外的真实损失。**

### 4.4 何时预算会真正相加

- 对**重叠或同一数据集**的多次独立发布（重复 `fit`、另建模型、`warm_start=True` 追加的树在
  forest.py:246–249 之后仍对全量数据重新训练）：顺序组合，成本按次累加；
- 若未来把叶计数、叶占用情况等额外发布，每新增一个访问私有数据的输出都需独立计入；
- 数据派生的 `bounds`/`classes` 无论累计多少 ε 都不受保护，只能通过公开给定来闭合缺口。

---

## 5. 结论速览

1. **结构选择**（特征、阈值、拓扑）只依赖公开特征界与 RNG，先于数据构建，零隐私成本，可公开。
2. **样本落叶与树间分区**是私有计算，但仅内部使用，不逐样本发布；分区与数据取值无关，
   保证一个个体只影响一棵树、一片叶。
3. **叶标签**是唯一的预算化私有输出：实叶经敏感度 1、单调的 Permute-and-Flip 以 ε-DP 发布；
   空叶是与数据独立的均匀随机标签，免费且无信息量，**不得解释为观测标签或数据证据**。
4. **预算组合**：叶间并行组合 → 每树 ε；树间因样本不相交再次并行组合 → 整片森林 ε，
   与 `n_estimators`、叶数无关。库默认会计朴素累加出的 Kε 是宽松上界。
5. **发布边界**：公开结构 + DP 叶标签及其全部预测派生量；数据派生的界/类别域、原始计数、
   路由与占用情况在边界之外。端到端 ε-DP 声明以调用方公开提供 `bounds` 和 `classes` 为前提。
