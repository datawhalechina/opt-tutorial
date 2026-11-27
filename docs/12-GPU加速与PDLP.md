# 第 12 章 一阶方法与 GPU 加速（PDLP / PDHG）

第 1 章的 [1.2 节](./01-线性规划.md) 只讲了单纯形法和内点法。本章补上第三条路线：**一阶方法**（first-order method），以及它为什么被拿去跑 GPU。

这一章有一个绕不开的现实约束，先讲清楚，后面的每一节都按这个纪律来：

> **写这章的机器是 Apple M2（8 核 ARM）、macOS，没有 NVIDIA GPU。**
> 所以：
> - 能在 CPU 上跑的（HiGHS 的 PDLP、COPT 的 CPU PDLP 对比）→ **全部实跑，表里是真实数字**；
> - 只能在 GPU 上跑的（Gurobi `PDHGGPU`、COPT 的 GPU 模式、cuPDLP-C）→ 代码给出，但每一处都标 **⚠️ 本机未实跑**，参数名一律用官方文档或本地 introspection 核实；
> - **本章不写任何未实测的加速比和耗时**。凡是"GPU 能快几倍"的问题，本章只给判断清单和测量方法，不给数字。

实测环境：HiGHS 1.16.0（pip wheel）、COPT 8.0.1、gurobipy 14.0.0、Python 3.12.7、Apple M2 / 8 核 / macOS 13.4。

注意版本，不然第一屏就会翻车：**`gurobipy` 要 12 以上才有 PDHG**（本机 `/usr/local/bin/python3`（3.9.12）里装的是 gurobipy 11.0.1，`GRB.Param.PDHGGPU` 直接不存在，`gurobipy._paramdetails` 这个模块也没有）；本章用的 `pdlp_*` 选项是在 `highspy` 1.16.0 上逐个实测过的，更老的 wheel 未必有。`python -c "import highspy, gurobipy"` 先过一遍再往下看。

## 12.1 为什么需要一阶方法

单纯形和内点法有一个共同点：**每一步都要"分解矩阵"**。

| 算法 | 每步在算什么 | 内存瓶颈 |
| :--- | :--- | :--- |
| 单纯形 | 基矩阵 $B$ 的 LU 分解 / 更新，定价（pricing）扫一遍非基列 | 分解产生的 fill-in，稀疏矩阵被填成稠密 |
| 内点法 | 每步解一个 $m\times m$（或 $n\times n$）的 Newton 方程组，即正规方程 $ADA^\top$ 的 Cholesky 分解 | 正规方程的 fill-in，通常比原矩阵大一个量级 |
| 一阶方法 | 只做**矩阵-向量乘** $\mathbf{A}\mathbf{x}$ 和 $\mathbf{A}^\top\mathbf{y}$ | 只需要存下 $\mathbf{A}$ 本身，$O(\text{nnz})$ |

fill-in 麻烦的地方在于它是**超线性**的：约束上千行时分解还很快，十万行时内存和时间都可能直接压垮机器。而矩阵-向量乘的代价严格是 $O(\text{nnz})$，没有 fill-in，且每个非零元素的计算互相独立——这正是 GPU 想要的形状。

代价也很明确，官方文档自己说得很直白（HiGHS GPU 文档原文）：

> First order solvers for LP are still very much "work in progress". Although impressive results have been reported, these are often to lower accuracy than is achieved by simplex and interior point solvers, have been obtained using top-of-the-range GPUs, and not achieved for all problem classes.

所以一阶方法不是"更快的单纯形"，而是**用精度和迭代次数换内存和并行度**。什么时候这笔交易划算，就是 12.3 和 12.5 要说的事。

## 12.2 PDLP 在做什么

本章说的一阶方法，具体指 **PDLP**（Primal-Dual hybrid gradient for Linear Programming），算法本体是 PDHG（Chambolle–Pock 形式的原始-对偶混合梯度），加上一整套工程化改进。

把 LP 写成

$$
\min_{\mathbf{x}} \ \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} = \mathbf{b}, \ \ \mathbf{x} \in \mathcal{X}
$$

其中 $\mathcal{X}$ 是箱式约束（本章的例子里是 $0 \le x \le u$，投影就是逐分量截断）。一次迭代是：

$$
\begin{aligned}
\mathbf{x}^{k+1} &= \mathrm{proj}_{\mathcal{X}}\!\left(\mathbf{x}^{k} - \tau\left(\mathbf{c} + \mathbf{A}^\top \mathbf{y}^{k}\right)\right)\\
\mathbf{y}^{k+1} &= \mathbf{y}^{k} + \sigma\left(\mathbf{A}\left(2\mathbf{x}^{k+1} - \mathbf{x}^{k}\right) - \mathbf{b}\right)
\end{aligned}
$$

- 这里写的是**等式形式** $A\mathbf x = \mathbf b$，目的是把公式讲清楚；本章 12.3 的实例是 $A\mathbf x \ge \mathbf b$，实现里对偶更新要在外层套一次 $\max(0,\cdot)$ 投影（cuPDLP-C 对 $Gx\ge h$ 那一组对偶变量就是这么做的）；
- $\tau,\sigma$ 是原始/对偶步长，要满足 $\tau\sigma\lVert\mathbf{A}\rVert^2 < 1$ 才保证收敛；
- $2\mathbf{x}^{k+1}-\mathbf{x}^{k}$ 是外推（extrapolation）；
- 每轮只有**两次矩阵-向量乘**和若干 $O(n)$ 的向量运算，没有分解、没有基、没有 fill-in。

工程上真正让 PDLP 能用于生产的是三件事，都能在实跑日志里看到痕迹（下一节的日志里 `running scaling / use Ruiz scaling / use PC scaling` 就是）：

1. **缩放（scaling）**：Ruiz 缩放 + Pock–Chambolle 缩放，把系数矩阵的行列量级拉平。一阶方法对量级极其敏感，不做缩放基本没法用。
2. **重启（restart）**：检测到"进展停滞"就把当前点当起点重新开始，比一路硬推快得多。cuPDLP-C 的参数里直接有 `eRestartMethod`（`0` 不重启 / `1` KKT 版本重启，官方参数表原文写的是 `1-KKTversion`——**和 GPU 无关**，别被名字骗了）。
3. **自适应步长**：固定步长对病态问题几乎不可用，实际实现用自适应 / Malitsky–Pock / PID 之类的策略。cuPDLP-C 的 `eLineSearchMethod` 就是它，官方参数表 Range 写的是 `0, 2`：`0` 固定、`2` 自适应，其中 `1`（Malitsky）在表里是**划掉**的。

### 三个关键特性（也是三个坑）

| 特性 | 好的一面 | 代价 |
| :--- | :--- | :--- |
| 只要矩阵-向量乘 | 内存 $O(\text{nnz})$，天然适合 GPU/多核；超大稀疏模型也能塞进去 | 迭代次数动辄上千上万，小模型上"迭代便宜"抵不过"次数太多" |
| 精度可调到"够用就行" | 容差放宽一个量级，迭代次数常常降一大截（12.3 有实测） | 放宽后**不再有"最优解"这个保证**，只有"对偶间隙和不可行度都小于 tol" |
| 不产生基 | 不需要维护基，逻辑简单 | 拿不到基 ⇒ 没法做灵敏度分析、热启动、单纯形交叉验证；要基就得额外跑 crossover |

## 12.3 CPU 实测：PDLP 到底什么时候快

先给结论，再给数字：**在这台 8 核 M2 上，nnz 一万量级时 PDLP 比内点法慢 2~3 倍，nnz 十万量级时反超，nnz 二十万量级时快近 20 倍。** 这不是"PDLP 更快"，而是"PDLP 的耗时随规模增长得更慢"。

### 12.3.1 测试例

`code/ch12_gpu/gen_lp.py` 用固定种子生成四个规模的稀疏 LP：

$$
\min\ \mathbf{c}^\top\mathbf{x}\quad\text{s.t.}\quad \mathbf{A}\mathbf{x} \ge \mathbf{b},\ 0 \le \mathbf{x} \le \mathbf{u}
$$

- 每列恰好 $k$ 个非零，所以 $\text{nnz} = \text{列数}\times k$，稀疏度可控；
- 先随机取一个内点 $\mathbf{x}_0\in(0.3u, 0.7u)$，再令 $\mathbf{b}=0.9\,\mathbf{A}\mathbf{x}_0$ ⇒ **必然严格可行**；
- $\mathbf{c}>0,\ \mathbf{x}\ge 0$ ⇒ 目标有下界 ⇒ **必然有最优解**；
- 同样的 `(rows, cols, k, seed)` 生成完全相同的矩阵（本仓库已验证两次生成逐元素相同）。

四个规模：

| 规模 | 行 | 列 | 每列非零 | nnz |
| :--- | ---: | ---: | ---: | ---: |
| small | 200 | 400 | 4 | 1 600 |
| medium | 1 000 | 3 000 | 6 | 18 000 |
| large | 3 000 | 12 000 | 8 | 96 000 |
| xlarge | 8 000 | 30 000 | 8 | 240 000 |

### 12.3.2 实测结果

`python code/ch12_gpu/bench_pdlp_highs.py --time-limit 120`，真实输出（耗时是 wall clock，含预处理；小实例取 3 次最小值，大实例只跑 1 次）：

| 规模 | nnz | 算法 | 状态 | 迭代次数 | 耗时 | 单次迭代 | 与参考解相对偏差 | 有基 |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| small | 1 600 | simplex | 最优 | 384 | 0.006 s | ~16 µs | 0 | 是 |
| | | ipm | 最优 | 17 | 0.005 s | ~0.29 ms | 2.8e-16 | 是 |
| | | **pdlp** | 最优 | 2 520 | 0.010 s | ~4 µs | 1.4e-09 | 否 |
| medium | 18 000 | simplex | 最优 | 5 085 | 0.743 s | ~0.15 ms | 0 | 是 |
| | | **ipm** | 最优 | 20 | **0.134 s** | ~6.7 ms | 5.2e-14 | 是 |
| | | pdlp | 最优 | 9 040 | 0.338 s | ~37 µs | 4.1e-11 | 否 |
| large | 96 000 | simplex | 最优 | 32 708 | 74.0 s | ~2.3 ms | 0 | 是 |
| | | ipm | 最优 | 23 | 4.57 s | ~0.20 s | 6.9e-11 | 是 |
| | | **pdlp** | 最优 | 9 160 | **1.66 s** | ~0.18 ms | 4.8e-09 | 否 |
| xlarge | 240 000 | simplex | **触时限（120 s）** | 34 584 | 120.8 s | ~3.5 ms | 5.6e-02 | 是 |
| | | ipm | 最优 | 30 | 96.2 s | ~3.2 s | 0 | 是 |
| | | **pdlp** | 最优 | 9 560 | **4.96 s** | ~0.52 ms | 1.2e-09 | 否 |

参考解取"第一个证到最优的那个算法"的目标值（检查顺序是 simplex → ipm → pdlp）：small / medium / large 都是 simplex 先到，取 simplex 的目标值；xlarge 上 simplex 触了时限，退到 ipm。那个 5.6e-02 的偏差行（xlarge / simplex）是**没解完的中间解**，连当参考值的资格都没有。

> **数字随机器负载波动，请以你自己的重跑为准。** 同一个脚本、同一台机器，三个轮次就能差出这么多：
>
> | | 本轮 | 一次桌面有负载的重跑 |
> | :--- | ---: | ---: |
> | large / simplex | 74.0 s | 88.0 s |
> | large / ipm | 4.57 s | 5.14 s |
> | xlarge / ipm | **96.2 s（最优）** | **>120 s（触时限）** |
> | xlarge / pdlp | 4.96 s | 6.12 s |
>
> **迭代次数是稳定的**（2 520 / 9 040 / 9 160 / 9 560 每次一模一样，只有触时限的行会变，因为迭代数由截止时间决定），**耗时是抖的**；`xlarge / ipm` 甚至会在"96 s 收工"和"120 s 还没收工"之间翻状态——这直接影响"谁是参考解"。所以本章凡是给结论的地方都留了余量，你复现出 ±30% 的差异不是脚本写错了。

### 12.3.3 怎么读这张表

**1）小模型上 PDLP 就是更慢，没有例外。** small 上 PDLP 比 ipm 慢 2.0 倍：迭代 2 520 次 vs 17 次。单次迭代 PDLP 只花 4 µs（内点 0.29 ms），但架不住次数是 148 倍。HiGHS 官方文档也是这么说的：

> PDLP works in double precision, and each iteration is cheap, so a GPU only pays off for large LPs. For small LPs, the cost of launching GPU kernels dominates, and HiPDLP on a GPU can be slower than on the CPU.

**2）"单次迭代代价"这一列才是 PDLP 的本钱。** 看 nnz 从 96k 涨到 240k（2.5 倍）：PDLP 单次迭代从 0.18 ms 涨到 0.52 ms（2.9 倍，和 nnz 的增长同量级，符合"只做矩阵-向量乘"）；内点法单次迭代从 0.20 s 涨到 3.2 s（16 倍，典型的分解 fill-in 超线性增长）。**两者单次迭代差 3 个数量级以上**（3.2 s vs 0.52 ms，约 6 000 倍），PDLP 靠多迭代几千次也追回来了。

**3）交叉点在 nnz 两万到十万之间。** 本机只采了 18k 和 96k 两个点，100k 附近没细扫；本机数据：

| 规模 | PDLP / ipm 耗时比 | 谁快 |
| :--- | ---: | :--- |
| small（1.6k） | 2.0× | ipm |
| medium（18k） | 2.5× | ipm |
| large（96k） | 0.36× | **PDLP 快 2.8 倍** |
| xlarge（240k） | 0.052× | **PDLP 快 19.4 倍** |

**4）单纯形在这个测试例上最早垮掉，但这不能推广。** large 上单纯形迭代 32 708 次、74.0 s；xlarge 上 120 s 还没证到最优，解差 5.6%。一个可能的原因是这批随机稀疏 LP 的退化比较严重，单纯形的退化处理（bland/perturbation）代价被放大——本章**没有测退化度**，这只是推测。**真实模型里单纯形往往比这好得多**——它是商用求解器的默认算法不是没有原因。这里只说明一件事：规模上去以后，单纯形的迭代次数 × 单次迭代代价会一起变大。

**5）PDLP 解的精度不是"最优"级别。** 四行的相对偏差在 1e-11 ~ 5e-09（large 那行是 4.8e-09），看着不错，但这是**目标值**的偏差；可行性是另一回事，见下面那段真实日志。而内点法/单纯形给的是 1e-14 量级（xlarge 的 simplex 除外，它压根没解完）。

**6）PDLP 没有基。** 表里最后一列 `basis_validity` 是 HiGHS 的 `HighsInfo.basis_validity`，单纯形和内点法都是 1，PDLP 全是 0。想拿基（做灵敏度分析、给 MIP 热启动）必须额外跑一次 crossover。

### 12.3.4 一段真实的 PDLP 日志

medium 实例上 `solver="pdlp"` 的真实日志（节选）：

```
LP has 1000 rows; 3000 cols; 18000 nonzeros
Presolving model
Presolve reductions: rows 1000(-0); columns 3000(-0); nonzeros 18000(-0) - Not reduced
Problem not reduced by presolve: solving the LP
Solving with cuPDLP-C
Using cost norm =      33.1 and RHS norm =  1.85e+03
--------------------------------------------------
running scaling
- use Ruiz scaling
- use PC scaling
--------------------------------------------------
||x0||_2 = 0.000000e+00
     Iter       Primal.Obj         Dual.Obj        Gap  Primal.Inf  Dual.Inf    Time
        0  +0.00000000e+00  +0.00000000e+00  +0.00e+00    9.99e-01  0.00e+00   0.00s [L]
     4000  +2.83355166e+03  +2.83354761e+03  +7.15e-07    5.59e-07  0.00e+00   0.00s [A]
     8000  +2.83355167e+03  +2.83355137e+03  +5.27e-08    1.26e-07  0.00e+00   0.00s [A]
     9040  +2.83355166e+03  +2.83355143e+03  +4.03e-08    9.90e-08  0.00e+00   0.00s [L]

WARNING: Solution optimality conditions
num/max    352 / 4.69e-05 (relative     81 / 4.24e-07) primal infeasibilities     (tolerance = 1e-07)
...
Model status is "Optimal" since relative violation of tolerances is no more than     4.24
Model status        : Optimal
PDLP      iterations: 9040
```

三个值得注意的点：

- `Solving with cuPDLP-C`：HiGHS 里 `solver="pdlp"` 跑的就是 cuPDLP-C（它另有原生的 HiPDLP，`solver="hipdlp"`，见 12.4）。本机 pip 装的是 CPU 版。
- 迭代区间打印的是 `Primal.Obj / Dual.Obj / Gap / Primal.Inf / Dual.Inf`——一阶方法没有基，所以日志里没有 `Primal.NInf / Dual.NInf` 那种基相关的诊断，只有**目标值差**和**不可行度**。
- `WARNING: LP solver claims optimality, but with ... relative primal infeasibilities (tolerance = 1e-07)`：**PDLP 说收敛了，HiGHS 复核时发现相对不可行度超了 4.24 倍**，最后按"容差内可接受"放行为 Optimal。这不是 bug，而是一阶方法的常态：它的终止判据是"残差小于 tol"，不是"满足 KKT 到机器精度"。**如果下游要用这个解做敏感性分析或二次规划的热启动，这个差别会咬人。**

### 12.3.5 精度旋钮：把 0.34 s 变成 0.016 s

`python code/ch12_gpu/bench_pdlp_highs.py --sweep-only`（medium 实例，3 次取最小）：

| 设置 | 迭代次数 | 耗时 | 与 ipm 参考解相对偏差 | 状态 |
| :--- | ---: | ---: | ---: | :--- |
| `kkt_tolerance=1e-7`（默认）+ `pdlp_optimality_tolerance=1e-4` | 5 360 | 0.205 s | 3.5e-05 | **未证最优** |
| 同上 + `pdlp_optimality_tolerance=1e-5` | 9 040 | 0.345 s | 4.1e-11 | 最优 |
| 同上 + `pdlp_optimality_tolerance=1e-6` | 9 040 | 0.341 s | 4.1e-11 | 最优 |
| 同上 + `pdlp_optimality_tolerance=1e-7`（默认） | 9 040 | 0.339 s | 4.1e-11 | 最优 |
| 同上 + `pdlp_optimality_tolerance=1e-8` | 12 120 | 0.453 s | 5.5e-10 | 最优 |
| `pdlp_optimality_tolerance` 默认 + `kkt_tolerance=1e-4` | **280** | **0.016 s** | 8.7e-06 | 最优 |
| 同上 + `kkt_tolerance=1e-5` | 920 | 0.040 s | 1.2e-06 | 最优 |
| 同上 + `kkt_tolerance=1e-6` | 3 600 | 0.138 s | 4.4e-08 | 最优 |

两件事在这里暴露得很清楚：

1. **只调 `pdlp_optimality_tolerance` 几乎没用**：1e-5、1e-6、1e-7 三个设置的迭代次数**都是 9 040**，一模一样。因为 HiGHS 判定"算不算最优"用的是 `kkt_tolerance`（默认 1e-7），PDLP 自己说"我到了"也没用。真正管用的总闸是 `kkt_tolerance`——放宽到 1e-4，迭代降到 280 次，**快了约 21 倍**，目标值只差 8.7e-06。
2. **容差放过头会丢状态**：`pdlp_optimality_tolerance=1e-4` 且 `kkt_tolerance` 保持默认时，解的目标值差 3.5e-05，状态是 **kUnknown（未证最优）**。这不是失败，是"我不保证"。下游代码如果只判断 `status == Optimal`，就会在这里掉进 else 分支。

HiGHS 官方对这件事的建议（GPU 文档原文）：

> If you use the HiGHS PDLP solver, in the first instance it is recommended that you increase the feasibility and optimality tolerances to `1e-4`, since this will result in the algorithm terminating much sooner. There are multiple feasibility and optimality tolerances, but all will be set to the value of `kkt_tolerance` (if it differs from its default value of `1e-4`).

> 读这段原文时注意一个坑：它说 `kkt_tolerance` 的"默认值"是 `1e-4`，但**本机 HiGHS 1.16.0 实测默认是 `1e-7`**（`getOptionValue("kkt_tolerance")` 读出来就是 1e-7，本程序也是靠"默认 1e-7"才复现出 9 040 次迭代的）。文档这句要么是旧版残留，要么是笔误——按你手上那个版本 `getOptionValue` 读到的默认值为准，别照抄这句话。

### 12.3.6 COPT 的 CPU PDLP：同一道题、另一个求解器

`python code/ch12_gpu/pdlp_copt.py`（medium 实例，走 MPS 文件，COPT 8.0.1 自带受限授权）：

| 路线 | 迭代次数 | 耗时 | 与单纯形参考解相对偏差 |
| :--- | ---: | ---: | ---: |
| `LpMethod=1`（对偶单纯形） | — | 0.777 s | 0 |
| `LpMethod=6` + `GPUMode=0`（PDLP，CPU） | 4 240 | 0.160 s | 1.4e-15 |

同一台机器、同一个模型：COPT 的 PDLP 比它自己的对偶单纯形快约 4.9 倍（nnz=18 000，和 HiGHS 的 0.134 s / 0.338 s 是同一量级）。**这个倍数不稳定**：PDLP 侧每次都是 0.16 s 上下，单纯形侧的 wall clock 在 0.78~1.6 s 之间抖（倍数就变成 5~10 倍），以你自己的重跑为准。注意这里 PDLP 的偏差是 1.4e-15，因为 **COPT 的 PDLP 默认跑完会做 crossover**（日志里的 `Experimental: using crossover to find a basic solution after PDLP`），拿回了一个真正的基。想要"纯一阶"的解就把 `Crossover` 设 0，代价是没有基、精度只能到 `PDLPTol`（默认 1e-6）那个级别。

这两家的数字口径不完全一样（预处理、默认容差、是否 crossover 都不同），**不要横向比较"谁的 PDLP 更强"**，只用来确认一件事：CPU 上 PDLP 的收益在这个规模就已经能被测出来。

## 12.4 GPU 路线现状

这一节每个 API/参数名都有出处。凡是标 **⚠️ 本机未实跑** 的，都是"代码给了、本机跑不了"。

| 求解器 | 一阶方法开关 | GPU 开关 | GPU 前提 | 本机情况 |
| :--- | :--- | :--- | :--- | :--- |
| Gurobi 14 | `Method=6`（`GRB.METHOD_PDHG`） | `PDHGGPU=1` | 需**单独下载 GPU 版**；Compute Server / Instant Cloud 上不可用 | 参数名已用 `_paramdetails.py` 核实（gurobipy 14.0.0；本机 `/usr/local/bin/python3` 的 11.0.1 里这些参数**根本不存在**）；⚠️ 未实跑（无 GPU，且许可证已过期） |
| COPT 8 | `LpMethod=6` | `GPUMode=1`（`GPUDevice` 选卡） | Windows / Linux + CUDA；**macOS 只有 CPU 版** | CPU PDLP 已实跑；⚠️ GPU 未实跑 |
| HiGHS 1.16 | `solver="pdlp"`（内部是 cuPDLP-C；另有 `hipdlp`） | 编译期决定：`-DCUPDLP_GPU=ON` / `-DHIPDLP_HIP=ON` | 本地 CMake 编译 + CUDA（或 ROCm） | CPU 已实跑（pip wheel 是 CPU 版）；⚠️ GPU 未实跑 |
| cuPDLP-C | 它本身就是一阶求解器 | `-DBUILD_CUDA=ON`（上游 CMake）/ `MODE=gpu bash build.sh`（COIN-OR 版） | C/C++ 编译器 + CMake ≥ 3.24 + zlib + Python 3 + NumPy；CUDA Toolkit 只有 GPU 模式才要。HiGHS：**COIN-OR 版内置 1.6.0 自动构建；上游 v0.4.x 必须先 `export HIGHS_HOME=/path-to-highs`**（见下方 cuPDLP-C 小节） | ⚠️ 未实跑 |
| OR-Tools PDLP | 内置 | CPU | — | 未实跑（不是本章重点） |

### Gurobi：PDHG + `PDHGGPU`

Gurobi 把一阶方法叫 **PDHG**（不是 PDLP），开关是 `Method=6`。用 `gurobipy` 自带的参数表（`python code/ch12_gpu/pdlp_gurobi.py --check-params`，不需要许可证）可以逐条验证。下面是**节选并合并了若干行**——脚本的真实输出是一行一个参数（`Method` 的 8 个取值会占 8 行）、还带完整描述，想看原样就跑那条命令：

```
Method  type=int default=-1 range=[-1, 7]
    0=primal simplex   1=dual simplex   2=barrier   6=PDHG (Primal-Dual Hybrid Gradient)
PDHGGPU  type=int default=-1 range=[-1, 1]
PDHGAbsTol   type=double default=1e-06 range=[1e-09, 0.01]
PDHGRelTol   type=double default=1e-06 range=[0, 1e+100]
PDHGConvTol  type=double default=1e-06 range=[0.0, 1.0]
PDHGIterLimit  type=double default=1e+100
PDHGEarlyCrossover  type=double default=2.0 range=[0.0, 16.0]
PDHGMIPHeurTime / PDHGMIPHeurWork / PDHGMIPHeurSolutions
```

> 顺手勘个误：`Method` 不止 0~4。`5` 是已废弃的"确定性并发单纯形"，`6` 是 PDHG，`7` 是"非线性 barrier"（`GRB.METHOD_NLBARRIER`）。`gurobipy` 里对应常量是 `GRB.METHOD_PRIMAL / METHOD_DUAL / METHOD_BARRIER / METHOD_CONCURRENT / METHOD_DETERMINISTIC_CONCURRENT / METHOD_DETERMINISTIC_CONCURRENT_SIMPLEX / METHOD_PDHG / METHOD_NLBARRIER`。

官方对 `PDHGGPU` 的定义（[参数页](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html) 原文）：

> You can set this parameter to 1 to have PDHG run on the GPU, if your Gurobi build has GPU support and your system has compatible hardware. Setting it to 0 forces PDHG to run on the CPU instead. The default setting (-1) is equivalent to 0 in most cases, except if you run on a Compute Server worker that has GPUs. In this case, the default is equivalent to 1.
>
> Note that you must additionally set the Method parameter to `GRB_METHOD_PDHG` (6) or one of the `PDHGMIPHeurTime`, `PDHGMIPHeurWork` or `PDHGMIPHeurSolutions` to a non-default value to enable PDHG.
>
> Note: Not available on Compute Server or Instant Cloud. Note: GPU support requires downloading a separate build.

三个信息量很大的细节：

- **`PDHGGPU=1` 单独设是没用的**，必须同时把 `Method` 设成 6（或者设那三个 MIP 启发式参数之一）——PDHG 才会被启用。
- **默认 `-1` 不是"自动检测"**，而是"多数情况等于 0（CPU）"。只有 Compute Server 的 worker 带 GPU 时才等于 1。
- **GPU 版是单独的安装包**，不是同一个 pip 包里加个开关。

`code/ch12_gpu/pdlp_gurobi.py` 里的写法（⚠️ 本机未实跑）：

```python
model = gp.read("code/ch12_gpu/output/medium.mps")   # 先跑 bench_pdlp_highs.py 生成
model.Params.Method = gp.GRB.METHOD_PDHG     # 6，这一行不能少
model.Params.PDHGGPU = 1                     # 1=GPU, 0=强制 CPU
model.Params.PDHGEarlyCrossover = 2.0        # 提前开 crossover，尽早拿基
model.optimize()
```

`PDHGEarlyCrossover` 值得单独说：PDHG 迭代到"放宽 10^X 倍的容差"时就先把 crossover 线程开起来（默认 X=2，即 1e-4 精度就开跑），让"求一阶解"和"还原基"并行。设 0 表示关掉并发 crossover。这是 Gurobi 对"一阶方法没有基"的工程回答。

### COPT：`LpMethod=6` + `GPUMode`

COPT 的参数名核实成本很低：**对不存在的参数名，`setParam` 会直接抛 `AttributeError: Invalid parameter name`**，所以能设成功就说明名字存在。但"能设"不等于"生效"——**越界的值会被静默夹到区间端点**，只往日志里吐一句 `(MAX)`：

```
setParam("LpMethod", 100)  -> getParam("LpMethod") == 6       # 夹到 MAX
setParam("GPUMode", 3)     -> getParam("GPUMode") == 2        # 夹到 MAX
setParam("PDLPTol", 1)     -> getParam("PDLPTol") == 1e-4     # 夹到 MAX
setParam("PDLPTol", 1e-15) -> getParam("PDLPTol") == 1e-12    # 夹到 MIN
```

所以复验一个参数的正确姿势是：**`setParam` 不报错，然后 `getParam` 读回来确实是你想要的那个值**，两步都过才算设上了。本章正文里出现的每个参数名都过了这两步（清单见 12.4 末尾），只有 `ConcurrentLpMode` 一个名字过不了。

```python
model.setParam("LpMethod", 6)    # 6 = First-order method (PDLP)
model.setParam("GPUMode", 1)     # 1 = 尝试用 GPU；0 强制 CPU
model.setParam("GPUDevice", 0)   # 多卡时选卡（单设它不会打开 GPU）
```

取值定义来自 [COPT 官方参数文档](https://guide.coap.online/copt/en-doc/parameter.html)：

- `LpMethod`：`-1` 自动、`1` 对偶单纯形、`2` barrier、`3` crossover、`4` 并发、`5` 单纯形/barrier 自动选、**`6` 一阶方法（PDLP）**。（本机实测 `setParam("LpMethod", 0)` 也能设上并保留 0，但官方表里没有 `0` 这个取值——名字存在不代表它有文档，别用。）
- `GPUMode`：官方原文

  > -1: Choose automatically. The first-order method (PDLP) will attempt to use the GPU by default, while the barrier method will use the CPU by default.
  >
  > 0: Force CPU mode.
  >
  > 1: Attempt to use the standard GPU mode.
  >
  > 2: For the barrier method, attempt to use the high-performance GPU mode when solving LP problems, which may lead to higher memory usage. For the first-order method (PDLP), this is equivalent to GPUMode=1 (standard GPU mode).

  注意 `-1` 的官方语义是"自动"，而对 PDLP 是 **默认就尝试用 GPU**（barrier 才默认 CPU）。本机之所以看到 `-1` 和 `0` 的日志一模一样，是因为 macOS 上根本没有 GPU 版可走——**这是平台短路，不是 `-1` 的语义**。想确定性地强制 CPU 必须显式写 `GPUMode=0`。**这四个取值本机能设上（`getParam` 复核一致），但本机没有 GPU，跑起来一样是 CPU——取值语义来自官方文档，不是本机验出来的。**
- 适用范围：官方明说 COPT 的 GPU 求解器支持 **LP 和 MILP 的根松弛**，且只有 PDLP 或 barrier 两种方法能用 GPU。

还有第四个参数在官方文档里有、**本机 8.0.1 里没有**，所以正文不把它当可用写法（详见 12.4 末尾的"待确认"）：`ConcurrentLpMode`，只在 `LpMethod=4`（并发）时生效，作用是把"GPU 上的 PDLP / barrier"当成并发求解器里的一个参赛选手。官方原文与取值见那张表，**别照抄代码**——`setParam` 会直接抛 `Invalid parameter name`。

macOS 的情况官方也写得明明白白（原文）：

> On platforms such as macOS, only the CPU version of COPT is currently available. Even if `GPUMode=1` or `2` is set, COPT will still run in CPU mode.

本机实测印证了这句话，而且顺手暴露了一个坑：

| 设置 | 日志 | 结论 |
| :--- | :--- | :--- |
| `LpMethod=6`，只设 `GPUDevice=1`（手工 `setParam` 探针，脚本里没有这个开关） | `Starting PDLP solver using CPU` | **设了设备号也不会触发 GPU，且没有任何提示** |
| `LpMethod=6` + `GPUMode=1` | `GPU mode is not supported` → `Starting PDLP solver using CPU` | 至少告诉你"不支持" |

所以**验证"到底跑没跑 GPU"的唯一可靠办法是看启动日志**，不要看参数设成什么。这一段属于"本机实跑"（跑的是 CPU 退化路径），GPU 路径 ⚠️ 未实跑。

### HiGHS：`solver="pdlp"` 背后就是 cuPDLP-C

HiGHS 自己的一阶求解器叫 HiPDLP，同时从 1.10 起把 **cuPDLP-C** 也接了进来。本机日志里 `Solving with cuPDLP-C` 就是这个。官方 GPU 文档（[docs/src/guide/gpu.md](https://github.com/ERGO-Code/HiGHS/blob/latest/docs/src/guide/gpu.md)）的关键三句：

> Whether HiPDLP (and cuPDLP-C) runs on the CPU or on a GPU is fixed at **build time**: the GPU backend is only compiled with `-DHIPDLP_HIP=ON` (AMD, HiPDLP only) or `-DCUPDLP_GPU=ON` (NVIDIA, HiPDLP and cuPDLP-C). The runtime `solver` option selects the *solver*, not the *device*: on a build without GPU support, `solver = "hipdlp"` still runs, but on the CPU.

也就是说：

- **pip 装下来的 wheel 是 CPU 版**，`solver="pdlp"` 改不了这一点（本机 wheel 里 `strings | grep -ci cuda` 结果是 0，日志也确认是 CPU）；
- 想要 GPU，得本地 CMake 编译，Linux / Windows + CUDA Toolkit（或 AMD 的 ROCm）；
- 反过来的坑也在同一段文档里：**GPU 版编进去以后 HiPDLP 不能退回 CPU**——原文 `Conversely, on a GPU build, HiPDLP cannot fall back to the CPU: if no GPU is available at run time, the solve returns with an error.` 也就是说"编了 GPU 版但没有卡"不是静默变慢，是直接报错。
- 还有一个选项名字很像 GPU 开关：`pdlp_cupdlpc_restart_method`（源码 `HighsOptions.h` 原文 `Restart mode for cuPDLP-C solver: 0 => none; 1 => GPU (default); 2 => CPU`，默认 1）。按取值名字推断，它管的是**重启那一步算在哪个设备上**（原文只给了 0/1/2 三个名字，没写解释），不是"整个求解跑在哪"。别拿它当加速开关——本机默认值就是 1，跑的依然是 CPU。

#### 怎么一眼分出跑的是 HiPDLP 还是 cuPDLP-C

这两个都是"HiGHS 里的一阶方法"，但完全不是一套代码：`solver="pdlp"` 走 **cuPDLP-C**，`solver="hipdlp"` 走 HiGHS 自己写的 **HiPDLP**。**别猜，看启动那一行**（本机 HiGHS 1.16.0 / pip wheel / CPU，同一道 small 实例实跑）：

`solver="pdlp"`：

```
Problem not reduced by presolve: solving the LP
Solving with cuPDLP-C
Using cost norm =      12.2 and RHS norm =       375
--------------------------------------------------
running scaling
- use Ruiz scaling
- use PC scaling
--------------------------------------------------
     Iter       Primal.Obj         Dual.Obj        Gap  Primal.Inf  Dual.Inf    Time
```

`solver="hipdlp"`：

```
Problem not reduced by presolve: solving the LP
Using HiPDLP first order PDLP solver on a CPU: performance may be disappointing!
HiPDLP uses Halpern restart only; ignoring pdlp_restart_strategy and restart-off feature flag.
HiPDLP only supports pdlp_step_size_strategy in {0: fixed, 3: PID}; using PID for unsupported values.
     Iter       primal obj         dual obj       pinf       dinf       gap     pr wt    time
```

判据就三条，都不用猜：

| 看哪里 | cuPDLP-C（`pdlp`） | HiPDLP（`hipdlp`） |
| :--- | :--- | :--- |
| 启动行 | `Solving with cuPDLP-C` | `Using HiPDLP first order PDLP solver on a CPU`（GPU 版这一行会不一样，具体写什么 ⚠️ 未实跑，别照这句话去 grep GPU build 的日志） |
| 迭代表头 | `Primal.Obj / Dual.Obj / Gap / Primal.Inf / Dual.Inf` | `primal obj / dual obj / pinf / dinf / gap / pr wt`（多一列 `pr wt`，即原始权重） |
| 参数生效情况 | 吃 `pdlp_restart_strategy`、`pdlp_step_size_strategy` 全部取值 | 只做 Halpern 重启、**忽略** `pdlp_restart_strategy`；步长只支持 `0: fixed` 和 `3: PID`，其余值会被强制成 PID |

同一道 small 实例：cuPDLP-C 迭代 2 520 次拿到最优，HiPDLP 迭代 3 680 次、目标值相对 cuPDLP-C 差 3.5e-09（相对单纯形参考解是 2.1e-09），两者 `basis_validity` 都是 0。**所以本章表里 `pdlp` 那一列是 cuPDLP-C 的数，不是 HiPDLP 的数**——想复现 HiPDLP 的数字得自己把 `solver` 换成 `"hipdlp"`。

⚠️ HiGHS 的 GPU 分支**本机未实跑**（无 NVIDIA GPU，也没有本地编译的 CUDA 版）。

### cuPDLP-C：NVIDIA 侧的开源实现

[COPT-Public/cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C)（MIT 协议）是 PDLP 的 CUDA C/C++ 实现，论文 [arXiv:2312.14832](https://arxiv.org/abs/2312.14832) 的 GPU 实验用的是 NVIDIA H100 80GB。它没有 Python 包，只有一个命令行程序 `plc`，吃 MPS 文件。

**编译前提只按官方 README / INSTALL 的说法写，不自己编版本号**（官方原文，见仓库 `README.md` 与 `INSTALL`）：

> Prerequisites: C/C++ compiler, CMake >=3.24, zlib development headers, Python 3 and NumPy. **Bundled HiGHS 1.6.0 is built automatically.** CUDA Toolkit and an NVIDIA GPU are needed only for GPU mode.

两点值得停一下：

- 那个 **HiGHS 1.6.0 是 cuPDLP-C 自己内置（bundled）的**，跟着它一起自动编译——不是"你要先去装一个 1.6.0"。它跟你本机 pip 装的 HiGHS 1.16 是两个不相干的库。官方原文：`Bundled HiGHS is built/installed automatically unless HIGHS_HOME is supplied.`
- ⚠️ **两份分发的要求不一样**，别混：上面那句"内置自动构建"是 **COIN-OR 打包版**（`MODE=gpu bash build.sh` 那条路）的行为；**上游 `v0.4.x` 的 README 要求你先自己装好 HiGHS 并导出 `export HIGHS_HOME=/path-to-highs`**（它的 `FindHiGHSConf.cmake` 直接读这个环境变量）。照上游的命令抄却没设 `HIGHS_HOME`，configure 会失败。
- **CUDA 没有官方最低版本号**，只要求"CUDA Toolkit + 匹配的 NVIDIA 驱动"。仓库 `VALIDATION.md` 记录的实际实测配置是 `NVIDIA RTX 4090, driver 580.82.07, CUDA Toolkit 12.8.93, architecture 89`——这是"验证过的版本"，不是"要求的版本"。

上游 CMake 直接编（老 README，`plc` 这个 target 就是这条路；`code/ch12_gpu/pdlp_cupdlp.py` 打印的也是这套）：

```bash
export CUDA_HOME=/path-to-cuda       # GPU 模式才需要
mkdir -p build && cd build
cmake -DBUILD_CUDA=ON ..             # 默认 OFF，不加这个开关编出来的是 CPU 版
cmake --build . --target plc
./bin/plc -fname model.mps -nIterLim 5000 -dTimeLim 600 -savesol true -outSol sol.json
```

> 注意输出参数名是 **`-outSol`**（解，默认 `./solution.json`），另有一个 `-out`（汇总，默认 `./solution-sum.json`）。写成 `-fout` 会不认识。

2026-10 起仓库换成了 COIN-OR onboarding 版，官方给出的编译入口变成 `build.sh`（内置 HiGHS 也走这条）：

```bash
export CUDA_HOME=/usr/local/cuda-12.8   # 换成你自己的 CUDA Toolkit 路径
export CUDA_VISIBLE_DEVICES=0
export GPU_ARCH=89                      # 默认 native；RTX 4090 用 89，H100 用 90
MODE=gpu bash build.sh                  # 默认是 cpu；也可 MODE=all
# 产物在 .coinor-deps/cupdlp-gpu-install/bin/plc
```

⚠️ 本机未实跑（macOS 上无法安装 CUDA）。`code/ch12_gpu/pdlp_cupdlp.py` 会打印上面两套命令和参数表，并用 `gen_lp.py` 生成它需要的 MPS 文件；有 `plc` 时 `--run` 才会真的执行。两条路都只是"照官方文档给出的命令"，没有在本机验过，真正跑之前请以你 clone 到的那个版本的 README / INSTALL 为准。

### 参数名复验清单（本机实跑）

这一章正文里出现的**每一个参数名**，都按下面的方法在本机过了一遍。方法是固定的，你自己也能重跑：

```python
import coptpy as cp
env = cp.Envr(); m = env.createModel("t"); m.setParam("Logging", 0)
m.setParam("<参数名>", 1)      # 报 AttributeError: Invalid parameter name 就是名字不存在
m.getParam("<参数名>")          # 再读回来：会不会被夹到区间端点，只有这一步能看出来
```

**COPT 8.0.1**（`✓` = `setParam` 不报错；`getParam` 一列是设 1 之后读回来的值）

| 参数名 | `setParam` | `getParam` | 官方参数表 |
| :--- | :---: | ---: | :--- |
| `LpMethod` | ✓ | 6（设 100 被夹到 MAX=6） | 有：`-1`、`1`~`6` |
| `GPUMode` | ✓ | 1（设 3 被夹到 MAX=2） | 有：`-1`/`0`/`1`/`2` |
| `GPUDevice` | ✓ | 1 | 有 |
| `Crossover` | ✓ | 1 | 有：`-1`/`0`/`1` |
| `PDLPTol` | ✓ | 1e-4（设 1 被夹到 MAX=1e-4） | 有：默认 1e-6，区间 1e-12 ~ 1e-4 |
| `PDLPPrimalTol` | ✓ | 1e-4 | **没有**（表里只有 `PDLPTol`） |
| `PDLPDualTol` | ✓ | 1e-4 | **没有** |
| `PDLPIterLimit` | ✓ | 1 | **没有**（Limits 表里没列，但名字能用） |
| `ConcurrentLpMode` | ✗ `Invalid parameter name` | — | 有（见"待确认"） |
| `ConcurrentMode` / `LpConcurrentMode` / `Concurrent` / `ConcurrentMethod` | ✗ | — | 没有 |
| `PDLPIter` | ✗ | — | 没有（`PDLPIterLimit` 才是对的那个） |

**HiGHS 1.16.0**（用 `highspy` 的 `setOptionValue` / `getOptionValue`）

| 选项名 | `setOptionValue` | 默认值 | 备注 |
| :--- | :---: | ---: | :--- |
| `solver` | ✓ | `choose` | `pdlp` / `hipdlp` / `ipm` / `simplex` 都接受 |
| `kkt_tolerance` | ✓ | **1e-7** | 官方 GPU 文档写"默认 1e-4"，与 1.16.0 实测不符 |
| `pdlp_optimality_tolerance` | ✓ | 1e-7 | — |
| `pdlp_cupdlpc_restart_method` | ✓ | 1 | 合法值 0/1/2（设 3 被夹回 1） |
| `pdlp_step_size_strategy` | ✓ | 1（自适应） | `0` 固定 / `1` 自适应 / `2` Malitsky-Pock / `3` PID |
| `pdlp_restart_strategy` | ✓ | 2 | 只是 HiPDLP 会忽略它（见上文） |
| `crossover` | ✗ | — | 报 `getOptionIndex: Option "crossover" is unknown` |
| `run_crossover` | ✓ | `on` | 存在，但对 PDLP 不产生基（`basis_validity` 仍为 0） |

**Gurobi 14.0.0**（用 `gurobipy/_paramdetails.py` 的参数表逐字打印，不需要许可证）：`Method`、`PDHGGPU`、`PDHGAbsTol`、`PDHGRelTol`、`PDHGConvTol`、`PDHGIterLimit`、`PDHGEarlyCrossover`、`PDHGMIPHeurTime`、`PDHGMIPHeurWork`、`PDHGMIPHeurSolutions` 全部存在，默认值/区间与 12.4 的节选一致。

### 待确认的项：查得到出处、但本机验不了

本章的纪律是"参数名必须能设上、数字必须能复现"。下面几项只做到一半——有官方出处，但本机没法证明它在这台机器上真的成立，单独列出来，读的时候打折。

| 项 | 官方怎么说 | 本机实测（COPT 8.0.1 / HiGHS 1.16.0） | 结论 |
| :--- | :--- | :--- | :--- |
| COPT `ConcurrentLpMode` | 参数文档 Ver 8.0 有完整定义：`-1` 自动、`0` 只 CPU（simplex + barrier）、`1` CPU + **GPU PDLP**、`2` CPU + GPU barrier、`3` 全上；`1`/`2` 用 0 号卡，`3` 用 0 和 1 两张卡，卡不够会**报错退出**。原文还有一句关键话：`LpMethod=4` 时 `GPUMode` 和 `GPUDevice` 会被**忽略**，设备和 GPU 用法完全由它决定 | `setParam("ConcurrentLpMode", 1)` → `AttributeError: Invalid parameter name`；`coptcore.so` 里 `strings \| grep -i concur` 也是 0 条 | **文档有、8.0.1 本机不存在**，可能是更新版本或企业版。**不要照抄**；真要用，先在本机 `setParam` 试一下 |
| COPT `PDLPPrimalTol` / `PDLPDualTol` | 官方容差表里只有 `PDLPTol`（默认 1e-6，区间 1e-12 ~ 1e-4） | 两个名字都能设上，`getParam` 也读得回 | 名字存在但**没有官方语义**，别依赖；调精度只认 `PDLPTol` |
| COPT `GPUMode=2` 对 PDLP | 原文："For the first-order method (PDLP), this is equivalent to GPUMode=1 (standard GPU mode)." | 能设上，但本机无 GPU，分不出 1 和 2 | 语义采信官方，未实测 |
| HiGHS `pdlp_cupdlpc_restart_method` | 源码 `HighsOptions.h` 原文：`Restart mode for cuPDLP-C solver: 0 => none; 1 => GPU (default); 2 => CPU` | 默认 1；设 0/1/2 都行，设 3 被夹回 1 | 只在**已经编进 GPU 后端**的 build 里才有意义；CPU build 上它照样是 1，别当开关 |
| Gurobi `PDHGGPU` 真能跑 GPU | 官方参数页原文见上 | 无 NVIDIA GPU。另外本机有两个 gurobipy：`/opt/homebrew/var/tmp/.venv_opt/bin/python`（3.12.7）里是 14.0.0（参数表有，逐字打印过），`/usr/local/bin/python3`（3.9.12）里是 **11.0.1，连 `GRB.Param.PDHGGPU` 都不存在、`gurobipy._paramdetails` 模块也没有** | 参数语义可信（14.0.0 的参数表逐字打印），**能不能跑 GPU 未实测**；顺带提醒：gurobipy < 12 连 PDHG 都没有 |
| 任何 GPU 加速比 | gpu.md 只说"GPU 只对大 LP 划得来"，不给数字 | 本机无 NVIDIA GPU | 本章不给任何未实测的倍数 |

## 12.5 要不要上 GPU：一份判断清单

按顺序问自己，任何一条答"是"就先去解决问题，而不是先买卡。

**1）模型有多大？** 粗略门槛：本机实测 nnz 一万八时 PDLP 还慢 2.5 倍，九万六时已经反超 2.8 倍——交叉点落在 $2\times10^4 \sim 10^5$ 之间（本章只采了 18k 和 96k 两个点，没细扫）。到 $10^6\sim10^7$ 才是一阶方法的主场。显存估算：CSC 存储每个非零约 `int32 索引 + double 值 = 12 B`，$10^8$ 个非零就是 1.2 GB——还要留出向量和多份副本的空间。

**2）精度要求是多少？** 1e-4 的目标值精度够用（看趋势、做近似决策、给 MIP 提供初始解）→ 适合。要 1e-9、要做灵敏度分析、要给下游当热启动 → 不适合，得 crossover 或者干脆用内点法。

**3）是纯 LP 还是 MIP？** 只有 LP（以及 MIP 的**根松弛**）能吃到 GPU 一阶方法的加速，分支定界的树搜索主体仍在 CPU。Gurobi 的 `PDHGMIPHeur*` 三个参数说的也是"MIP 启发式里用 PDHG 解节点松弛找可行解"，不是"整个 MIP 用 GPU 求"。

**4）能不能接受"没有基"？** 需要 `basis_validity`、需要 dual values 做定价、需要热启动 → 要么开 crossover（Gurobi `PDHGEarlyCrossover`、COPT `Crossover=1`），要么别用一阶方法。

**5）算力成本与许可证。** 云上 GPU 实例的单价、以及求解器许可证是否覆盖 GPU 求解，都要提前确认（Gurobi 的 GPU 版是单独安装包；COPT 的 GPU 模式有独立的授权口径）。

**6）怎么测才不算自欺欺人？** 没有下面这套协议，测出来的"加速比"基本不能信：

- 同一个模型文件、同一台机器、同一组容差；否则比的是参数不是算法；
- 小实例至少跑 3 次取最小值（本章的 small / medium 是这么做的；large / xlarge 单次就要几十秒到两分钟，本章各只跑了 1 次，所以那两行只能看数量级）；不同轮次之间 ±10%~30% 的抖动很正常，`xlarge` 上甚至能差出一个"触时限"；
- **同时记录迭代次数和不可行度**，不能只记时间：PDLP 少迭代两次就"更快"，但解可能是废的；
- 精度要对齐再比时间：拿 PDLP 的 1e-4 解去比内点法的 1e-9 解，等于换了个问题；
- 单卡 vs 单核、GPU vs CPU 要说清楚；本章的 CPU 数字是 8 核机器上的单进程结果。

## 12.6 踩过的坑

**1）`PDHGGPU=1` 不配 `Method=6`，等于没设。** 官方原文写着"you must additionally set the Method parameter to `GRB_METHOD_PDHG` (6)"。不设 `Method`，PDHG 根本不会启动，`PDHGGPU` 就是个死参数、还不报错。

**2）设了 GPU 参数 ≠ 跑了 GPU。**

- COPT：只设 `GPUDevice` 不设 `GPUMode`，日志照样是 `Starting PDLP solver using CPU`，**一句提示都没有**；设 `GPUMode=1` 才会打印 `GPU mode is not supported`；
- HiGHS：设备在**编译期**就定死了，`solver="pdlp"` 在 CPU build 上照样跑 CPU；`pdlp_cupdlpc_restart_method=1` 里的 "GPU" 指的是重启策略的执行设备；
- 通用做法：**看启动日志**，别信参数。

**3）调错容差，精度降不下来。** HiGHS 里 `pdlp_optimality_tolerance` 单独调是没用的（实测 1e-5/1e-6/1e-7 迭代数都是 9 040），总闸是 `kkt_tolerance`。放宽到 1e-4，medium 实例从 9 040 次迭代 / 0.339 s 降到 280 次 / 0.016 s。

**4）容差放过头，状态会变成"未证最优"。** `pdlp_optimality_tolerance=1e-4` 且 `kkt_tolerance` 默认时，HiGHS 返回 `kUnknown`。下游只判断 `kOptimal` 的代码会在这一步走进 else 分支。

**5）`passModel` 失败不抛异常，模型会静默变空。** 写这章时踩到的：给 `HighsLp` 设 `a_matrix_.format_ = kColwise`，却把 csr 矩阵的 `indptr/indices` 塞进去，`passModel` 返回 `kError`，随后所有求解都返回 `kModelEmpty`、目标值 0——**没有任何报错**。本章所有 `passModel` 调用都检查了返回值。这和第 9 章 9.2 节那些"API 语义和你以为的不一样"是同一类问题。

**6）PDLP 说 "Optimal" 不等于解满足 KKT。** 12.3.4 的真实日志里，HiGHS 复核发现相对原始不可行度 4.24e-07，是容差 1e-07 的 4.24 倍，最后按"容差内可接受"放行。要用这个解做敏感分析前，先看 `HighsInfo` 里的 `max_relative_primal_infeasibility` 这些字段。

**7）PDLP 没有基（`basis_validity=0`）。** 想要基就得 crossover：Gurobi 用 `PDHGEarlyCrossover`，COPT 用 `Crossover=1`。HiGHS 这边**没有能让 PDLP 产出基的开关**：名字叫 `crossover` 的选项根本不存在（设了会报 `getOptionIndex: Option "crossover" is unknown`）；确实有个 `run_crossover`（默认 `on`），但它只服务于内点法的基恢复，对 PDLP 无效——本机实测默认 `on` 的情况下 `basis_validity` 依然是 0、`HighsInfo.crossover_iteration_count` 恒为 0。

**8）平台限制是硬的。** macOS 上没有 CUDA：COPT 官方明说 macOS 只有 CPU 版；cuPDLP-C 必须 CUDA；HiGHS 的 GPU 版要本地 CMake + CUDA/ROCm。想在 Mac 上验证 GPU 路线，只能远程到 Linux 机器。

**9）别把"GPU 加速比"当常数。** 它依赖卡型、显存带宽、模型结构、精度要求、以及求解器有没有把整个算法搬上 GPU。本章因此不给任何未实测的倍数——要数字就按 12.5 第 6 条自己测。

## 12.7 完整代码与复现

| 文件 | 内容 | 本机状态 |
| :--- | :--- | :--- |
| [`code/ch12_gpu/gen_lp.py`](../code/ch12_gpu/gen_lp.py) | 生成可复现的稀疏 LP（四个规模），可导出 MPS | ✅ 已实跑 |
| [`code/ch12_gpu/bench_pdlp_highs.py`](../code/ch12_gpu/bench_pdlp_highs.py) | HiGHS 上 simplex / ipm / pdlp 的 CPU 对比 + 精度扫描 | ✅ 已实跑（12.3 的表） |
| [`code/ch12_gpu/pdlp_copt.py`](../code/ch12_gpu/pdlp_copt.py) | COPT `LpMethod=6` + `GPUMode`，日志核实设备 | ✅ CPU 已实跑；⚠️ GPU 未实跑 |
| [`code/ch12_gpu/pdlp_gurobi.py`](../code/ch12_gpu/pdlp_gurobi.py) | Gurobi `Method=6` + `PDHGGPU`；参数表自查 | ✅ 参数表已实跑（需 gurobipy ≥ 12，实测 14.0.0）；⚠️ 求解未实跑（许可证过期） |
| [`code/ch12_gpu/pdlp_cupdlp.py`](../code/ch12_gpu/pdlp_cupdlp.py) | cuPDLP-C 的编译与调用命令、参数表 | ⚠️ 未实跑（需 CUDA） |

复现本章所有数字：

```bash
# 1) 主表（四个规模，约 5~6 分钟；只想看趋势加 --quick）
python code/ch12_gpu/bench_pdlp_highs.py --time-limit 120

# 2) 精度扫描（几秒钟，结果写 output/bench_sweep.tsv）
python code/ch12_gpu/bench_pdlp_highs.py --sweep-only

# 3) COPT 的 CPU PDLP（需要 coptpy + 授权）
python code/ch12_gpu/pdlp_copt.py

# 4) Gurobi 参数表自查（不需要许可证，但需要 gurobipy >= 12）
python code/ch12_gpu/pdlp_gurobi.py --check-params
```

结果写在 `code/ch12_gpu/output/`（`bench.tsv` / `bench_sweep.tsv` / 各家日志）。注意 **`bench.tsv` 每次运行都会覆盖**，只保留最后一次跑的那些规模——想复现 12.3.2 的完整 12 行就一次性跑全量；用 Tab 分隔而不是 CSV，是为了躲开某些终端安全软件对 `.csv` 的加密。该目录已在 `.gitignore` 里（`code/**/output/`），不进版本库——**跑完不要把 output/ 加进 git**，里面有 1 MB 级的 `.mps` 和一堆日志。

## 参考资料

- Applegate, Díaz, Lu, Lubin, Mirrokni, Sun, Teo, Wang, Wang: *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient*, [arXiv:2106.04756](https://arxiv.org/abs/2106.04756)（PDLP 原始论文）
- Lu, Yang, Hu, Huangfu, Liu, Liu, Ye, Zhang, Ge: *cuPDLP-C: A Strengthened Implementation of cuPDLP for Linear Programming by C language*, [arXiv:2312.14832](https://arxiv.org/abs/2312.14832)
- [cuPDLP-C 仓库（COPT-Public/cuPDLP-C）](https://github.com/COPT-Public/cuPDLP-C)（编译前提引自它的 `README.md` / `INSTALL` / `VALIDATION.md`：CMake ≥ 3.24、内置 HiGHS 1.6.0、实测 CUDA 12.8.93 + RTX 4090）
- [HiGHS GPU 加速文档](https://github.com/ERGO-Code/HiGHS/blob/latest/docs/src/guide/gpu.md)（PDLP 警告、编译期决定设备、GPU build 不能回退 CPU、`kkt_tolerance` 建议都出自这里）
- [HiGHS 文档首页](https://ergo-code.github.io/HiGHS/)
- [Gurobi 参数参考](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html)（`Method`、`PDHGGPU`、`PDHG*` 的定义）
- [COPT 参数文档](https://guide.coap.online/copt/en-doc/parameter.html)（`LpMethod`、`GPUMode`、`GPUDevice`、`ConcurrentLpMode`；注意 `ConcurrentLpMode` 在本机 8.0.1 里不存在，见 12.4"待确认"）
