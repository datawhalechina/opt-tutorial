# 优化求解器入门手册

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT%20%2F%20CC%20BY--NC--SA%204.0-green)](LICENSE)
[![solvers](https://img.shields.io/badge/solvers-5%20%28Gurobi%20%C2%B7%20COPT%20%C2%B7%20CPLEX%20%C2%B7%20HiGHS%20%C2%B7%20SCIP%29-orange)](docs/00-环境准备.md)

<!-- CI 通过后可以把下面这行加回来（需先把 .github/workflows/verify.yml 推到默认分支并跑过一次）：
[![verify](https://img.shields.io/github/actions/workflow/status/datawhalechina/opt-tutorial/verify.yml?branch=main&label=verify)](https://github.com/datawhalechina/opt-tutorial/actions/workflows/verify.yml)
-->

OptTutorial 是一个面向初学者的**多求解器**优化建模教程：同一道例题，用 **Gurobi、COPT、CPLEX、HiGHS、SCIP** 各写一遍，再对比它们在建模、求解、日志、文件格式上的差异。

和一般教程的区别只有一条：**每一段代码都真跑过，且跨求解器互相校验。**

- 5 个求解器：Gurobi、COPT、CPLEX（docplex）、HiGHS、SCIP（后两个开源免费）
- 7 道例题：线性规划、0-1 背包、设施选址、半连续生产、文件读写、日志解读、车辆路径
- 外加建模层 PuLP / Pyomo：模型写一次，后端随便换
- 一键校验：`python scripts/verify_all.py` 把所有能跑的求解器跑一遍，**比对最优目标值是否一致**

```bash
pip install -r requirements.txt
python scripts/verify_all.py
```

```
汇总（✓ 通过 / - 跳过 / ✗ 失败）
例题               COPT  CPLEX  Gurobi  HiGHS  PuLP  Pyomo  SCIP
lp_profit          ✓     -      -       ✓      ·     ·      ✓
knapsack           ✓     -      -       ✓      ·     ·      ✓
facility           ✓     -      -       ✓      ·     ·      ✓
全部通过 ✅
```

没有 license 也能跑通——HiGHS 和 SCIP 免授权，`-` 表示本机没装或没授权，不算失败。

## 目录

**入门**

| 章节 | 内容 |
| :--- | :--- |
| [第 0 章 环境准备](docs/00-环境准备.md) | 五个求解器的安装、授权、版本矩阵、自检 |
| [第 1 章 线性规划](docs/01-线性规划.md) | 五种写法并排，建立"换求解器只换 API"的直觉 |
| [第 2 章 整数规划](docs/02-整数规划.md) | 0-1 背包，binary / integer 变量的不同写法 |
| [第 3 章 混合整数规划](docs/03-混合整数规划.md) | 设施选址，big-M 约束怎么取 |

**进阶**

| 章节 | 内容 |
| :--- | :--- |
| [第 4 章 常见建模技巧](docs/04-常见建模技巧.md) | 半连续、指示约束 vs big-M、逻辑约束、对称性破除、数值稳定性 |
| [第 5 章 文件格式](docs/05-文件格式.md) | MPS / LP / MST / SOL / ILP 各自用途，IIS 排查 |
| [第 6 章 日志解读与调参](docs/06-日志解读与调参.md) | 读懂 LP / MIP 日志，Gap 公式，参数速查 |
| [第 7 章 车辆路径问题](docs/07-车辆路径问题.md) | CVRP + MTZ 子回路消除，四个求解器实测 |
| [第 8 章 建模层 PuLP 与 Pyomo](docs/08-建模层.md) | 模型写一次，后端一行切换 |
| [第 9 章 常见错误与排查](docs/09-常见错误与排查.md) | 参数位置、已废弃 API、格式误解、状态码语义 |

**推荐路径**：0 → 1 → 2 → 3 打基础；要落地再看 4、5、6；做路径/调度类问题看 7；多求解器协作看 8；出 bug 先翻 9。

## 代码结构

```
code/
├── common/
│   ├── data.py              # 所有例题的数据，正文里的数字都来自这里
│   └── verify.py            # 统一打印与标准答案比对
├── ch01_lp/                 # 线性规划：profit_{gurobi,copt,cplex,highs,scip}.py
├── ch02_knapsack/           # 0-1 背包：knapsack_{...}.py
├── ch03_milp/               # 设施选址：facility_{...}.py
├── ch04_techniques/         # 半连续产量：scheduling_{...}.py
├── ch05_formats/            # 文件读写：formats_{gurobi,copt,highs}.py
├── ch06_logging/            # 日志生成：log_{gurobi,copt,highs}.py
├── ch07_cvrp/               # 车辆路径：cvrp_{gurobi,copt,highs,scip}.py
└── ch08_modeling/           # 建模层：pulp_{lp,milp}.py、pyomo_{lp,milp}.py
scripts/verify_all.py        # 一键校验
```

每个脚本都能单独运行：

```bash
python code/ch01_lp/profit_gurobi.py
python code/ch07_cvrp/cvrp_copt.py
```

## 版本与验证

教程里的代码在下列版本上实际运行过（详见[第 0 章](docs/00-环境准备.md)）：

| 求解器 | 验证版本 | 本仓库是否实跑 |
| :--- | :--- | :--- |
| Gurobi | 14.0 | API 按官方 type stub 逐条核对；求解需 license |
| COPT | 8.0.1 | ✅ 实跑 |
| CPLEX / docplex | docplex 2.32 | 建模语法与 LP 导出已核对；求解需 CPLEX 运行时 |
| HiGHS | 1.16.0 | ✅ 实跑 |
| SCIP | 6.3.0 | ✅ 实跑 |
| PuLP | 3.3.1 | ✅ 实跑（CBC / HiGHS / COPT 三个后端） |
| Pyomo | 6.10.1 | ✅ 实跑（appsi_highs 后端） |

求解器 API 变化很快（本仓库就因此修过一批老代码，见[第 9 章](docs/09-常见错误与排查.md)）。遇到行为不一致，先跑 `scripts/verify_all.py`。

## 官方文档

写代码时以官方文档为准，本仓库所有 API 都对照过：

**求解器**

- Gurobi：[优化器文档](https://docs.gurobi.com/projects/optimizer/en/current/index.html) · [官方示例](https://docs.gurobi.com/projects/examples/en/current/)
- COPT：[官方文档](https://www.shanshu.ai/copt-document/) · [Python 指南](https://guide.coap.online/copt/zh-doc/) · [PyPI](https://pypi.org/project/coptpy/)
- CPLEX：[产品文档](https://www.ibm.com/docs/en/icos) · [docplex 文档](https://ibmdecisionoptimization.github.io/docplex-doc/)
- HiGHS：[文档](https://ergo-code.github.io/HiGHS/) · [Python 接口](https://ergo-code.github.io/HiGHS/dev/interfaces/python/)
- SCIP：[文档](https://scipopt.org/doc/html/) · [PySCIPOpt](https://pyscipopt.readthedocs.io/)

**建模层**

- PuLP：[文档](https://coin-or.github.io/pulp/)
- Pyomo：[文档](https://pyomo.readthedocs.io/en/stable/)

## 贡献

欢迎补充新的求解器（比如 Xpress、GLPK、OR-Tools）或新的例题。加一个求解器只要两步：

1. 在对应 `code/chXX_*/` 下照着现有文件名加一个 `<问题>_<求解器>.py`，暴露 `solve()` 返回 `(目标值, 解字典)`
2. 在 `scripts/verify_all.py` 的 `CASES` 里登记

然后跑一次校验，**目标值必须和其他求解器一致**再提 PR。

```bash
python scripts/verify_all.py          # 全部
python scripts/verify_all.py cvrp     # 只跑某一类
python scripts/verify_all.py -v       # 失败时打印完整堆栈
```

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :--- | :--- | :--- |
| [邓宇文](https://github.com/GKDGKD) | 项目负责人 | Datawhale 成员、杉数科技算法工程师 |

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width="180" height="180">
</div>

## LICENSE

- **代码**（`code/`、`scripts/`）：[MIT](LICENSE)
- **文档**（`docs/`、`README.md`）：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
