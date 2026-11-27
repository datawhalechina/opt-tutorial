"""第十一章 根松弛与回调 —— 用 COPT 求解，并在回调里取内部信息。

习题：第 7 章的 CVRP（1 个配送中心 + 5 个客户，MTZ 写法）。做三件事：
  1. 单独建 LP 松弛模型（0-1 变量放开成 [0,1] 连续）求一次 → 根松弛界
  2. 建 MIP，用回调在 MIPRELAX / MIPSOL 里取松弛解与整数解
  3. 和整数最优对比，算根节点 Gap

COPT 的回调要点：
  - 回调对象必须继承 coptpy.CallbackBase，覆写 callback(self)
  - 用 model.setCallback(cb, COPT.CBCONTEXT_*) 注册，context 是位掩码
  - self.where() 是**方法**，返回 context 编号（1=MIPRELAX, 2=MIPSOL, 4=MIPNODE, 8=INCUMBENT）
  - self.getRelaxSol() 返回一个 **list**（按变量创建顺序），不是 MVar
运行：python code/ch11_callback/rootrelax_copt.py
"""

import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import CVRP
from common.verify import check, show

PROBLEM = "cvrp"

# context 编号 → 名字，打印用
CONTEXT_NAME = {
    COPT.CBCONTEXT_MIPRELAX: "MIPRELAX",
    COPT.CBCONTEXT_MIPSOL: "MIPSOL",
    COPT.CBCONTEXT_MIPNODE: "MIPNODE",
    COPT.CBCONTEXT_INCUMBENT: "INCUMBENT",
}


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def add_cvrp_constraints(model, x, u, dist, demand, capacity):
    """把 CVRP 的约束写进 model。LP 松弛和整数模型用的是同一份约束。"""
    nodes = range(len(demand))
    customers = range(1, len(demand))

    model.setObjective(
        cp.quicksum(dist[i, j] * x[i, j] for (i, j) in dist), sense=COPT.MINIMIZE
    )

    for i in customers:
        model.addConstr(
            cp.quicksum(x[i, j] for j in nodes if j != i) == 1, name=f"out_{i}"
        )
    for j in customers:
        model.addConstr(
            cp.quicksum(x[i, j] for i in nodes if i != j) == 1, name=f"in_{j}"
        )

    for i in customers:
        model.addConstr(u[i] >= demand[i], name=f"load_lb_{i}")
        model.addConstr(u[i] <= capacity, name=f"load_ub_{i}")
    for i in customers:
        for j in customers:
            if i != j:
                model.addConstr(
                    u[j] >= u[i] + demand[j] - capacity * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}",
                )

    model.addConstr(
        cp.quicksum(x[0, j] for j in customers)
        == cp.quicksum(x[i, 0] for i in customers),
        name="depot_balance",
    )


def new_vars(model, dist, demand, binary):
    """整数模型用 BINARY，LP 松弛用 [0,1] 连续变量。约束结构完全一样。"""
    x = {
        (i, j): model.addVar(
            lb=0,
            ub=1,
            vtype=COPT.BINARY if binary else COPT.CONTINUOUS,
            name=f"x{i}_{j}",
        )
        for (i, j) in dist
    }
    u = {
        i: model.addVar(lb=0, vtype=COPT.CONTINUOUS, name=f"u{i}")
        for i in range(1, len(demand))
    }
    return x, u


def solve_lp_relaxation():
    """方案 A（可移植）：单独建 LP 松弛模型求一次，不需要任何回调。"""
    data = CVRP
    dist = build_distance(data["coord"])

    env = cp.Envr()
    model = env.createModel("CVRP_LP")
    model.setParam("Logging", 0)

    x, u = new_vars(model, dist, data["demand"], binary=False)
    add_cvrp_constraints(model, x, u, dist, data["demand"], data["capacity"])
    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"LP 松弛没解出来，status={model.status}")

    values = {(i, j): x[i, j].x for (i, j) in dist}
    fractional = {a: v for a, v in values.items() if 1e-9 < v < 1 - 1e-9}
    return model.objval, values, fractional


class RootCallback(cp.CallbackBase):
    """记录回调触发次数，并抓第一次根松弛解 / 第一个整数解。

    注意：回调里抛出的异常**不会**变成 traceback，而是直接把进程打死
    （libc++abi: Swig::DirectorMethodException）。所以这里整体兜住。
    """

    def __init__(self):
        super().__init__()
        self.counts = {}
        self.relax_obj = None
        self.relax_vec = None
        self.mip_vec = None
        self.errors = []

    def callback(self):
        try:
            where = self.where()          # 注意：是方法调用，不是属性
            self.counts[where] = self.counts.get(where, 0) + 1

            if where == COPT.CBCONTEXT_MIPRELAX and self.relax_vec is None:
                self.relax_obj = self.getInfo(COPT.CbInfo.RelaxSolObj)
                self.relax_vec = self.getRelaxSol()      # 返回 list，按变量创建顺序
            elif where == COPT.CBCONTEXT_MIPSOL and self.mip_vec is None:
                self.mip_vec = self.getSolution()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")


def solve_mip_with_callback():
    """方案 B：在 MIP 求解过程中用回调抓内部信息。"""
    data = CVRP
    dist = build_distance(data["coord"])
    arcs = list(dist)

    env = cp.Envr()
    model = env.createModel("CVRP")
    model.setParam("Logging", 0)

    x, u = new_vars(model, dist, data["demand"], binary=True)
    add_cvrp_constraints(model, x, u, dist, data["demand"], data["capacity"])

    cb = RootCallback()
    model.setCallback(
        cb,
        COPT.CBCONTEXT_MIPRELAX
        | COPT.CBCONTEXT_MIPSOL
        | COPT.CBCONTEXT_MIPNODE
        | COPT.CBCONTEXT_INCUMBENT,
    )
    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，status={model.status}")

    counts = {CONTEXT_NAME.get(k, str(k)): v for k, v in cb.counts.items()}
    solution = {f"x{i}_{j}": x[i, j].x for (i, j) in arcs}
    return model.objval, solution, cb, counts


def main():
    lp_obj, lp_values, fractional = solve_lp_relaxation()
    print(f"[COPT] 根 LP 松弛界        = {lp_obj:.6g}")
    print(f"[COPT] LP 解里的分数弧     = {len(fractional)} 条")
    for (i, j), value in list(fractional.items())[:6]:
        print(f"        x{i}_{j} = {value:.4f}")

    mip_obj, solution, cb, counts = solve_mip_with_callback()
    arcs = list(lp_values)
    dist = build_distance(CVRP["coord"])
    # 回调给的松弛解按变量创建顺序排：先是 30 个 x，再是 5 个 u
    if cb.relax_vec is None:
        # 本章 11.4.1 说过：有些模型上 MIPRELAX 一次都不触发，必须有回退路径
        print("[COPT] MIPRELAX 一次都没触发，回落用方案 A 的 LP 松弛界")
        cb_relax_obj = lp_obj
    else:
        cb_relax_obj = sum(dist[a] * cb.relax_vec[k] for k, a in enumerate(arcs))
    # MIPSOL 里的 BestObj 是"当前最好界"，不是这个解的目标值；目标值要按解自己算
    cand_obj = (
        sum(dist[a] * cb.mip_vec[k] for k, a in enumerate(arcs))
        if cb.mip_vec is not None
        else float("nan")          # 这个模型上 MIPSOL 没触发过
    )

    print(f"[COPT] 回调触发次数        = {counts}")
    print(f"[COPT] 回调拿到的松弛界    = {cb.relax_obj}")
    print(f"[COPT] 用回调松弛解自算目标 = {cb_relax_obj:.6g}")
    # 多线程下先触发哪个启发式不确定，这个数每次可能不同
    print(f"[COPT] 第一个 MIPSOL 解目标 = {cand_obj:.6g}（按解自己算）")
    print(f"[COPT] 回调里被兜住的异常   = {len(cb.errors)} 次")
    print(f"[COPT] 整数最优            = {mip_obj:.6g}")
    print(f"[COPT] 根节点 Gap           = {(mip_obj - lp_obj) / abs(mip_obj) * 100:.2f}%")

    show("COPT", mip_obj, solution)
    check(PROBLEM, mip_obj)


if __name__ == "__main__":
    main()
