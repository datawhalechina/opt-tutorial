"""第十一章 根松弛与回调 —— 用 Gurobi 求解（需要 license，本脚本未实跑）。

⚠️ 本机 Gurobi 许可证已过期（License expired 2025-07-02），CI 里也没有，
   所以这份脚本只做了 API 签名核对，**没有实际运行过**。装上 license 后：
       python code/ch11_callback/rootrelax_gurobi.py

习题同为第 7 章的 CVRP（MTZ 写法）。Gurobi 的回调写法：
  - model.optimize(callback=fn)，fn 的签名是 fn(model, where)
  - where 是回调类型：GRB.Callback.MIPNODE / MIPSOL / MIP / SIMPLEX / BARRIER ...
  - 只有 where == MIPNODE 且 cbGet(MIPNODE_STATUS) == OPTIMAL 时才能调 cbGetNodeRel
  - cbGetNodeRel(vars) 可以传变量、变量列表、dict 或 MVar（官方文档写明都接受）
  - 加割：cbCut(lhs, sense, rhs) 只能在 MIPNODE；加懒约束：cbLazy(lhs, sense, rhs) 在 MIPNODE 或 MIPSOL
"""

import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import CVRP
from common.verify import check, show

PROBLEM = "cvrp"

WHERE_NAME = {
    GRB.Callback.MIPNODE: "MIPNODE",
    GRB.Callback.MIPSOL: "MIPSOL",
    GRB.Callback.MIP: "MIP",
    GRB.Callback.SIMPLEX: "SIMPLEX",
    GRB.Callback.BARRIER: "BARRIER",
    GRB.Callback.POLLING: "POLLING",
}


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def build_model(binary):
    """同一个 CVRP 模型，binary=False 时只求 LP 松弛。"""
    data = CVRP
    demand, capacity = data["demand"], data["capacity"]
    nodes = range(len(demand))
    customers = range(1, len(demand))
    dist = build_distance(data["coord"])
    arcs = list(dist)

    model = gp.Model("CVRP")
    model.Params.OutputFlag = 0

    # addVars 传入 (i, j) 列表，返回以元组为键的 tupledict
    x = model.addVars(
        arcs,
        lb=0,
        ub=1,
        vtype=GRB.BINARY if binary else GRB.CONTINUOUS,
        name="x",
    )
    u = model.addVars(customers, lb=0, vtype=GRB.CONTINUOUS, name="u")

    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for (i, j) in arcs), GRB.MINIMIZE
    )

    model.addConstrs(
        (gp.quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers),
        name="out",
    )
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in nodes if i != j) == 1 for j in customers),
        name="in",
    )
    model.addConstrs((u[i] >= demand[i] for i in customers), name="load_lb")
    model.addConstrs((u[i] <= capacity for i in customers), name="load_ub")
    model.addConstrs(
        (
            u[j] >= u[i] + demand[j] - capacity * (1 - x[i, j])
            for i in customers
            for j in customers
            if i != j
        ),
        name="mtz",
    )
    model.addConstr(
        gp.quicksum(x[0, j] for j in customers)
        == gp.quicksum(x[i, 0] for i in customers),
        name="depot_balance",
    )
    return model, x, dist


def solve_lp_relaxation():
    model, x, dist = build_model(binary=False)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP 松弛没解出来，Status={model.Status}")
    fractional = {a: x[a].X for a in dist if 1e-9 < x[a].X < 1 - 1e-9}
    return model.ObjVal, fractional


class RootCallback:
    """抓根节点 LP 松弛解、第一个整数解，并统计回调触发次数。"""

    def __init__(self, x):
        self.x = x
        self.counts = {}
        self.root_bound = None
        self.root_relax = None
        self.first_mip = None

    def __call__(self, model, where):
        name = WHERE_NAME.get(where, str(where))
        self.counts[name] = self.counts.get(name, 0) + 1

        if where == GRB.Callback.MIPNODE:
            # 只有节点 LP 解到最优时才拿得到松弛解
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0 and self.root_relax is None:
                self.root_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                self.root_relax = model.cbGetNodeRel(self.x)
        elif where == GRB.Callback.MIPSOL and self.first_mip is None:
            self.first_mip = model.cbGetSolution(self.x)


def solve_mip_with_callback():
    model, x, dist = build_model(binary=True)
    callback = RootCallback(x)
    model.optimize(callback=callback)

    if model.SolCount == 0:
        raise RuntimeError(f"没有找到可行解，Status={model.Status}")
    if model.Status != GRB.OPTIMAL:
        print(f"注意：未证明最优（Status={model.Status}），以下为当前最好解")

    solution = {f"x{i}_{j}": x[i, j].X for (i, j) in dist}
    return model.ObjVal, solution, callback


def main():
    lp_obj, fractional = solve_lp_relaxation()
    print(f"[Gurobi] 根 LP 松弛界       = {lp_obj:.6g}")
    print(f"[Gurobi] LP 解里的分数弧    = {len(fractional)} 条")

    mip_obj, solution, callback = solve_mip_with_callback()
    print(f"[Gurobi] 回调触发次数       = {callback.counts}")
    print(f"[Gurobi] 根节点下界         = {callback.root_bound}")
    if callback.root_relax is not None:
        own_obj = sum(
            build_distance(CVRP["coord"])[a] * callback.root_relax[a]
            for a in build_distance(CVRP["coord"])
        )
        print(f"[Gurobi] 回调松弛解自算目标  = {own_obj:.6g}")
    print(f"[Gurobi] 整数最优           = {mip_obj:.6g}")
    print(f"[Gurobi] 根节点 Gap          = {(mip_obj - lp_obj) / abs(mip_obj) * 100:.2f}%")

    show("Gurobi", mip_obj, solution)
    check(PROBLEM, mip_obj)


if __name__ == "__main__":
    try:
        main()
    except gp.GurobiError as exc:
        raise SystemExit(f"需要有效的 Gurobi license：{exc}") from exc
