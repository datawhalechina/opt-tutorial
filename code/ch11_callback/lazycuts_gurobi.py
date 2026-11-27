"""第十一章 回调能做什么（一）：用懒约束在求解中途补割 —— Gurobi 版（需要 license，未实跑）。

⚠️ 本机 Gurobi 许可证已过期（License expired 2025-07-02），这份脚本只做了 API 签名核对，
   **没有实际运行过**。装上 license 后：
       python code/ch11_callback/lazycuts_gurobi.py
   COPT 版的同一套逻辑（code/ch11_callback/lazycuts_copt.py）是实跑过的，结果 119.48。

习题：第 7 章 CVRP 去掉载重约束，就是 6 点 TSP（0 号点是配送中心，1~5 是客户）。
      模型里故意不写子回路消除，靠 cbLazy 在回调里补。

Gurobi 懒约束三个必做动作：
  1. model.Params.LazyConstraints = 1        ← 忘了这行，cbLazy 加的约束不生效
  2. 在 where == GRB.Callback.MIPSOL 里判断当前整数解有没有子回路
  3. model.cbLazy(lhs, GRB.LESS_EQUAL, rhs)
"""

import itertools
import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import CVRP

EXPECTED_TSP = 119.48     # 与 lazycuts_copt.py 的实测值一致


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def find_cycles(arc_values):
    """把"被选中的弧"还原成若干条回路。"""
    nxt = {i: j for (i, j), value in arc_values.items() if value > 0.5}
    cycles, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        cycle, cur = [start], nxt[start]
        while cur != start and cur not in seen:
            cycle.append(cur)
            cur = nxt[cur]
        seen.update(cycle)
        cycles.append(cycle)
    return cycles


def build_tsp(use_lazy):
    data = CVRP
    coord = data["coord"]
    n = len(coord)
    dist = build_distance(coord)

    model = gp.Model("TSP")
    model.Params.OutputFlag = 0
    if use_lazy:
        model.Params.LazyConstraints = 1        # 不加这行，cbLazy 的约束不会生效

    x = model.addVars(list(dist), vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(dist[i, j] * x[i, j] for (i, j) in dist), GRB.MINIMIZE)
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1, name=f"in_{j}")
    return model, x, dist


class SubtourCutter:
    """在 MIPSOL 里分离子回路约束并加进去。"""

    def __init__(self, x, arcs):
        self.x = x
        self.arcs = arcs
        self.cuts = []

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return
        values = dict(zip(self.arcs, model.cbGetSolution([self.x[a] for a in self.arcs])))
        for cycle in find_cycles(values):
            if 0 not in cycle:
                model.cbLazy(
                    gp.quicksum(
                        self.x[i, j] for i in cycle for j in cycle if i != j
                    ),
                    GRB.LESS_EQUAL,
                    len(cycle) - 1,
                )
                self.cuts.append(sorted(cycle))


def solve_tsp(use_lazy):
    model, x, dist = build_tsp(use_lazy)
    cutter = None
    if use_lazy:
        cutter = SubtourCutter(x, list(dist))
        model.optimize(cutter)
    else:
        model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"没有找到可行解，Status={model.Status}")
    values = {a: x[a].X for a in dist}
    return model.ObjVal, find_cycles(values), cutter


def brute_force_tsp():
    """6 个点、固定起点 0，只有 5! = 120 条回路。"""
    coord = CVRP["coord"]
    dist = build_distance(coord)
    best = min(
        sum(dist[tour[k], tour[k + 1]] for k in range(len(coord)))
        for perm in itertools.permutations(range(1, len(coord)))
        for tour in [(0,) + perm + (0,)]
    )
    return round(best, 2)


def main():
    base_obj, base_cycles, _ = solve_tsp(use_lazy=False)
    print("[Gurobi] 不写子回路消除：")
    print(f"        求解器报告最优 = {base_obj:.6g}")
    print(f"        但这个'解'其实是 {len(base_cycles)} 条回路：{base_cycles}")

    tsp_obj, cycles, cutter = solve_tsp(use_lazy=True)
    print("[Gurobi] 用懒约束补子回路消除：")
    print(f"        求解器报告最优 = {tsp_obj:.6g}")
    print(f"        回路 = {cycles}")
    print(f"        回调里一共加了 {len(cutter.cuts)} 条懒约束")

    brute = brute_force_tsp()
    print(f"[Gurobi] 暴力枚举（120 条回路）= {brute:.6g}")
    assert abs(tsp_obj - brute) < 1e-6, "懒约束的结果和暴力枚举对不上"
    assert abs(tsp_obj - EXPECTED_TSP) < 1e-6, "和本章记录的实测值对不上"


if __name__ == "__main__":
    try:
        main()
    except gp.GurobiError as exc:
        raise SystemExit(f"需要有效的 Gurobi license：{exc}") from exc
