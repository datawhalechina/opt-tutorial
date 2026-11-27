"""第十一章 回调能做什么（一）：用懒约束在求解中途补割 —— COPT 版，可实跑。

习题：把第 7 章 CVRP 的"载重"去掉，只留"每个点进出各一次"，就是一个 6 点 TSP。
      0 号点是配送中心，1~5 是客户，坐标沿用 common.data.CVRP。

关键演示：
  1. 不在模型里写子回路消除约束 → 求解器会给你一个"最优解"，
     其实是由两条/多条互不相连的回路拼出来的，对 TSP 根本不可行（本脚本实测 110.8）
  2. 用回调在 MIPSOL 里检查整数解：发现有回路没经过 0 号点，就 addLazyConstr 补一条
     sum_{i,j ∈ S} x_ij <= |S| - 1，让求解器重新考虑 → 得到真正的 TSP 最优 119.48
  3. 最后用 5! = 120 条回路的暴力枚举验证 119.48 没算错

COPT 的懒约束接口：
  - 回调对象继承 coptpy.CallbackBase，覆写 callback(self)
  - 在 MIPSOL 里 self.getSolution() 取整数解，self.addLazyConstr(约束) 加懒约束
运行：python code/ch11_callback/lazycuts_copt.py
"""

import itertools
import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import CVRP

EXPECTED_TSP = 119.48     # 本章实测值，main() 末尾用暴力枚举再核对一次


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def find_cycles(arc_values):
    """把"被选中的弧"还原成若干条回路。arc_values 是 {(i, j): 值}。"""
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


def build_tsp():
    """只写出入度约束的 TSP 模型：故意不写子回路消除。"""
    data = CVRP
    coord = data["coord"]
    n = len(coord)
    dist = build_distance(coord)

    env = cp.Envr()
    model = env.createModel("TSP")
    model.setParam("Logging", 0)

    x = {(i, j): model.addVar(vtype=COPT.BINARY, name=f"x{i}_{j}") for (i, j) in dist}
    model.setObjective(
        cp.quicksum(dist[i, j] * x[i, j] for (i, j) in dist), sense=COPT.MINIMIZE
    )
    for i in range(n):
        model.addConstr(cp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
    for j in range(n):
        model.addConstr(cp.quicksum(x[i, j] for i in range(n) if i != j) == 1, name=f"in_{j}")
    return model, x, dist


class SubtourCutter(cp.CallbackBase):
    """在 MIPSOL 里分离子回路约束并加进去。"""

    def __init__(self, x):
        super().__init__()
        self.x = x
        self.cuts = []

    def callback(self):
        if self.where() != COPT.CBCONTEXT_MIPSOL:
            return
        values = dict(zip(self.x, self.getSolution()))
        for cycle in find_cycles(values):
            # 回路里没有 0 号点（配送中心）就是子回路，补 sum x_ij <= |S| - 1
            if 0 not in cycle:
                self.addLazyConstr(
                    cp.quicksum(
                        self.x[i, j] for i in cycle for j in cycle if i != j
                    )
                    <= len(cycle) - 1
                )
                self.cuts.append(sorted(cycle))


def solve_tsp(use_lazy):
    model, x, dist = build_tsp()
    cutter = None
    if use_lazy:
        cutter = SubtourCutter(x)
        model.setCallback(cutter, COPT.CBCONTEXT_MIPSOL)
    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，status={model.status}")

    values = {a: x[a].x for a in dist}
    return model.objval, find_cycles(values), cutter


def brute_force_tsp():
    """6 个点、固定起点 0，只有 5! = 120 条回路，直接枚举。"""
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
    print("[COPT] 不写子回路消除：")
    print(f"        求解器报告最优 = {base_obj:.6g}")
    print(f"        但这个'解'其实是 {len(base_cycles)} 条回路：{base_cycles}")

    tsp_obj, cycles, cutter = solve_tsp(use_lazy=True)
    print("[COPT] 用懒约束补子回路消除：")
    print(f"        求解器报告最优 = {tsp_obj:.6g}")
    print(f"        回路 = {cycles}")
    print(f"        回调里一共加了 {len(cutter.cuts)} 条懒约束：{cutter.cuts}")

    brute = brute_force_tsp()
    print(f"[COPT] 暴力枚举（120 条回路）= {brute:.6g}")
    assert abs(tsp_obj - brute) < 1e-6, "懒约束的结果和暴力枚举对不上"
    assert abs(tsp_obj - EXPECTED_TSP) < 1e-6, "和本章记录的实测值对不上"


if __name__ == "__main__":
    main()
