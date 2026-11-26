"""第七章 车辆路径问题（CVRP）—— 用 Gurobi 求解。

例题与模型同 cvrp_copt.py：两下标流模型 + MTZ 子回路消除（大 M 写法）。
运行：python code/ch07_cvrp/cvrp_gurobi.py
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


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def print_routes(arcs_used, demand, capacity):
    """把解里的边还原成路线。带防御：万一模型漏了子回路消除约束，这里报错而不是死循环。"""
    next_node = dict(arcs_used)
    for start in [j for (i, j) in arcs_used if i == 0]:
        tour, cur, visited = [0, start], start, {0, start}
        while True:
            if cur not in next_node:
                raise RuntimeError(f"解的路线断在节点 {cur}，模型可能有误")
            cur = next_node[cur]
            if cur == 0:                     # 回到配送中心，这条路线结束
                tour.append(0)
                break
            if cur in visited:
                raise RuntimeError(f"解中出现子回路（节点 {cur} 被重复访问），检查 MTZ 约束")
            visited.add(cur)
            tour.append(cur)
        load = sum(demand[node] for node in tour[1:-1])
        print(f"    路线 {' -> '.join(map(str, tour))}，载重 {load}/{capacity}")


def solve():
    data = CVRP
    coord, demand, capacity = data["coord"], data["demand"], data["capacity"]
    nodes = range(len(coord))
    customers = range(1, len(coord))
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    dist = build_distance(coord)

    model = gp.Model("CVRP")
    model.Params.OutputFlag = 0

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = model.addVars([i for i in customers], lb=0, vtype=GRB.CONTINUOUS, name="u")

    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i, j in arcs), GRB.MINIMIZE
    )

    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == 1, name=f"out_{i}")
    for j in customers:
        model.addConstr(gp.quicksum(x[i, j] for i in nodes if i != j) == 1, name=f"in_{j}")

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
        gp.quicksum(x[0, j] for j in customers) == gp.quicksum(x[i, 0] for i in customers),
        name="depot_balance",
    )

    model.optimize()
    # 先看有没有解（SolCount>0），再看是否证明了最优：
    # 到时限/解数上限等状态依然有可用解，不该当失败处理
    if model.SolCount == 0:
        raise RuntimeError(f"没有找到可行解，模型状态 Status={model.Status}")
    if model.Status != GRB.OPTIMAL:
        print(f"注意：未证明最优（Status={model.Status}），以下为当前最好解")

    used = [arc for arc in arcs if x[arc].X > 0.5]
    print_routes(used, demand, capacity)
    return model.ObjVal, {f"x{i}_{j}": x[i, j].X for (i, j) in used}


if __name__ == "__main__":
    objective, solution = solve()
    show("Gurobi", objective, solution)
    check(PROBLEM, objective)
