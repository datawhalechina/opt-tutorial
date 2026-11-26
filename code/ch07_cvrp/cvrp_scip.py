"""第七章 车辆路径问题（CVRP）—— 用 SCIP（开源）求解。

例题与模型同 cvrp_copt.py：两下标流模型 + MTZ 子回路消除（大 M 写法）。
运行：python code/ch07_cvrp/cvrp_scip.py
"""

import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import Model, quicksum

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
    next_node = dict(arcs_used)
    for start in [j for (i, j) in arcs_used if i == 0]:
        tour, cur = [0, start], start
        while cur != 0:
            cur = next_node[cur]
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

    model = Model("CVRP")
    model.hideOutput()

    x = {(i, j): model.addVar(vtype="B", name=f"x{i}_{j}") for (i, j) in arcs}
    u = {i: model.addVar(vtype="C", lb=0, name=f"u{i}") for i in customers}

    model.setObjective(
        quicksum(dist[i, j] * x[i, j] for (i, j) in arcs), sense="minimize"
    )

    for i in customers:
        model.addCons(quicksum(x[i, j] for j in nodes if j != i) == 1, name=f"out_{i}")
    for j in customers:
        model.addCons(quicksum(x[i, j] for i in nodes if i != j) == 1, name=f"in_{j}")

    for i in customers:
        model.addCons(u[i] >= demand[i], name=f"load_lb_{i}")
        model.addCons(u[i] <= capacity, name=f"load_ub_{i}")
    for i in customers:
        for j in customers:
            if i != j:
                model.addCons(
                    u[j] >= u[i] + demand[j] - capacity * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}",
                )

    model.addCons(
        quicksum(x[0, j] for j in customers) == quicksum(x[i, 0] for i in customers),
        name="depot_balance",
    )

    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={model.getStatus()}")

    used = [arc for arc in arcs if model.getVal(x[arc]) > 0.5]
    print_routes(used, demand, capacity)
    return model.getObjVal(), {f"x{i}_{j}": model.getVal(x[i, j]) for (i, j) in used}


if __name__ == "__main__":
    objective, solution = solve()
    show("SCIP", objective, solution)
    check(PROBLEM, objective)
