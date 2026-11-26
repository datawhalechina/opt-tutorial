"""第七章 车辆路径问题（CVRP）—— 用 HiGHS（开源）求解。

例题与模型同 cvrp_copt.py：两下标流模型 + MTZ 子回路消除（大 M 写法）。
运行：python code/ch07_cvrp/cvrp_highs.py
"""

import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

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

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    x = {
        (i, j): model.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger, name=f"x{i}_{j}")
        for (i, j) in arcs
    }
    u = {i: model.addVariable(lb=0, name=f"u{i}") for i in customers}

    # setObjective 只设目标，不会求解（minimize() 会立刻求解一次）
    model.setObjective(
        sum(dist[i, j] * x[i, j] for (i, j) in arcs), highspy.ObjSense.kMinimize
    )

    for i in customers:
        model.addConstr(sum(x[i, j] for j in nodes if j != i) == 1, f"out_{i}")
    for j in customers:
        model.addConstr(sum(x[i, j] for i in nodes if i != j) == 1, f"in_{j}")

    for i in customers:
        model.addConstr(u[i] >= demand[i], f"load_lb_{i}")
        model.addConstr(u[i] <= capacity, f"load_ub_{i}")
    for i in customers:
        for j in customers:
            if i != j:
                model.addConstr(
                    u[j] - u[i] - demand[j] + capacity * (1 - x[i, j]) >= 0,
                    f"mtz_{i}_{j}",
                )

    model.addConstr(
        sum(x[0, j] for j in customers) - sum(x[i, 0] for i in customers) == 0,
        "depot_balance",
    )

    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，模型状态={model.getModelStatus()}")

    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，模型状态={model.getModelStatus()}")
    used = [arc for arc in arcs if model.variableValue(x[arc]) > 0.5]
    print_routes(used, demand, capacity)
    return model.getObjectiveValue(), {f"x{i}_{j}": model.variableValue(x[i, j]) for (i, j) in used}


if __name__ == "__main__":
    objective, solution = solve()
    show("HiGHS", objective, solution)
    check(PROBLEM, objective)
