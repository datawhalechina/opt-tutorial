"""第七章 车辆路径问题（CVRP）—— 用 COPT 求解。

例题：1 个配送中心 + 5 个客户，车辆载重 15，求总行驶距离最短的配送路线。
模型：两下标流模型 + MTZ 子回路消除约束。MTZ 用的是大 M 写法，
      所以五个求解器都能直接跑，不依赖指示约束。

运行：python code/ch07_cvrp/cvrp_copt.py
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


def build_distance(coord):
    """两点间欧氏距离，保留 2 位小数，保证不同求解器读到的是同一份数据。"""
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
    coord = data["coord"]
    demand = data["demand"]
    capacity = data["capacity"]
    nodes = range(len(coord))
    customers = range(1, len(coord))               # 0 是配送中心
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    dist = build_distance(coord)

    env = cp.Envr()
    model = env.createModel("CVRP")
    model.setParam("Logging", 0)

    x = model.addVars(arcs, vtype=COPT.BINARY, nameprefix="x")
    u = model.addVars(customers, lb=0, vtype=COPT.CONTINUOUS, nameprefix="u")

    model.setObjective(cp.quicksum(dist[i, j] * x[i, j] for i, j in arcs), sense=COPT.MINIMIZE)

    # 每个客户恰好被进入一次、离开一次
    for i in customers:
        model.addConstr(cp.quicksum(x[i, j] for j in nodes if j != i) == 1, name=f"out_{i}")
    for j in customers:
        model.addConstr(cp.quicksum(x[i, j] for i in nodes if i != j) == 1, name=f"in_{j}")

    # MTZ：u[i] 表示车辆离开客户 i 时车上已装的货量
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

    # 离开配送中心的车辆数 = 回到配送中心的车辆数
    model.addConstr(
        cp.quicksum(x[0, j] for j in customers) == cp.quicksum(x[i, 0] for i in customers),
        name="depot_balance",
    )

    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 status={model.status}")

    used = [(i, j) for (i, j) in arcs if x[i, j].x > 0.5]
    print_routes(used, demand, capacity)
    return model.objval, {f"x{i}_{j}": x[i, j].x for (i, j) in used}


if __name__ == "__main__":
    objective, solution = solve()
    show("COPT", objective, solution)
    check(PROBLEM, objective)
