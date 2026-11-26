"""第二章 整数规划 —— 0-1 背包，用 Gurobi 求解。

例题：4 件物品，各有重量和价值，背包容量 10，每件物品只能放或不放。
运行：python code/ch02_knapsack/knapsack_gurobi.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import KNAPSACK
from common.verify import check, show

PROBLEM = "knapsack"


def solve():
    data = KNAPSACK
    n = len(data["weight"])

    model = gp.Model("Knapsack")
    model.Params.OutputFlag = 0

    # 0-1 变量：每件物品放(1)或不放(0)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # 目标：价值最大
    model.setObjective(
        gp.quicksum(data["value"][i] * x[i] for i in range(n)), GRB.MAXIMIZE
    )
    # 约束：总重量不超过容量
    model.addConstr(
        gp.quicksum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        name="Capacity",
    )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 Status={model.Status}")

    return model.ObjVal, {f"x{i + 1}": x[i].X for i in range(n)}


if __name__ == "__main__":
    objective, solution = solve()
    show("Gurobi", objective, solution)
    check(PROBLEM, objective)
