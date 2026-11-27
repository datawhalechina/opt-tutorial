"""第二章 整数规划 —— 0-1 背包，用 COPT 求解。

例题：4 件物品，各有重量和价值，背包容量 10，每件物品只能放或不放。
运行：python code/ch02_knapsack/knapsack_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import KNAPSACK
from common.verify import check, show

PROBLEM = "knapsack"


def solve():
    data = KNAPSACK
    n = len(data["weight"])

    env = cp.Envr()
    model = env.createModel("Knapsack")
    model.setParam("Logging", 0)

    # 0-1 变量：批量建变量用 nameprefix 给名字加前缀
    x = model.addVars(n, vtype=COPT.BINARY, nameprefix="x")

    model.setObjective(
        cp.quicksum(data["value"][i] * x[i] for i in range(n)), sense=COPT.MAXIMIZE
    )
    model.addConstr(
        cp.quicksum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        name="Capacity",      # ⚠️ COPT 必须写 name=：第 2 个位置参数是 sense（约束关系符），不是名字
    )

    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 status={model.status}")

    return model.objval, {v.getName(): v.x for v in model.getVars()}


if __name__ == "__main__":
    objective, solution = solve()
    show("COPT", objective, solution)
    check(PROBLEM, objective)
