"""第二章 整数规划 —— 0-1 背包，用 SCIP（开源）求解。

例题：4 件物品，各有重量和价值，背包容量 10，每件物品只能放或不放。
运行：python code/ch02_knapsack/knapsack_scip.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import Model, quicksum

from common.data import KNAPSACK
from common.verify import check, show

PROBLEM = "knapsack"


def solve():
    data = KNAPSACK
    n = len(data["weight"])

    model = Model("Knapsack")
    model.hideOutput()

    # vtype="B" 表示 0-1 变量
    x = [model.addVar(vtype="B", name=f"x{i + 1}") for i in range(n)]

    model.setObjective(quicksum(data["value"][i] * x[i] for i in range(n)), sense="maximize")
    model.addCons(
        quicksum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        name="Capacity",
    )

    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={model.getStatus()}")

    return model.getObjVal(), {f"x{i + 1}": model.getVal(x[i]) for i in range(n)}


if __name__ == "__main__":
    objective, solution = solve()
    show("SCIP", objective, solution)
    check(PROBLEM, objective)
