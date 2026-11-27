"""第二章 整数规划 —— 0-1 背包，用 CPLEX（docplex）求解。

例题：4 件物品，各有重量和价值，背包容量 10，每件物品只能放或不放。
运行：python code/ch02_knapsack/knapsack_cplex.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docplex.mp.model import Model
from docplex.util.status import JobSolveStatus

from common.data import KNAPSACK
from common.verify import check, show

PROBLEM = "knapsack"


def solve():
    data = KNAPSACK
    n = len(data["weight"])

    model = Model("Knapsack")

    # 0-1 变量：binary_var_list 一次建一组
    x = model.binary_var_list(n, name="x")

    model.maximize(model.sum(data["value"][i] * x[i] for i in range(n)))
    model.add_constraint(
        model.sum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        ctname="Capacity",
    )

    solution = model.solve(log_output=False)
    if solution is None:
        raise RuntimeError(f"没有找到可行解：{model.get_solve_status()}")
    if model.get_solve_status() != JobSolveStatus.OPTIMAL_SOLUTION:
        print(f"注意：未证明最优（{model.get_solve_status()}），以下为当前最好解")

    return solution.objective_value, {v.name: solution.get_value(v) for v in model.iter_variables()}


if __name__ == "__main__":
    objective, solution = solve()
    show("CPLEX", objective, solution)
    check(PROBLEM, objective)
