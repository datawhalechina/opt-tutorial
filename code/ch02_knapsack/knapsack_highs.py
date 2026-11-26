"""第二章 整数规划 —— 0-1 背包，用 HiGHS（开源）求解。

例题：4 件物品，各有重量和价值，背包容量 10，每件物品只能放或不放。
运行：python code/ch02_knapsack/knapsack_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import KNAPSACK
from common.verify import check, show

PROBLEM = "knapsack"


def solve():
    data = KNAPSACK
    n = len(data["weight"])

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    # 0-1 变量 = 下界 0、上界 1 的整数变量
    x = [
        model.addVariable(
            lb=0,
            ub=1,
            obj=data["value"][i],
            type=highspy.HighsVarType.kInteger,
            name=f"x{i + 1}",
        )
        for i in range(n)
    ]

    model.addConstr(
        sum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"], "Capacity"
    )
    # 目标系数在 addVariable 时已经通过 obj= 给好了，这里再声明求解方向即可
    model.maximize(sum(data["value"][i] * x[i] for i in range(n)))

    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，模型状态={model.getModelStatus()}")

    return model.getObjectiveValue(), {
        f"x{i + 1}": model.variableValue(x[i]) for i in range(n)
    }


if __name__ == "__main__":
    objective, solution = solve()
    show("HiGHS", objective, solution)
    check(PROBLEM, objective)
