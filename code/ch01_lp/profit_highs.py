"""第一章 线性规划 —— 用 HiGHS（highspy，开源）求解。

例题：工厂生产 A、B 两种产品，求利润最大化的产量。
运行：python code/ch01_lp/profit_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def solve():
    data = LP_PROFIT
    profit_a, profit_b = data["profit"]

    # 1. 建模型
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)   # True 打印求解日志

    # 2. 建变量：lb 是下界
    a = model.addVariable(lb=0, name="A")
    b = model.addVariable(lb=0, name="B")

    # 3. 目标函数：直接用 a、b 这些变量拼表达式
    model.maximize(profit_a * a + profit_b * b)

    # 4. 约束：表达式和常数比较即可，第二个参数是约束名
    model.addConstr(
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        "Production_Time",
    )
    model.addConstr(
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        "Raw_Material",
    )

    # 5. 求解
    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，模型状态={model.getModelStatus()}")

    # 6. 取解：model.variableValue(变量) 取单个变量的值
    return model.getObjectiveValue(), {"A": model.variableValue(a), "B": model.variableValue(b)}


if __name__ == "__main__":
    objective, solution = solve()
    show("HiGHS", objective, solution)
    check(PROBLEM, objective)
