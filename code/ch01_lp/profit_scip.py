"""第一章 线性规划 —— 用 SCIP（PySCIPOpt，开源）求解。

例题：工厂生产 A、B 两种产品，求利润最大化的产量。
运行：python code/ch01_lp/profit_scip.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import Model

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def solve():
    data = LP_PROFIT
    profit_a, profit_b = data["profit"]

    # 1. 建模型
    model = Model("Maximize_Profit")
    model.hideOutput()                   # 想看求解日志就删掉这一行

    # 2. 建变量：vtype="C" 表示连续变量，"B" 是 0-1，"I" 是整数
    a = model.addVar(vtype="C", lb=0, name="A")
    b = model.addVar(vtype="C", lb=0, name="B")

    # 3. 目标函数：sense 写 "maximize" 或 "minimize"
    model.setObjective(profit_a * a + profit_b * b, sense="maximize")

    # 4. 约束
    model.addCons(
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        name="Production_Time",
    )
    model.addCons(
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        name="Raw_Material",
    )

    # 5. 求解：getStatus() 返回字符串，如 "optimal"
    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={model.getStatus()}")

    # 6. 取解：getVal(变量) 取变量值，getObjective() 取目标值
    return model.getObjVal(), {"A": model.getVal(a), "B": model.getVal(b)}


if __name__ == "__main__":
    objective, solution = solve()
    show("SCIP", objective, solution)
    check(PROBLEM, objective)
