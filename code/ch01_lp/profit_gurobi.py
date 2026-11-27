"""第一章 线性规划 —— 用 Gurobi 求解。

例题：工厂生产 A、B 两种产品，求利润最大化的产量。
运行：python code/ch01_lp/profit_gurobi.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def solve():
    data = LP_PROFIT
    profit_a, profit_b = data["profit"]

    # 1. 建模型
    model = gp.Model("Maximize_Profit")
    model.Params.OutputFlag = 0          # 0 关闭求解日志，想看日志就改成 1

    # 2. 建变量：lb 是下界，等价于 x >= 0
    a = model.addVar(lb=0, name="A")
    b = model.addVar(lb=0, name="B")

    # 3. 目标函数：最大化总利润
    model.setObjective(profit_a * a + profit_b * b, GRB.MAXIMIZE)

    # 4. 约束
    model.addConstr(
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        name="Production_Time",
    )
    model.addConstr(
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        name="Raw_Material",
    )

    # 5. 求解
    model.optimize()
    # 先看有没有解（SolCount>0），再看是否证明了最优：
    # 到时限/解数上限等状态依然有可用解，不该当失败处理
    if model.SolCount == 0:
        raise RuntimeError(f"没有找到可行解，模型状态 Status={model.Status}")
    if model.Status != GRB.OPTIMAL:
        print(f"注意：未证明最优（Status={model.Status}），以下为当前最好解")

    # 6. 取解：ObjVal 是目标值，var.X 是变量取值
    return model.ObjVal, {v.VarName: v.X for v in model.getVars()}


if __name__ == "__main__":
    objective, solution = solve()
    show("Gurobi", objective, solution)
    check(PROBLEM, objective)
