"""第一章 线性规划 —— 用 COPT 求解。

例题：工厂生产 A、B 两种产品，求利润最大化的产量。
运行：python code/ch01_lp/profit_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def solve():
    data = LP_PROFIT
    profit_a, profit_b = data["profit"]

    # 1. 建环境和模型：COPT 要先有 Envr 才能建模型
    env = cp.Envr()
    model = env.createModel("Maximize_Profit")
    model.setParam("Logging", 0)         # 0 关闭日志，想看日志就改成 1

    # 2. 建变量
    a = model.addVar(lb=0, name="A")
    b = model.addVar(lb=0, name="B")

    # 3. 目标函数
    model.setObjective(profit_a * a + profit_b * b, sense=COPT.MAXIMIZE)

    # 4. 约束：⚠️ COPT 的 addConstr 第 2 个位置参数是 sense，不是 name，
    #    名字必须写成 name=...，否则名字会被静默丢弃（写成 R0、R1）。
    model.addConstr(
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        name="Production_Time",
    )
    model.addConstr(
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        name="Raw_Material",
    )

    # 5. 求解
    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 status={model.status}")

    # 6. 取解：objval 是目标值，var.x 是变量取值
    return model.objval, {v.getName(): v.x for v in model.getVars()}


if __name__ == "__main__":
    objective, solution = solve()
    show("COPT", objective, solution)
    check(PROBLEM, objective)
