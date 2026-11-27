"""第四章 建模技巧 —— 半连续产量：机器一旦启用，产量必须不低于下限。

例题：2 台机器、3 个产品。机器启用有固定成本；不启用则产量为 0，
启用则产量必须落在 [min_output, capacity] 区间内 —— 这是典型的"或"逻辑，
有两种写法：指示约束（解法一，本文件）和大 M 约束（解法二，见 highs 版本）。

运行：python code/ch04_techniques/scheduling_gurobi.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import SCHEDULING
from common.verify import check, show

PROBLEM = "scheduling"


def solve():
    data = SCHEDULING
    n_mach = len(data["fixed_cost"])
    n_prod = len(data["demand"])

    model = gp.Model("Scheduling")
    model.Params.OutputFlag = 0

    y = model.addVars(n_mach, vtype=GRB.BINARY, name="y")        # 机器是否启用
    p = model.addVars(n_mach, n_prod, lb=0, name="p")            # 各机器各产品的产量

    model.setObjective(
        gp.quicksum(data["fixed_cost"][i] * y[i] for i in range(n_mach))
        + gp.quicksum(
            data["cost"][i][k] * p[i, k] for i in range(n_mach) for k in range(n_prod)
        ),
        GRB.MINIMIZE,
    )

    # 需求必须满足
    model.addConstrs(
        (gp.quicksum(p[i, k] for i in range(n_mach)) == data["demand"][k] for k in range(n_prod)),
        name="demand",
    )
    # 未启用的机器产量为 0：这条用乘法就能表达，不需要指示约束
    model.addConstrs(
        (
            gp.quicksum(p[i, k] for k in range(n_prod)) <= data["capacity"][i] * y[i]
            for i in range(n_mach)
        ),
        name="capacity",
    )
    # 启用的机器产量不低于下限：这是"如果 y=1 则 ..."的逻辑，用指示约束最直观
    for i in range(n_mach):
        model.addGenConstrIndicator(
            y[i],
            1,
            gp.quicksum(p[i, k] for k in range(n_prod)) >= data["min_output"][i],
            name=f"min_output{i + 1}",
        )

    model.optimize()
    # 先看有没有解（SolCount>0），再看是否证明了最优：
    # 到时限/解数上限等状态依然有可用解，不该当失败处理
    if model.SolCount == 0:
        raise RuntimeError(f"没有找到可行解，模型状态 Status={model.Status}")
    if model.Status != GRB.OPTIMAL:
        print(f"注意：未证明最优（Status={model.Status}），以下为当前最好解")

    solution = {f"y{i + 1}": y[i].X for i in range(n_mach)}
    solution.update({f"p{i + 1}{k + 1}": p[i, k].X for i in range(n_mach) for k in range(n_prod)})
    return model.ObjVal, solution


if __name__ == "__main__":
    objective, solution = solve()
    show("Gurobi（指示约束）", objective, solution)
    check(PROBLEM, objective)
