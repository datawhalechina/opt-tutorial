"""第三章 混合整数规划 —— 设施选址，用 Gurobi 求解。

例题：3 个候选仓库、4 个客户。开仓有固定成本，运输按单位需求计费，
求"开仓 + 运输"总成本最小。
运行：python code/ch03_milp/facility_gurobi.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def solve():
    data = FACILITY
    n_fac = len(data["fixed_cost"])
    n_cus = len(data["demand"])
    big_m = sum(data["demand"])          # 连边系数，客户总需求量就是天然的 big-M

    model = gp.Model("FacilityLocation")
    model.Params.OutputFlag = 0

    # y[i]: 是否开仓（0-1）；x[i, j]: 从仓库 i 运给客户 j 的量（连续）
    y = model.addVars(n_fac, vtype=GRB.BINARY, name="y")
    x = model.addVars(n_fac, n_cus, vtype=GRB.CONTINUOUS, name="x")

    # 目标：固定成本 + 运输成本
    model.setObjective(
        gp.quicksum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + gp.quicksum(
            data["transport_cost"][i][j] * x[i, j] for i in range(n_fac) for j in range(n_cus)
        ),
        GRB.MINIMIZE,
    )

    # 每个客户的需求都要被满足
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(n_fac)) == data["demand"][j] for j in range(n_cus)),
        name="demand",
    )
    # 只有开了仓才能往外出货（big-M 写法）
    model.addConstrs(
        (
            gp.quicksum(x[i, j] for j in range(n_cus)) <= big_m * y[i]
            for i in range(n_fac)
        ),
        name="open_link",
    )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 Status={model.Status}")

    solution = {f"y{i + 1}": y[i].X for i in range(n_fac)}
    solution.update({f"x{i + 1}{j + 1}": x[i, j].X for i in range(n_fac) for j in range(n_cus)})
    return model.ObjVal, solution


if __name__ == "__main__":
    objective, solution = solve()
    show("Gurobi", objective, solution)
    check(PROBLEM, objective)
