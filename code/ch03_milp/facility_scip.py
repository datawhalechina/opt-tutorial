"""第三章 混合整数规划 —— 设施选址，用 SCIP（开源）求解。

例题：3 个候选仓库、4 个客户。开仓有固定成本，运输按单位需求计费，
求"开仓 + 运输"总成本最小。
运行：python code/ch03_milp/facility_scip.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import Model, quicksum

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def solve():
    data = FACILITY
    n_fac = len(data["fixed_cost"])
    n_cus = len(data["demand"])
    big_m = sum(data["demand"])

    model = Model("FacilityLocation")
    model.hideOutput()

    y = [model.addVar(vtype="B", name=f"y{i + 1}") for i in range(n_fac)]
    x = {
        (i, j): model.addVar(vtype="C", lb=0, name=f"x{i + 1}{j + 1}")
        for i in range(n_fac)
        for j in range(n_cus)
    }

    model.setObjective(
        quicksum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + quicksum(
            data["transport_cost"][i][j] * x[i, j]
            for i in range(n_fac)
            for j in range(n_cus)
        ),
        sense="minimize",
    )

    for j in range(n_cus):
        model.addCons(
            quicksum(x[i, j] for i in range(n_fac)) == data["demand"][j], name=f"demand{j + 1}"
        )
    for i in range(n_fac):
        model.addCons(
            quicksum(x[i, j] for j in range(n_cus)) <= big_m * y[i], name=f"open_link{i + 1}"
        )

    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={model.getStatus()}")

    result = {f"y{i + 1}": model.getVal(y[i]) for i in range(n_fac)}
    result.update(
        {f"x{i + 1}{j + 1}": model.getVal(x[i, j]) for i in range(n_fac) for j in range(n_cus)}
    )
    return model.getObjVal(), result


if __name__ == "__main__":
    objective, solution = solve()
    show("SCIP", objective, solution)
    check(PROBLEM, objective)
