"""第八章 建模层 —— 用 PuLP 建混合整数模型（设施选址）。

运行：python code/ch08_modeling/pulp_milp.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pulp

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def solve(backend=None):
    data = FACILITY
    n_fac, n_cus = len(data["fixed_cost"]), len(data["demand"])
    big_m = sum(data["demand"])
    backend = backend or pulp.HiGHS(msg=False)

    model = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)

    y = [pulp.LpVariable(f"y{i + 1}", cat="Binary") for i in range(n_fac)]
    x = {
        (i, j): pulp.LpVariable(f"x{i + 1}{j + 1}", lowBound=0)
        for i in range(n_fac)
        for j in range(n_cus)
    }

    model += (
        pulp.lpSum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + pulp.lpSum(data["transport_cost"][i][j] * x[i, j] for i in range(n_fac) for j in range(n_cus))
    )

    for j in range(n_cus):
        model += pulp.lpSum(x[i, j] for i in range(n_fac)) == data["demand"][j], f"demand{j + 1}"
    for i in range(n_fac):
        model += pulp.lpSum(x[i, j] for j in range(n_cus)) <= big_m * y[i], f"open_link{i + 1}"

    model.solve(backend)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={pulp.LpStatus[model.status]}")

    solution = {v.name: v.value() for v in model.variables()}
    return pulp.value(model.objective), solution


if __name__ == "__main__":
    objective, solution = solve()
    show("PuLP + HiGHS", objective, solution)
    check(PROBLEM, objective)
