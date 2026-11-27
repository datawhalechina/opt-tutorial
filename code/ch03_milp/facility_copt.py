"""第三章 混合整数规划 —— 设施选址，用 COPT 求解。

例题：3 个候选仓库、4 个客户。开仓有固定成本，运输按单位需求计费，
求"开仓 + 运输"总成本最小。
运行：python code/ch03_milp/facility_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def solve():
    data = FACILITY
    n_fac = len(data["fixed_cost"])
    n_cus = len(data["demand"])
    big_m = sum(data["demand"])

    env = cp.Envr()
    model = env.createModel("FacilityLocation")
    model.setParam("Logging", 0)

    y = model.addVars(n_fac, vtype=COPT.BINARY, nameprefix="y")
    x = model.addVars(n_fac, n_cus, vtype=COPT.CONTINUOUS, nameprefix="x")

    model.setObjective(
        cp.quicksum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + cp.quicksum(
            data["transport_cost"][i][j] * x[i, j] for i in range(n_fac) for j in range(n_cus)
        ),
        sense=COPT.MINIMIZE,
    )

    model.addConstrs(
        (cp.quicksum(x[i, j] for i in range(n_fac)) == data["demand"][j] for j in range(n_cus)),
        nameprefix="demand",
    )
    model.addConstrs(
        (cp.quicksum(x[i, j] for j in range(n_cus)) <= big_m * y[i] for i in range(n_fac)),
        nameprefix="open_link",
    )

    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 status={model.status}")

    solution = {f"y{i + 1}": y[i].x for i in range(n_fac)}
    solution.update({f"x{i + 1}{j + 1}": x[i, j].x for i in range(n_fac) for j in range(n_cus)})
    return model.objval, solution


if __name__ == "__main__":
    objective, solution = solve()
    show("COPT", objective, solution)
    check(PROBLEM, objective)
