"""第四章 建模技巧 —— 半连续产量（指示约束写法），用 COPT 求解。

例题同 scheduling_gurobi.py：机器一旦启用，产量必须不低于下限。
运行：python code/ch04_techniques/scheduling_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import SCHEDULING
from common.verify import check, show

PROBLEM = "scheduling"


def solve():
    data = SCHEDULING
    n_mach = len(data["fixed_cost"])
    n_prod = len(data["demand"])

    env = cp.Envr()
    model = env.createModel("Scheduling")
    model.setParam("Logging", 0)

    y = model.addVars(n_mach, vtype=COPT.BINARY, nameprefix="y")
    p = model.addVars(n_mach, n_prod, lb=0, vtype=COPT.CONTINUOUS, nameprefix="p")

    model.setObjective(
        cp.quicksum(data["fixed_cost"][i] * y[i] for i in range(n_mach))
        + cp.quicksum(
            data["cost"][i][k] * p[i, k] for i in range(n_mach) for k in range(n_prod)
        ),
        sense=COPT.MINIMIZE,
    )

    model.addConstrs(
        (cp.quicksum(p[i, k] for i in range(n_mach)) == data["demand"][k] for k in range(n_prod)),
        nameprefix="demand",
    )
    model.addConstrs(
        (
            cp.quicksum(p[i, k] for k in range(n_prod)) <= data["capacity"][i] * y[i]
            for i in range(n_mach)
        ),
        nameprefix="capacity",
    )

    # 指示约束：y[i] = 1 时，产量 >= 下限。
    # 签名是 addGenConstrIndicator(binvar, binval, lhs, sense, rhs, ...)，
    # 这里把整条不等式拆成 lhs / sense / rhs 三段来写。
    for i in range(n_mach):
        model.addGenConstrIndicator(
            y[i],
            1,
            cp.quicksum(p[i, k] for k in range(n_prod)),
            COPT.GREATER_EQUAL,
            data["min_output"][i],
            name=f"min_output{i + 1}",
        )

    model.solve()
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"没有求到最优解，模型状态 status={model.status}")

    solution = {f"y{i + 1}": y[i].x for i in range(n_mach)}
    solution.update({f"p{i + 1}{k + 1}": p[i, k].x for i in range(n_mach) for k in range(n_prod)})
    return model.objval, solution


if __name__ == "__main__":
    objective, solution = solve()
    show("COPT（指示约束）", objective, solution)
    check(PROBLEM, objective)
