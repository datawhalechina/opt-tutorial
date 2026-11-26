"""第八章 建模层 —— 用 Pyomo 建混合整数模型（设施选址）。

运行：python code/ch08_modeling/pyomo_milp.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pyomo.environ as pyo

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def build_model():
    data = FACILITY
    n_fac, n_cus = len(data["fixed_cost"]), len(data["demand"])
    big_m = sum(data["demand"])

    model = pyo.ConcreteModel("FacilityLocation")
    model.FAC = pyo.RangeSet(0, n_fac - 1)
    model.CUS = pyo.RangeSet(0, n_cus - 1)
    model.ARC = pyo.Set(initialize=[(i, j) for i in range(n_fac) for j in range(n_cus)])

    model.y = pyo.Var(model.FAC, domain=pyo.Binary)
    model.x = pyo.Var(model.ARC, domain=pyo.NonNegativeReals)

    model.cost = pyo.Objective(
        expr=sum(data["fixed_cost"][i] * model.y[i] for i in model.FAC)
        + sum(data["transport_cost"][i][j] * model.x[i, j] for i, j in model.ARC),
        sense=pyo.minimize,
    )

    model.demand = pyo.Constraint(
        model.CUS, rule=lambda m, j: sum(m.x[i, j] for i in m.FAC) == data["demand"][j]
    )
    model.open_link = pyo.Constraint(
        model.FAC, rule=lambda m, i: sum(m.x[i, j] for j in m.CUS) <= big_m * m.y[i]
    )
    return model


def solve(solver_name="appsi_highs"):
    model = build_model()
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Pyomo backend {solver_name!r} is not available："
            f"appsi_highs / highs 需要 pip install highspy；"
            "名字写错时 Pyomo 不会报错，只会返回一个不可用的 UnknownSolver"
        )
    result = solver.solve(model)
    if (result.solver.status != pyo.SolverStatus.ok
            or str(result.solver.termination_condition) != "optimal"):
        raise RuntimeError(f"没有求到最优解，终止条件={result.solver.termination_condition}")

    solution = {f"y{i + 1}": pyo.value(model.y[i]) for i in model.FAC}
    solution.update({f"x{i + 1}{j + 1}": pyo.value(model.x[i, j]) for i, j in model.ARC})
    return pyo.value(model.cost), solution


if __name__ == "__main__":
    objective, solution = solve()
    show(f"Pyomo + {solver_name}", objective, solution)
    check(PROBLEM, objective)
