"""第八章 建模层 —— 用 Pyomo 建模，后端可选（HiGHS / Cbc / SCIP / ...）。

Pyomo 的特点是"模型"和"求解器"彻底分离：模型只写一次，
SolverFactory("名字") 换成任何已装的后端即可。
运行：python code/ch08_modeling/pyomo_lp.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pyomo.environ as pyo

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def build_model():
    data = LP_PROFIT
    model = pyo.ConcreteModel("Maximize_Profit")

    model.A = pyo.Var(domain=pyo.NonNegativeReals)
    model.B = pyo.Var(domain=pyo.NonNegativeReals)

    model.profit = pyo.Objective(
        expr=data["profit"][0] * model.A + data["profit"][1] * model.B,
        sense=pyo.maximize,
    )
    model.production_time = pyo.Constraint(
        expr=data["time_per_unit"][0] * model.A + data["time_per_unit"][1] * model.B
        <= data["time_limit"]
    )
    model.raw_material = pyo.Constraint(
        expr=data["material_per_unit"][0] * model.A + data["material_per_unit"][1] * model.B
        <= data["material_limit"]
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

    return pyo.value(model.profit), {"A": pyo.value(model.A), "B": pyo.value(model.B)}


if __name__ == "__main__":
    objective, solution = solve()
    show(f"Pyomo + {solver_name}", objective, solution)
    check(PROBLEM, objective)
