"""第四章 建模技巧 —— 半连续产量（指示约束写法），用 CPLEX（docplex）求解。

例题同 scheduling_gurobi.py：机器一旦启用，产量必须不低于下限。
运行：python code/ch04_techniques/scheduling_cplex.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docplex.mp.model import Model

from common.data import SCHEDULING
from common.verify import check, show

PROBLEM = "scheduling"


def solve():
    data = SCHEDULING
    n_mach = len(data["fixed_cost"])
    n_prod = len(data["demand"])

    model = Model("Scheduling")

    y = model.binary_var_list(n_mach, name="y")
    p = model.continuous_var_matrix(n_mach, n_prod, lb=0, name="p")

    model.minimize(
        model.sum(data["fixed_cost"][i] * y[i] for i in range(n_mach))
        + model.sum(
            data["cost"][i][k] * p[i, k] for i in range(n_mach) for k in range(n_prod)
        )
    )

    model.add_constraints(
        (
            model.sum(p[i, k] for i in range(n_mach)) == data["demand"][k]
            for k in range(n_prod)
        ),
        names=[f"demand{k + 1}" for k in range(n_prod)],
    )
    model.add_constraints(
        (
            model.sum(p[i, k] for k in range(n_prod)) <= data["capacity"][i] * y[i]
            for i in range(n_mach)
        ),
        names=[f"capacity{i + 1}" for i in range(n_mach)],
    )

    # docplex 的指示约束写法：binary_var 取 active_value 时，linear_ct 必须成立
    for i in range(n_mach):
        model.add_indicator(
            y[i],
            model.sum(p[i, k] for k in range(n_prod)) >= data["min_output"][i],
            active_value=1,
            name=f"min_output{i + 1}",
        )

    solution = model.solve(log_output=False)
    if solution is None:
        raise RuntimeError(f"没有求到最优解：{model.get_solve_status()}")

    result = {f"y{i + 1}": solution.get_value(y[i]) for i in range(n_mach)}
    result.update(
        {f"p{i + 1}{k + 1}": solution.get_value(p[i, k]) for i in range(n_mach) for k in range(n_prod)}
    )
    return solution.objective_value, result


if __name__ == "__main__":
    objective, solution = solve()
    show("CPLEX（指示约束）", objective, solution)
    check(PROBLEM, objective)
