"""第三章 混合整数规划 —— 设施选址，用 CPLEX（docplex）求解。

例题：3 个候选仓库、4 个客户。开仓有固定成本，运输按单位需求计费，
求"开仓 + 运输"总成本最小。
运行：python code/ch03_milp/facility_cplex.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docplex.mp.model import Model
from docplex.util.status import JobSolveStatus

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def solve():
    data = FACILITY
    n_fac = len(data["fixed_cost"])
    n_cus = len(data["demand"])
    big_m = sum(data["demand"])

    model = Model("FacilityLocation")

    y = model.binary_var_list(n_fac, name="y")
    # continuous_var_matrix 一次建二维连续变量，下标就是 (i, j)
    x = model.continuous_var_matrix(n_fac, n_cus, lb=0, name="x")

    model.minimize(
        model.sum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + model.sum(
            data["transport_cost"][i][j] * x[i, j] for i in range(n_fac) for j in range(n_cus)
        )
    )

    model.add_constraints(
        (
            model.sum(x[i, j] for i in range(n_fac)) == data["demand"][j]
            for j in range(n_cus)
        ),
        names=[f"demand{j + 1}" for j in range(n_cus)],
    )
    model.add_constraints(
        (
            model.sum(x[i, j] for j in range(n_cus)) <= big_m * y[i]
            for i in range(n_fac)
        ),
        names=[f"open_link{i + 1}" for i in range(n_fac)],
    )

    solution = model.solve(log_output=False)
    if solution is None:
        raise RuntimeError(f"没有找到可行解：{model.get_solve_status()}")
    if model.get_solve_status() != JobSolveStatus.OPTIMAL_SOLUTION:
        print(f"注意：未证明最优（{model.get_solve_status()}），以下为当前最好解")

    result = {f"y{i + 1}": solution.get_value(y[i]) for i in range(n_fac)}
    result.update(
        {f"x{i + 1}{j + 1}": solution.get_value(x[i, j]) for i in range(n_fac) for j in range(n_cus)}
    )
    return solution.objective_value, result


if __name__ == "__main__":
    objective, solution = solve()
    show("CPLEX", objective, solution)
    check(PROBLEM, objective)
