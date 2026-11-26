"""第三章 混合整数规划 —— 设施选址，用 HiGHS（开源）求解。

例题：3 个候选仓库、4 个客户。开仓有固定成本，运输按单位需求计费，
求"开仓 + 运输"总成本最小。
运行：python code/ch03_milp/facility_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import FACILITY
from common.verify import check, show

PROBLEM = "facility"


def solve():
    data = FACILITY
    n_fac = len(data["fixed_cost"])
    n_cus = len(data["demand"])
    big_m = sum(data["demand"])

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    # y[i] 是 0-1 变量（下界 0、上界 1 的整数变量）
    y = [
        model.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger, name=f"y{i + 1}")
        for i in range(n_fac)
    ]
    # x[i, j] 是连续变量，默认下界 0
    x = {
        (i, j): model.addVariable(lb=0, name=f"x{i + 1}{j + 1}")
        for i in range(n_fac)
        for j in range(n_cus)
    }

    model.minimize(
        sum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + sum(
            data["transport_cost"][i][j] * x[i, j]
            for i in range(n_fac)
            for j in range(n_cus)
        )
    )

    for j in range(n_cus):
        model.addConstr(
            sum(x[i, j] for i in range(n_fac)) == data["demand"][j], f"demand{j + 1}"
        )
    for i in range(n_fac):
        model.addConstr(
            sum(x[i, j] for j in range(n_cus)) <= big_m * y[i], f"open_link{i + 1}"
        )

    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，模型状态={model.getModelStatus()}")

    result = {f"y{i + 1}": model.variableValue(y[i]) for i in range(n_fac)}
    result.update(
        {
            f"x{i + 1}{j + 1}": model.variableValue(x[i, j])
            for i in range(n_fac)
            for j in range(n_cus)
        }
    )
    return model.getObjectiveValue(), result


if __name__ == "__main__":
    objective, solution = solve()
    show("HiGHS", objective, solution)
    check(PROBLEM, objective)
