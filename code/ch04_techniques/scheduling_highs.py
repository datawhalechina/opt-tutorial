"""第四章 建模技巧 —— 半连续产量（大 M 写法），用 HiGHS（开源）求解。

HiGHS 不支持指示约束，所以这里用"大 M"把
"如果 y=1 则 产量 >= 下限" 写成两条普通线性约束：
    产量 >= min_output * y        （y=0 时退化为 产量 >= 0）
    产量 <= capacity   * y        （y=0 时退化为 产量 <= 0，即不生产）

运行：python code/ch04_techniques/scheduling_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import SCHEDULING
from common.verify import check, show

PROBLEM = "scheduling"


def solve():
    data = SCHEDULING
    n_mach = len(data["fixed_cost"])
    n_prod = len(data["demand"])

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    y = [
        model.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger, name=f"y{i + 1}")
        for i in range(n_mach)
    ]
    p = {
        (i, k): model.addVariable(lb=0, name=f"p{i + 1}{k + 1}")
        for i in range(n_mach)
        for k in range(n_prod)
    }

    model.minimize(
        sum(data["fixed_cost"][i] * y[i] for i in range(n_mach))
        + sum(data["cost"][i][k] * p[i, k] for i in range(n_mach) for k in range(n_prod))
    )

    for k in range(n_prod):
        model.addConstr(
            sum(p[i, k] for i in range(n_mach)) == data["demand"][k], f"demand{k + 1}"
        )

    for i in range(n_mach):
        total = sum(p[i, k] for k in range(n_prod))
        # 两条线性约束就等价于"启用才生产、且不低于下限"
        model.addConstr(total - data["capacity"][i] * y[i] <= 0, f"capacity{i + 1}")
        model.addConstr(total - data["min_output"][i] * y[i] >= 0, f"min_output{i + 1}")

    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，模型状态={model.getModelStatus()}")

    result = {f"y{i + 1}": model.variableValue(y[i]) for i in range(n_mach)}
    result.update(
        {f"p{i + 1}{k + 1}": model.variableValue(p[i, k]) for i in range(n_mach) for k in range(n_prod)}
    )
    return model.getObjectiveValue(), result


if __name__ == "__main__":
    objective, solution = solve()
    show("HiGHS（大 M 写法）", objective, solution)
    check(PROBLEM, objective)
