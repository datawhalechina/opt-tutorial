"""第四章 建模技巧 —— 半连续产量（指示约束写法），用 SCIP（开源）求解。

例题同 scheduling_gurobi.py：机器一旦启用，产量必须不低于下限。
运行：python code/ch04_techniques/scheduling_scip.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import Model, quicksum

from common.data import SCHEDULING
from common.verify import check, show

PROBLEM = "scheduling"


def solve():
    data = SCHEDULING
    n_mach = len(data["fixed_cost"])
    n_prod = len(data["demand"])

    model = Model("Scheduling")
    model.hideOutput()

    y = [model.addVar(vtype="B", name=f"y{i + 1}") for i in range(n_mach)]
    p = {
        (i, k): model.addVar(vtype="C", lb=0, name=f"p{i + 1}{k + 1}")
        for i in range(n_mach)
        for k in range(n_prod)
    }

    model.setObjective(
        quicksum(data["fixed_cost"][i] * y[i] for i in range(n_mach))
        + quicksum(data["cost"][i][k] * p[i, k] for i in range(n_mach) for k in range(n_prod)),
        sense="minimize",
    )

    for k in range(n_prod):
        model.addCons(
            quicksum(p[i, k] for i in range(n_mach)) == data["demand"][k], name=f"demand{k + 1}"
        )
    for i in range(n_mach):
        model.addCons(
            quicksum(p[i, k] for k in range(n_prod)) <= data["capacity"][i] * y[i],
            name=f"capacity{i + 1}",
        )
        # 指示约束：y[i] 取 1 时，后一条不等式必须成立
        model.addConsIndicator(
            quicksum(p[i, k] for k in range(n_prod)) >= data["min_output"][i],
            binvar=y[i],
            activeone=True,
            name=f"min_output{i + 1}",
        )

    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={model.getStatus()}")

    result = {f"y{i + 1}": model.getVal(y[i]) for i in range(n_mach)}
    result.update(
        {f"p{i + 1}{k + 1}": model.getVal(p[i, k]) for i in range(n_mach) for k in range(n_prod)}
    )
    return model.getObjVal(), result


if __name__ == "__main__":
    objective, solution = solve()
    show("SCIP（指示约束）", objective, solution)
    check(PROBLEM, objective)
