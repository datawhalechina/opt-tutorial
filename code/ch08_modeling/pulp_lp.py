"""第八章 建模层 —— 用 PuLP 建模，后端可选（CBC / HiGHS / COPT / ...）。

同一份模型代码，只改 solve() 里的后端参数，就能换求解器。
默认优先 HiGHS（需 pip install highspy），没装就自动退回 PuLP 内置的 CBC。
运行：python code/ch08_modeling/pulp_lp.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pulp

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def solve(backend=None):
    data = LP_PROFIT
    # 默认用 HiGHS（需要 pip install highspy）；没装就退回 PuLP 内置的 CBC
    if backend is None:
        highs = pulp.HiGHS(msg=False)
        backend = highs if highs.available() else pulp.PULP_CBC_CMD(msg=False)   # 换成 pulp.PULP_CBC_CMD(msg=False) 就是 CBC

    model = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)
    a = pulp.LpVariable("A", lowBound=0)
    b = pulp.LpVariable("B", lowBound=0)

    model += data["profit"][0] * a + data["profit"][1] * b
    model += (
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        "Production_Time",
    )
    model += (
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        "Raw_Material",
    )

    model.solve(backend)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"没有求到最优解，模型状态={pulp.LpStatus[model.status]}")

    return pulp.value(model.objective), {v.name: v.value() for v in model.variables()}


if __name__ == "__main__":
    objective, solution = solve()
    show("PuLP", objective, solution)
    check(PROBLEM, objective)
