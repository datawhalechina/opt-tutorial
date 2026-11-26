"""第一章 线性规划 —— 用 CPLEX（docplex）求解。

例题：工厂生产 A、B 两种产品，求利润最大化的产量。
运行：python code/ch01_lp/profit_cplex.py

说明：docplex 是本仓库推荐的 CPLEX Python 接口，建模语法比原生 cplex 模块简单，
且导出 LP 文件不需要许可证；真正调用 CPLEX 引擎求解才需要有 CPLEX 运行时。
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docplex.mp.model import Model

from common.data import LP_PROFIT
from common.verify import check, show

PROBLEM = "lp_profit"


def solve():
    data = LP_PROFIT
    profit_a, profit_b = data["profit"]

    # 1. 建模型
    model = Model("Maximize_Profit")

    # 2. 建变量
    a = model.continuous_var(lb=0, name="A")
    b = model.continuous_var(lb=0, name="B")

    # 3. 目标函数
    model.maximize(profit_a * a + profit_b * b)

    # 4. 约束
    model.add_constraint(
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        ctname="Production_Time",
    )
    model.add_constraint(
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        ctname="Raw_Material",
    )

    # 5. 求解（这一步需要本机装有 CPLEX 运行时）
    solution = model.solve(log_output=False)
    if solution is None:
        raise RuntimeError(f"没有求到最优解：{model.get_solve_status()}")

    # 6. 取解
    return solution.objective_value, {v.name: solution.get_value(v) for v in model.iter_variables()}


if __name__ == "__main__":
    objective, solution = solve()
    show("CPLEX", objective, solution)
    check(PROBLEM, objective)
