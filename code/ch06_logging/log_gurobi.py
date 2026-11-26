"""第六章 日志解读 —— 生成 Gurobi 的 LP 与 MIP 求解日志。

同时把日志打印到屏幕和写入文件 output/log_gurobi_lp.log / log_gurobi_mip.log，
方便对照第五章（原文第六章）的解读。
运行：python code/ch06_logging/log_gurobi.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import FACILITY, LP_PROFIT

OUT = pathlib.Path(__file__).resolve().parent / "output"


def solve_lp():
    data = LP_PROFIT
    model = gp.Model("LP_log")
    model.Params.LogFile = str(OUT / "log_gurobi_lp.log")
    model.Params.OutputFlag = 1
    a = model.addVar(lb=0, name="A")
    b = model.addVar(lb=0, name="B")
    model.setObjective(data["profit"][0] * a + data["profit"][1] * b, GRB.MAXIMIZE)
    model.addConstr(2 * a + 4 * b <= 100, name="Production_Time")
    model.addConstr(3 * a + b <= 50, name="Raw_Material")
    model.optimize()
    return model.ObjVal


def solve_mip():
    data = FACILITY
    n_fac, n_cus = len(data["fixed_cost"]), len(data["demand"])
    big_m = sum(data["demand"])
    model = gp.Model("MIP_log")
    model.Params.LogFile = str(OUT / "log_gurobi_mip.log")
    model.Params.OutputFlag = 1
    y = model.addVars(n_fac, vtype=GRB.BINARY, name="y")
    x = model.addVars(n_fac, n_cus, lb=0, name="x")
    model.setObjective(
        gp.quicksum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + gp.quicksum(
            data["transport_cost"][i][j] * x[i, j] for i in range(n_fac) for j in range(n_cus)
        ),
        GRB.MINIMIZE,
    )
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(n_fac)) == data["demand"][j] for j in range(n_cus)),
        name="demand",
    )
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in range(n_cus)) <= big_m * y[i] for i in range(n_fac)),
        name="open_link",
    )
    model.optimize()
    return model.ObjVal


def main():
    OUT.mkdir(exist_ok=True)
    print(f"LP 最优值 {solve_lp():.0f}，MIP 最优值 {solve_mip():.0f}")


if __name__ == "__main__":
    main()
