"""第六章 日志解读 —— 生成 COPT 的 LP 与 MIP 求解日志。

日志同时打印到屏幕并写入 output/log_copt_lp.log / log_copt_mip.log。
运行：python code/ch06_logging/log_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import FACILITY, LP_PROFIT

OUT = pathlib.Path(__file__).resolve().parent / "output"


def solve_lp():
    data = LP_PROFIT
    env = cp.Envr()
    model = env.createModel("LP_log")
    model.setParam("Logging", 1)
    (OUT / "log_copt_lp.log").unlink(missing_ok=True)   # setLogFile 是追加写，先清掉旧文件
    model.setLogFile(str(OUT / "log_copt_lp.log"))
    a = model.addVar(lb=0, name="A")
    b = model.addVar(lb=0, name="B")
    model.setObjective(data["profit"][0] * a + data["profit"][1] * b, sense=COPT.MAXIMIZE)
    model.addConstr(2 * a + 4 * b <= 100, name="Production_Time")
    model.addConstr(3 * a + b <= 50, name="Raw_Material")
    model.solve()
    return model.objval


def solve_mip():
    data = FACILITY
    n_fac, n_cus = len(data["fixed_cost"]), len(data["demand"])
    big_m = sum(data["demand"])
    env = cp.Envr()
    model = env.createModel("MIP_log")
    model.setParam("Logging", 1)
    (OUT / "log_copt_mip.log").unlink(missing_ok=True)
    model.setLogFile(str(OUT / "log_copt_mip.log"))
    y = model.addVars(n_fac, vtype=COPT.BINARY, nameprefix="y")
    x = model.addVars(n_fac, n_cus, lb=0, vtype=COPT.CONTINUOUS, nameprefix="x")
    model.setObjective(
        cp.quicksum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + cp.quicksum(
            data["transport_cost"][i][j] * x[i, j] for i in range(n_fac) for j in range(n_cus)
        ),
        sense=COPT.MINIMIZE,
    )
    model.addConstrs(
        (cp.quicksum(x[i, j] for i in range(n_fac)) == data["demand"][j] for j in range(n_cus)),
        nameprefix="demand",
    )
    model.addConstrs(
        (cp.quicksum(x[i, j] for j in range(n_cus)) <= big_m * y[i] for i in range(n_fac)),
        nameprefix="open_link",
    )
    model.solve()
    return model.objval


def main():
    OUT.mkdir(exist_ok=True)
    print(f"LP 最优值 {solve_lp():.0f}，MIP 最优值 {solve_mip():.0f}")


if __name__ == "__main__":
    main()
