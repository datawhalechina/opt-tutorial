"""第五章 文件格式 —— COPT 的模型读写与解文件。

和 Gurobi 相比有四个差别：
1. COPT 没有全局的 cp.read()，必须先建环境、建模型，再 model.read(文件)
2. 参数文件后缀是 .par（不是 .prm）
3. IIS 写成 .iis（不是 .ilp）
4. .sol / .mst 可以用 model.read() 读回来当 MIP 初始解；.bas 只对连续 LP 有意义

运行：python code/ch05_formats/formats_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import KNAPSACK, LP_PROFIT

OUT = pathlib.Path(__file__).resolve().parent / "output"


def new_model(name):
    env = cp.Envr()
    model = env.createModel(name)
    model.setParam("Logging", 0)
    return model


def build_knapsack():
    data = KNAPSACK
    n = len(data["weight"])
    model = new_model("Knapsack")
    x = model.addVars(n, vtype=COPT.BINARY, nameprefix="x")
    model.setObjective(cp.quicksum(data["value"][i] * x[i] for i in range(n)), sense=COPT.MAXIMIZE)
    model.addConstr(
        cp.quicksum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        name="Capacity",
    )
    return model


def build_lp():
    data = LP_PROFIT
    model = new_model("Profit")
    a = model.addVar(lb=0, name="A")
    b = model.addVar(lb=0, name="B")
    model.setObjective(
        data["profit"][0] * a + data["profit"][1] * b, sense=COPT.MAXIMIZE
    )
    model.addConstr(
        data["time_per_unit"][0] * a + data["time_per_unit"][1] * b <= data["time_limit"],
        name="Production_Time",
    )
    model.addConstr(
        data["material_per_unit"][0] * a + data["material_per_unit"][1] * b <= data["material_limit"],
        name="Raw_Material",
    )
    return model


def main():
    OUT.mkdir(exist_ok=True)

    # 1. 写模型文件 + 求解 + 写解文件
    model = build_knapsack()
    model.write(str(OUT / "knapsack_copt.lp"))
    model.write(str(OUT / "knapsack_copt.mps"))
    model.solve()
    model.write(str(OUT / "knapsack_copt.sol"))
    model.write(str(OUT / "knapsack_copt.par"))

    # 2. 读回模型（没有 cp.read()，用 model.read()），再把 .sol 当初始解读进来
    reloaded = new_model("Reloaded")
    reloaded.read(str(OUT / "knapsack_copt.mps"))
    reloaded.read(str(OUT / "knapsack_copt.sol"))     # 作为 MIP 初始解
    reloaded.solve()

    # 3. 单纯形基文件：只有连续 LP 才有基，MIP 写出来是空的
    lp_model = build_lp()
    lp_model.solve()
    lp_model.write(str(OUT / "profit_copt.bas"))

    # 4. 不可行模型：算 IIS 并存成 .iis（COPT 的 IIS 后缀不是 .ilp）
    bad = new_model("Infeasible")
    u = bad.addVar(lb=0, name="u")
    v = bad.addVar(lb=0, name="v")
    bad.addConstr(2 * u + v <= 5, name="LIM1")
    bad.addConstr(3 * u >= 10, name="LIM2")
    bad.addConstr(2 * v == 7, name="LIM3")
    bad.solve()
    if bad.status == COPT.INFEASIBLE:
        bad.computeIIS()
        bad.writeIIS(str(OUT / "conflict_copt.iis"))

    print(f"写出模型：{OUT / 'knapsack_copt.lp'}、{OUT / 'knapsack_copt.mps'}")
    print(f"写出解文件：{OUT / 'knapsack_copt.sol'}（可被 model.read 读回当初始解）")
    print(f"写回求解结果：{reloaded.objval:.0f}（原模型 {model.objval:.0f}）")
    print(f"写出基文件：{OUT / 'profit_copt.bas'}（MIP 没有基，所以用 LP 模型演示）")
    print(f"冲突集：{OUT / 'conflict_copt.iis'}")


if __name__ == "__main__":
    main()
