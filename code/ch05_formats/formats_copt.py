"""第五章 文件格式 —— COPT 的模型读写与解文件。

和 Gurobi 的差别只有三点：
1. COPT 没有全局的 cp.read()，必须先建环境、建模型，再 model.read(文件)
2. 求解基底用 .bas，解用 .sol（这两个是只写文件）
3. 参数用 setParam 设置

运行：python code/ch05_formats/formats_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

from common.data import KNAPSACK

OUT = pathlib.Path(__file__).resolve().parent / "output"


def build_knapsack():
    data = KNAPSACK
    n = len(data["weight"])
    env = cp.Envr()
    model = env.createModel("Knapsack")
    model.setParam("Logging", 0)
    x = model.addVars(n, vtype=COPT.BINARY, nameprefix="x")
    model.setObjective(cp.quicksum(data["value"][i] * x[i] for i in range(n)), sense=COPT.MAXIMIZE)
    model.addConstr(
        cp.quicksum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        name="Capacity",
    )
    return env, model


def main():
    OUT.mkdir(exist_ok=True)

    # 1. 写模型文件
    env, model = build_knapsack()
    model.write(str(OUT / "knapsack_copt.lp"))
    model.write(str(OUT / "knapsack_copt.mps"))
    model.solve()

    # 2. 读回模型：没有 cp.read()，用 model.read()
    env2 = cp.Envr()
    reloaded = env2.createModel("Reloaded")
    reloaded.setParam("Logging", 0)
    reloaded.read(str(OUT / "knapsack_copt.mps"))
    reloaded.solve()

    # 3. 解文件与基底文件
    model.write(str(OUT / "knapsack_copt.sol"))
    model.write(str(OUT / "knapsack_copt.bas"))

    print(f"写出模型：{OUT / 'knapsack_copt.lp'}、{OUT / 'knapsack_copt.mps'}")
    print(f"读回模型求得的解：{reloaded.objval:.0f}（原模型 {model.objval:.0f}）")
    print(f"写出解文件：{OUT / 'knapsack_copt.sol'}、{OUT / 'knapsack_copt.bas'}")


if __name__ == "__main__":
    main()
