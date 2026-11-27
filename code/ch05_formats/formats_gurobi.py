"""第五章 文件格式 —— Gurobi 的模型读写与解文件。

演示四件事：
1. 把模型写成 .lp / .mps，再读回来求解（跨语言交换模型的标准做法）
2. 把解写成 .sol / .mst，读回来当初始解（热启动）
3. 模型不可行时导出 .ilp，用于排查冲突约束
4. 注意三个格式细节：
   - gp.read() 只建模型，读不了解文件；读解要用 model.read()
   - .mst 只保存整数变量的值，不含连续变量；完整解请用 .sol 或 .json
   - .json 是"解的附加信息"，不是模型文件，也不能读回

运行：python code/ch05_formats/formats_gurobi.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

from common.data import KNAPSACK

OUT = pathlib.Path(__file__).resolve().parent / "output"


def build_knapsack():
    data = KNAPSACK
    n = len(data["weight"])
    model = gp.Model("Knapsack")
    model.Params.OutputFlag = 0
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(data["value"][i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(
        gp.quicksum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"],
        name="Capacity",
    )
    return model


def main():
    OUT.mkdir(exist_ok=True)

    # 1. 写模型文件，再读回来
    model = build_knapsack()
    model.write(str(OUT / "knapsack.lp"))
    model.write(str(OUT / "knapsack.mps"))
    model.optimize()

    reloaded = gp.read(str(OUT / "knapsack.lp"))
    reloaded.Params.OutputFlag = 0
    reloaded.optimize()

    # 2. 写解文件：.sol 保存完整解，.mst 只保存整数变量（给 MIP 热启动用）
    model.write(str(OUT / "knapsack.sol"))
    model.write(str(OUT / "knapsack.mst"))

    # 读回解文件必须用 model.read()，gp.read() 只管模型
    warm = gp.read(str(OUT / "knapsack.lp"))
    warm.Params.OutputFlag = 0
    warm.read(str(OUT / "knapsack.sol"))      # 作为 MIP start 导入
    warm.optimize()

    # 3. 不可行模型 + IIS 排查
    bad = gp.Model("Infeasible")
    bad.Params.OutputFlag = 0
    a = bad.addVar(lb=0, name="a")
    b = bad.addVar(lb=0, name="b")
    bad.addConstr(2 * a + b <= 5, name="LIM1")
    bad.addConstr(3 * a >= 10, name="LIM2")
    bad.addConstr(2 * b == 7, name="LIM3")
    bad.optimize()
    if bad.Status == GRB.INFEASIBLE:
        bad.computeIIS()
        bad.write(str(OUT / "conflict.ilp"))

    print(f"写出模型：{OUT / 'knapsack.lp'}、{OUT / 'knapsack.mps'}")
    print(f"重读模型求得的解：{reloaded.ObjVal:.0f}（原模型 {model.ObjVal:.0f}）")
    print(f"写出解文件：{OUT / 'knapsack.sol'}（完整解）、{OUT / 'knapsack.mst'}（只含整数变量）")
    print(f"用 .sol 热启动后求得：{warm.ObjVal:.0f}")
    print(f"不可行模型导出的冲突集：{OUT / 'conflict.ilp'}")

    return model.ObjVal, {v.VarName: v.X for v in model.getVars()}


if __name__ == "__main__":
    objective, solution = main()
    print(f"[Gurobi] 最优目标值 = {objective:.6g}")
    for key, value in solution.items():
        if abs(value) > 1e-9:
            print(f"    {key} = {value:g}")
