"""第五章 文件格式 —— HiGHS（开源）的模型读写与解文件。

HiGHS 的写法：
  model.writeModel("xxx.lp")   写模型（.lp / .mps）
  model.readModel("xxx.mps")   读模型
  model.writeSolution("xxx.sol", 0)  写解（第二个参数 0 = 原始格式）

运行：python code/ch05_formats/formats_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import KNAPSACK

OUT = pathlib.Path(__file__).resolve().parent / "output"


def build_knapsack():
    data = KNAPSACK
    n = len(data["weight"])
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    x = [
        model.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger, name=f"x{i + 1}")
        for i in range(n)
    ]
    model.addConstr(
        sum(data["weight"][i] * x[i] for i in range(n)) <= data["capacity"], "Capacity"
    )
    model.maximize(sum(data["value"][i] * x[i] for i in range(n)))
    return model


def main():
    OUT.mkdir(exist_ok=True)

    model = build_knapsack()
    model.writeModel(str(OUT / "knapsack_highs.lp"))
    model.writeModel(str(OUT / "knapsack_highs.mps"))
    model.run()

    reloaded = highspy.Highs()
    reloaded.setOptionValue("output_flag", False)
    reloaded.readModel(str(OUT / "knapsack_highs.mps"))
    reloaded.run()

    model.writeSolution(str(OUT / "knapsack_highs.sol"), 0)

    print(f"写出模型：{OUT / 'knapsack_highs.lp'}、{OUT / 'knapsack_highs.mps'}")
    print(f"读回模型求得的目标值：{reloaded.getObjectiveValue():.0f}")
    print(f"写出解文件：{OUT / 'knapsack_highs.sol'}")


if __name__ == "__main__":
    main()
