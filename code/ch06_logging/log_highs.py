"""第六章 日志解读 —— 生成 HiGHS（开源）的 LP 与 MIP 求解日志。

HiGHS 用 setOptionValue("log_file", 路径) 写日志文件。
运行：python code/ch06_logging/log_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import FACILITY, LP_PROFIT

OUT = pathlib.Path(__file__).resolve().parent / "output"


def solve_lp():
    data = LP_PROFIT
    model = highspy.Highs()
    model.setOptionValue("output_flag", True)
    a = model.addVariable(lb=0, name="A")
    b = model.addVariable(lb=0, name="B")
    # 用 setObjective 而不是 maximize()：后者会立刻求解一次，日志里会多一段空模型
    model.setObjective(
        data["profit"][0] * a + data["profit"][1] * b, highspy.ObjSense.kMaximize
    )
    model.addConstr(2 * a + 4 * b <= 100, "Production_Time")
    model.addConstr(3 * a + b <= 50, "Raw_Material")
    (OUT / "log_highs_lp.log").unlink(missing_ok=True)   # log_file 是追加写
    model.setOptionValue("log_file", str(OUT / "log_highs_lp.log"))
    model.run()
    return model.getObjectiveValue()


def solve_mip():
    data = FACILITY
    n_fac, n_cus = len(data["fixed_cost"]), len(data["demand"])
    big_m = sum(data["demand"])
    model = highspy.Highs()
    model.setOptionValue("output_flag", True)
    y = [
        model.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger, name=f"y{i + 1}")
        for i in range(n_fac)
    ]
    x = {
        (i, j): model.addVariable(lb=0, name=f"x{i + 1}{j + 1}")
        for i in range(n_fac)
        for j in range(n_cus)
    }
    model.setObjective(
        sum(data["fixed_cost"][i] * y[i] for i in range(n_fac))
        + sum(
            data["transport_cost"][i][j] * x[i, j]
            for i in range(n_fac)
            for j in range(n_cus)
        ),
        highspy.ObjSense.kMinimize,
    )
    for j in range(n_cus):
        model.addConstr(sum(x[i, j] for i in range(n_fac)) == data["demand"][j], f"demand{j + 1}")
    for i in range(n_fac):
        model.addConstr(sum(x[i, j] for j in range(n_cus)) <= big_m * y[i], f"open_link{i + 1}")
    (OUT / "log_highs_mip.log").unlink(missing_ok=True)
    model.setOptionValue("log_file", str(OUT / "log_highs_mip.log"))
    model.run()
    return model.getObjectiveValue()


def main():
    OUT.mkdir(exist_ok=True)
    print(f"LP 最优值 {solve_lp():.0f}，MIP 最优值 {solve_mip():.0f}")


if __name__ == "__main__":
    main()
