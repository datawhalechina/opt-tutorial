"""第十章 不可行模型排查 —— HiGHS 的 IIS（开源，免授权）。

HiGHS 1.16 起有 IIS 接口，但有两个坑：
1. getIis() 返回的是**二元组** (HighsStatus, HighsIis)，不是单个对象；
2. 默认的 "light" 策略对纯约束型冲突可能返回**空 IIS**（rows=0），
   必须显式把 iis_strategy 设成 kIisStrategyIrreducible 才能拿到真正的冲突集。

运行：python code/ch10_infeasible/iis_highs.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共模块（本章数据自包含，不共用 data.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

OUT = pathlib.Path(__file__).resolve().parent / "output"

# 约束表：(名字, {变量下标: 系数}, 关系符 'L'/'G'/'E', 右端项)
ROWS = [
    ("cap_A", {0: 1.0}, "L", 15.0),
    ("cap_B", {1: 1.0}, "L", 5.0),
    ("order_A", {0: 1.0}, "G", 20.0),
    ("order_B", {1: 1.0}, "G", 8.0),
    ("shared", {0: 1.0, 1: 1.0}, "L", 100.0),
]

# row_status_ 的取值含义
ROW_STATUS = {-1: "不在冲突里", 0: "可能在冲突里", 1: "★ 在冲突里"}

# IIS 子模型里，某一列保留哪一侧的界
COL_BOUND = {
    highspy.IisBoundStatus.kIisBoundStatusFree: "界被放开（free）",
    highspy.IisBoundStatus.kIisBoundStatusLower: "只保留下界",
    highspy.IisBoundStatus.kIisBoundStatusUpper: "只保留上界",
    highspy.IisBoundStatus.kIisBoundStatusBoxed: "上下界都保留",
}


def build():
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    x = [model.addVariable(lb=0, name=name) for name in ("a", "b")]
    model.setObjective(40 * x[0] + 50 * x[1], highspy.ObjSense.kMaximize)
    for name, coefs, sense, rhs in ROWS:
        expr = sum(coef * x[i] for i, coef in coefs.items())
        if sense == "L":
            model.addConstr(expr <= rhs, name)
        elif sense == "G":
            model.addConstr(expr >= rhs, name)
        else:
            model.addConstr(expr == rhs, name)
    return model, x


def main():
    OUT.mkdir(exist_ok=True)
    model, _ = build()
    model.run()
    print(f"求解状态：{model.modelStatusToString(model.getModelStatus())}")

    # ---- 坑 1：默认 light 策略，本例返回空 IIS ----
    status, light = model.getIis()
    print(f"默认策略：status={status}，rows={len(light.row_index_)}，cols={len(light.col_index_)}")

    # ---- 换成 irreducible 策略，再算一次 ----
    model.setOptionValue("iis_strategy", highspy.IisStrategy.kIisStrategyIrreducible)
    status, iis = model.getIis()
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"getIis 失败：{status}")

    lp = model.getLp()
    print(f"irreducible 策略：rows={len(iis.row_index_)}，cols={len(iis.col_index_)}")
    for j, name in enumerate(lp.row_names_):
        print(f"    {name:8s} {ROW_STATUS[iis.row_status_[j]]}")
    # col_index_ / col_bound_ 是**并列**的两个数组（都按 IIS 内部顺序），
    # 不是"变量界在冲突里"，而是"这一列在 IIS 子模型里保留了哪一侧的界"。
    for k, col in enumerate(iis.col_index_):
        label = COL_BOUND[highspy.IisBoundStatus(int(iis.col_bound_[k]))]
        print(f"    列 {lp.col_names_[col]:5s} {label}")

    # ---- 落盘：IIS 单独写成一个 LP，方便肉眼核对 ----
    path = OUT / "iis_highs.lp"
    model.writeIisModel(str(path))
    print(f"写出 output/{path.name}：")
    for line in path.read_text().splitlines():
        if not line.startswith("\\"):
            print(f"    {line}")


if __name__ == "__main__":
    main()
