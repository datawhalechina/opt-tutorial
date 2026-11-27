"""第十章 不可行模型排查 —— HiGHS 的可行性松弛（内置接口 + 手工弹性模型）。

先说结论：HiGHS 1.15 起**有**内置的可行性松弛 `feasibilityRelaxation(...)`，
本脚本先跑它，再手工搭一遍弹性模型——手工版适用于 SCIP / PuLP 这类没有内置接口的
求解器，也能让你看清内置接口到底在最小化什么。

手工弹性模型的做法（对每条约束一行）：
把每条约束改成"允许违反一点点"，让求解器去买单。

做法（对每条约束一行）：
    原式  a·x <= b   （>=、== 同理）
    弹性  a·x + sp - sm == b,   sp >= 0, sm >= 0
    违反量：'<=' 行看 sm；'>=' 行看 sp；'==' 行两边都算
    目标  min Σ w_r · 违反量_r        （w_r 就是业务上"这条不能动"的程度）

解出来之后，某个 sp/sm 的取值就是"这条约束被动了多少"，逐个打印即可。

运行：python code/ch10_infeasible/feasrelax_highs.py
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


def build_plain():
    """原模型：已知不可行，先用来说明"无解"。"""
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
    return model


def build_elastic(weights):
    """弹性模型：每条约束一对松弛变量 sp/sm，目标是最小化加权违反量。"""
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    x = [model.addVariable(lb=0, name=name) for name in ("a", "b")]
    sp, sm = {}, {}
    for name, coefs, _sense, rhs in ROWS:
        sp[name] = model.addVariable(lb=0, name=f"sp_{name}")
        sm[name] = model.addVariable(lb=0, name=f"sm_{name}")
        expr = sum(coef * x[i] for i, coef in coefs.items())
        model.addConstr(expr + sp[name] - sm[name] == rhs, f"{name}_elastic")

    # 只有"真正构成违反"的那一侧进目标，否则求解器会两边同时抬起来虚增目标值
    terms = []
    for name, _coefs, sense, _rhs in ROWS:
        weight = weights.get(name, 1.0)
        if sense == "L":
            terms.append(weight * sm[name])
        elif sense == "G":
            terms.append(weight * sp[name])
        else:
            terms.append(weight * (sp[name] + sm[name]))
    model.setObjective(sum(terms), highspy.ObjSense.kMinimize)
    return model, x, sp, sm


def violation(values, name, sense, sp, sm):
    """从解向量里读出这条约束被违反了多少。"""
    if sense == "L":
        return values[sm[name].index]
    if sense == "G":
        return values[sp[name].index]
    return values[sp[name].index] + values[sm[name].index]


def solve_and_report(weights, title, write_lp=False):
    model, x, sp, sm = build_elastic(weights)
    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"弹性模型没解出来：{model.modelStatusToString(model.getModelStatus())}")

    values = model.getSolution().col_value
    print(f"\n{title}")
    print(f"  加权违反量（目标值）= {model.getObjectiveValue():g}")
    print(f"  解：a = {values[x[0].index]:g}，b = {values[x[1].index]:g}")

    changed = 0
    for name, _coefs, sense, rhs in ROWS:
        amount = violation(values, name, sense, sp, sm)
        mark = "★ 被改" if amount > 1e-9 else ""
        if amount > 1e-9:
            changed += 1
        print(f"    {name:8s} 原右端项 {rhs:6g}   违反量 {amount:6g}  {mark}")
    print(f"  共改动 {changed} 条约束")

    if write_lp:
        path = OUT / "feasrelax_highs.lp"
        model.writeModel(str(path))
        print(f"  弹性模型已写出：output/{path.name}")
    return model


def solve_builtin():
    """HiGHS 1.15+ 的内置可行性松弛：一次调用，结果在 getSolution() 里。"""
    model = build_plain()
    model.run()
    status = model.feasibilityRelaxation(1.0, 1.0, 1.0)   # 变量下界/上界/约束 RHS 的全局惩罚
    values = model.getSolution().col_value
    print("\n【内置】feasibilityRelaxation(1, 1, 1)")
    print(f"  返回 {status}，加权违反量（目标值）= {model.getObjectiveValue():g}")
    print(f"  解：a = {values[0]:g}，b = {values[1]:g}"
          f"  → 改的是 order_A(5) 与 cap_B(3)，和 COPT FeasRelaxMode=0 同一组")
    return model


def main():
    OUT.mkdir(exist_ok=True)

    original = build_plain()
    original.run()
    print(f"原模型状态：{original.modelStatusToString(original.getModelStatus())}")

    # 先看内置接口：一行调用，总违反量 8，修复集与 COPT 一致
    solve_builtin()

    # 等权：认为改产能和改订单一样贵 → 挑违反量最小的组合
    solve_and_report({}, "等权（所有约束惩罚都是 1）", write_lp=True)

    # 订单不能动：合同违约的惩罚调成 10 倍 → 求解器改去动产能
    solve_and_report(
        {"order_A": 10.0, "order_B": 10.0},
        "订单优先（order_* 的惩罚 = 10，cap_* 仍是 1）",
    )


if __name__ == "__main__":
    main()
