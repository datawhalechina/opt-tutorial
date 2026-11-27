"""第 10 章 补充实验 —— 松弛量到底挂在哪：约束界 vs 变量界。

三件事，都是文档里那几条结论的直接证据：

1. 变量界自相矛盾（lb > ub）时，松弛量出现在变量的 RelaxLB 上，不是约束上；
2. `crelax=False` 只允许改变量界时，松弛量出现在 RelaxUB 上；
3. COPT 的可行性松弛**不会往模型里加松弛变量**（前后变量数不变），
   结果只以"界放宽了多少"的形式返回；而 mode 4/5 是二次目标，只支持 LP。

运行：python code/ch10_infeasible/feasrelax_varbounds.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT


def new_model(name):
    env = cp.Envr()
    model = env.createModel(name)
    model.setParam("Logging", 0)
    return model


def show_relaxation(model, title):
    print(f"\n{title}")
    print(f"  FeasRelaxObj = {model.feasRelaxObj:g}")
    for con in model.getConstrs():
        if abs(con.RelaxLB) > 1e-9 or abs(con.RelaxUB) > 1e-9:
            print(f"    约束 {con.name:<10} RelaxLB={con.RelaxLB:+.4f}  RelaxUB={con.RelaxUB:+.4f}")
    for var in model.getVars():
        if abs(var.RelaxLB) > 1e-9 or abs(var.RelaxUB) > 1e-9:
            print(f"    变量 {var.name:<10} RelaxLB={var.RelaxLB:+.4f}  RelaxUB={var.RelaxUB:+.4f}"
                  f"   原界=[{var.lb:g}, {var.ub:g}]")


def case_conflicting_bounds():
    """变量下界 > 上界：模型没有任何约束，仍然不可行。"""
    model = new_model("ConflictingBounds")
    v = model.addVar(lb=10, ub=5, name="v")     # 下界 10 > 上界 5
    model.setObjective(v, sense=COPT.MINIMIZE)
    model.setParam("FeasRelaxMode", 0)
    model.feasRelaxS(True, True)
    show_relaxation(model, "① 变量界自相矛盾（lb=10, ub=5）：")
    print(f"    结论：下界要减 5（RelaxLB=-5 → 新下界 = 10 + (-5) = 5），"
          f"没有任何约束被改动（约束数={len(model.getConstrs())}）")


def case_bound_only():
    """约束 z >= 10 与 z 的上界 5 冲突；只允许松弛变量界（crelax=False）。"""
    model = new_model("BoundOnly")
    z = model.addVar(lb=0, ub=5, name="z")
    model.setObjective(z, sense=COPT.MAXIMIZE)
    model.addConstr(z >= 10, name="need10")
    model.setParam("FeasRelaxMode", 0)
    model.feasRelaxS(vrelax=True, crelax=False)      # 只允许改变量界
    show_relaxation(model, "② 只允许松弛变量界（crelax=False）：")
    print("    结论：上界要加 5（RelaxUB=+5 → 新上界 = 5 + 5 = 10），约束原封不动")


def case_no_extra_vars():
    """COPT 不往模型里加松弛变量：前后变量数、约束数都不变。"""
    model = new_model("NoExtraVars")
    x = model.addVar(lb=0, name="x")
    y = model.addVar(lb=0, name="y")
    model.setObjective(x + y, sense=COPT.MINIMIZE)
    model.addConstr(x + y >= 10, name="c1")
    before = (len(model.getVars()), len(model.getConstrs()))
    model.setParam("FeasRelaxMode", 0)
    model.feasRelaxS(True, True)
    after = (len(model.getVars()), len(model.getConstrs()))
    print(f"\n③ 松弛前后：变量数 {before[0]} → {after[0]}，约束数 {before[1]} → {after[1]}"
          f"（hasFeasRelaxSol={model.hasFeasRelaxSol}）")
    print("    Gurobi 会真的往模型里塞 ArtP_/ArtN_ 变量，COPT 不塞，"
          "只给边界放宽量 —— 这是两家接口最大的差异")


def case_l2_is_lp_only():
    """mode 4/5 是二次目标（平方和），只支持 LP；整数模型直接报错。"""
    model = new_model("LpOnlyMode")
    a = model.addVar(lb=0, vtype=COPT.INTEGER, name="a")
    b = model.addVar(lb=0, vtype=COPT.INTEGER, name="b")
    model.setObjective(40 * a + 50 * b, sense=COPT.MAXIMIZE)
    model.addConstr(a <= 15, name="cap_A")
    model.addConstr(a >= 20, name="order_A")
    model.setParam("FeasRelaxMode", 4)
    try:
        model.feasRelaxS(True, True)
        print("\n④ mode 4 + 整数变量：居然成功了（与预期不符，请复查）")
    except Exception as exc:
        print(f"\n④ mode 4 + 整数变量：{type(exc).__name__}: {exc}")
        print("    结论：L2（平方和）是二次目标，COPT 只对 LP 支持；整数模型请用 mode 0/1/2/3")


def main():
    case_conflicting_bounds()
    case_bound_only()
    case_no_extra_vars()
    case_l2_is_lp_only()


if __name__ == "__main__":
    main()
