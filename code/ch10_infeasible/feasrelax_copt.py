"""第十章 不可行模型排查 —— COPT 的可行性松弛（FeasRelax）。

feasRelax 回答的问题不是"哪几条约束打架"，而是"最少改多少，模型就可行"。
COPT 的 Python 简化版接口只要两个开关：
    model.feasRelaxS(vrelax, crelax)     是否允许改变量界 / 是否允许改约束

关键点（实测，COPT 8.0.1）：
1. 结果不在 objval 里——原模型仍然不可行，连 a.x 都会报 INVALID；
   答案在属性 HasFeasRelaxSol / FeasRelaxObj 里；
2. 每条约束被改了多少，用 Information RelaxLB / RelaxUB 查；
   口径是"新界 - 原界"，所以**松弛下界时 RelaxLB 是负数**；
3. 改哪个目标由参数 FeasRelaxMode 控制，同一个模型会给出完全不同的"最少改动"；
4. writeRelax 把改完的模型写成 .relax 文件；要拿松弛后变量的取值，
   就把改动落回模型再解一次（本脚本最后一段演示）。

运行：python code/ch10_infeasible/feasrelax_copt.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共模块（本章数据自包含，不共用 data.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import coptpy as cp
from coptpy import COPT

OUT = pathlib.Path(__file__).resolve().parent / "output"

# 约束表：(名字, {变量下标: 系数}, 关系符, 右端项)
ROWS = [
    ("cap_A", {0: 1.0}, COPT.LESS_EQUAL, 15.0),
    ("cap_B", {1: 1.0}, COPT.LESS_EQUAL, 5.0),
    ("order_A", {0: 1.0}, COPT.GREATER_EQUAL, 20.0),
    ("order_B", {1: 1.0}, COPT.GREATER_EQUAL, 8.0),
    ("shared", {0: 1.0, 1: 1.0}, COPT.LESS_EQUAL, 100.0),
]

# FeasRelaxMode 的含义（官方文档 8.0）
MODE_MEANING = {
    0: "违反量之和",
    1: "违反量之和最小前提下，优化原目标",
    2: "违反条数",
    3: "违反条数最小前提下，优化原目标",
    4: "违反量平方和",
    5: "违反量平方和最小前提下，优化原目标",
}


def build(env, mode):
    model = env.createModel(f"FeasRelax_mode{mode}")
    model.setParam("Logging", 0)
    model.setParam("FeasRelaxMode", mode)
    a = model.addVar(lb=0, name="a")
    b = model.addVar(lb=0, name="b")
    x = [a, b]
    model.setObjective(40 * a + 50 * b, sense=COPT.MAXIMIZE)
    for name, coefs, sense, rhs in ROWS:
        expr = cp.quicksum(coef * x[i] for i, coef in coefs.items())
        model.addConstr(expr, sense, rhs, name=name)
    return model


def relax_changes(model):
    """返回 {约束名: (新下界 - 原下界, 新上界 - 原上界)}，只保留真的动过的。"""
    changes = {}
    for con in model.getConstrs():
        delta_lb = model.getInfo("RelaxLB", con)
        delta_ub = model.getInfo("RelaxUB", con)
        if abs(delta_lb) > 1e-6 or abs(delta_ub) > 1e-6:
            changes[con.name] = (delta_lb, delta_ub)
    return changes


def report(model, title):
    """打印总代价 + 每条约束被改了多少。"""
    print(f"\n{title}")
    print(f"  HasFeasRelaxSol={model.getAttr('HasFeasRelaxSol')} "
          f"FeasRelaxObj={model.getAttr('FeasRelaxObj'):g}")
    for con in model.getConstrs():
        delta_lb = model.getInfo("RelaxLB", con)
        delta_ub = model.getInfo("RelaxUB", con)
        if abs(delta_lb) < 1e-6 and abs(delta_ub) < 1e-6:
            continue
        print(f"    ★ {con.name:8s} 原界 [{con.lb:g}, {con.ub:g}] → "
              f"[{con.lb + delta_lb:g}, {con.ub + delta_ub:g}]"
              f"   违反量 = {abs(delta_lb) + abs(delta_ub):g}")


def changes_text(changes):
    """把 {(d_lb, d_ub)} 写成人能读的一句话：改了哪一侧、多少。"""
    parts = []
    for name, (delta_lb, delta_ub) in changes.items():
        if abs(delta_ub) > 1e-6:
            parts.append(f"{name}(上界+{delta_ub:g})")
        else:
            parts.append(f"{name}(下界{delta_lb:+g})")
    return ", ".join(parts)


def main():
    OUT.mkdir(exist_ok=True)
    env = cp.Envr()

    # ---- 1. 单个 mode 的细节：mode 0（默认，最小化违反量之和）----
    model = build(env, 0)
    model.solve()
    print(f"原模型状态：{model.status}（COPT.INFEASIBLE={COPT.INFEASIBLE}）")
    model.feasRelaxS(True, True)          # vrelax=True, crelax=True
    report(model, "FeasRelaxMode=0（最小化违反量之和，L1）")

    relax_path = OUT / "feasrelax_copt.relax"
    model.writeRelax(str(relax_path))
    print(f"\n  写出 output/{relax_path.name}，可以逐行核对改完的界：")
    for line in relax_path.read_text().splitlines():
        if line.strip() and not line.startswith("\\"):
            print(f"    {line}")

    # ---- 2. 六个 mode 扫一遍：同一个模型，"最少改动"的答案不止一个 ----
    print("\n六个 FeasRelaxMode 的对比（FeasRelaxObj / 被改的约束）：")
    for mode in range(6):
        m = build(env, mode)
        m.solve()
        m.feasRelaxS(True, True)
        changes = relax_changes(m)
        print(f"  mode {mode}（{MODE_MEANING[mode]}）："
              f"FeasRelaxObj={m.getAttr('FeasRelaxObj'):g}  {changes_text(changes)}")

    # ---- 3. 想把"改完之后到底多少产量"算出来，就把改动落回模型再解一次 ----
    m = build(env, 0)
    m.solve()
    m.feasRelaxS(True, True)
    delta = relax_changes(m)
    repaired = build(env, 0)
    for con in repaired.getConstrs():
        if con.name in delta:
            d_lb, d_ub = delta[con.name]
            con.lb = con.lb + d_lb
            con.ub = con.ub + d_ub
    repaired.solve()
    print(f"\n把 mode 0 的松弛量落回模型后求解：目标值 = {repaired.objval:.0f}")
    for var in repaired.getVars():
        print(f"    {var.name} = {var.x:g}")


if __name__ == "__main__":
    main()
