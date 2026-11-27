"""第十章 不可行模型排查 —— COPT 的 IIS（不可约不一致子系统）。

例题：两条产线，各自有一对"产能上限 vs 合同交货下限"的冲突：
    cap_A:   a <= 15      产线 A 最多 15 件
    order_A: a >= 20      合同要求 A 至少 20 件
    cap_B:   b <= 5
    order_B: b >= 8
    shared:  a + b <= 100 共享原料，和冲突无关
两组冲突互不相关，所以模型里有**两个独立的 IIS**，而 COPT 一次只返回一个。
修掉第一个再算一次，才能看到第二个——这就是实操里最常见的"修不完"的原因。

运行：python code/ch10_infeasible/iis_copt.py
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

# 修冲突的方案：把订单下限降到产能上限，超出产能的部分外协
REPAIR = {"order_A": 15.0, "order_B": 5.0}

# 原始右端项，打印"从多少改到多少"用
ORIGINAL_RHS = {name: rhs for name, _, _, rhs in ROWS}


def build(env, override=None):
    """建模。override 用来模拟"修掉某条约束后再看下一轮"。"""
    override = override or {}
    model = env.createModel("Infeasible_Lines")
    model.setParam("Logging", 0)
    a = model.addVar(lb=0, name="a")
    b = model.addVar(lb=0, name="b")
    x = [a, b]
    model.setObjective(40 * a + 50 * b, sense=COPT.MAXIMIZE)
    for name, coefs, sense, rhs in ROWS:
        # COPT 的三段式写法：addConstr(表达式, sense, 右端项, name=...)
        expr = cp.quicksum(coef * x[i] for i, coef in coefs.items())
        model.addConstr(expr, sense, override.get(name, rhs), name=name)
    return model


def main():
    OUT.mkdir(exist_ok=True)
    env = cp.Envr()
    override = {}

    for round_no in range(1, 4):
        model = build(env, override)
        model.solve()

        if model.status == COPT.OPTIMAL:
            print(f"\n第 {round_no} 轮：已经是可行模型，目标值 = {model.objval:.0f}")
            for var in model.getVars():
                print(f"    {var.name} = {var.x:g}")
            break

        print(f"\n第 {round_no} 轮：不可行（status={model.status}，COPT.INFEASIBLE={COPT.INFEASIBLE}）")

        # ---- 1. 算 IIS ----
        model.computeIIS()

        # ---- 2. 看 IIS 的规模：有几条约束、几个变量界在冲突里 ----
        print(
            f"  HasIIS={model.getAttr('HasIIS')} "
            f"IsMinIIS={model.getAttr('IsMinIIS')} "
            f"IISRows={model.getAttr('IISRows')} "
            f"IISCols={model.getAttr('IISCols')}"
        )

        # ---- 3. 逐条查 IIS 标记：iislb 命中下界，iisub 命中上界 ----
        in_iis = []
        for con in model.getConstrs():
            if con.iislb or con.iisub:
                in_iis.append(con.name)
                print(
                    f"    ★ {con.name:8s} 界=[{con.lb:g}, {con.ub:g}] "
                    f"iislb={con.iislb} iisub={con.iisub}"
                )
        for var in model.getVars():
            if var.iislb or var.iisub:
                print(f"    ★ 变量界 {var.name}: iislb={var.iislb} iisub={var.iisub}")

        # ---- 4. 落盘：后缀必须是 .iis ----
        path = OUT / f"iis_copt_round{round_no}.iis"
        model.writeIIS(str(path))
        print(f"  写出 output/{path.name}")
        if round_no == 1:
            print("  ---- 文件内容（不含冲突的 shared 已被剔除）----")
            for line in path.read_text().splitlines():
                if line.strip() and not line.startswith("\\"):
                    print(f"    {line}")

        # ---- 5. 修掉一条，再来一轮 ----
        target = sorted(name for name in in_iis if name in REPAIR)[0]
        new_rhs = REPAIR[target]
        override[target] = new_rhs
        print(f"  修复：{target} 的下限 {ORIGINAL_RHS[target]:g} → {new_rhs:g}"
              "（超出产能的订单外协）")


if __name__ == "__main__":
    main()
