"""第十章 不可行模型排查 —— SCIP 的 IIS（开源，免授权）。

SCIP 的 IIS 接口最"轻"：generateIIS() 返回一个 IIS 对象，
真正的冲突约束要去它的子 SCIP（getSubscip()）里取。

运行：python code/ch10_infeasible/iis_scip.py
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共模块（本章数据自包含，不共用 data.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import Model

# 约束表：(名字, {变量下标: 系数}, 关系符 'L'/'G'/'E', 右端项)
ROWS = [
    ("cap_A", {0: 1.0}, "L", 15.0),
    ("cap_B", {1: 1.0}, "L", 5.0),
    ("order_A", {0: 1.0}, "G", 20.0),
    ("order_B", {1: 1.0}, "G", 8.0),
    ("shared", {0: 1.0, 1: 1.0}, "L", 100.0),
]


def build():
    model = Model("Infeasible_Lines")
    model.hideOutput()
    x = [model.addVar(name, lb=0) for name in ("a", "b")]
    model.setObjective(40 * x[0] + 50 * x[1], "maximize")
    for name, coefs, sense, rhs in ROWS:
        expr = sum(coef * x[i] for i, coef in coefs.items())
        if sense == "L":
            model.addCons(expr <= rhs, name=name)
        elif sense == "G":
            model.addCons(expr >= rhs, name=name)
        else:
            model.addCons(expr == rhs, name=name)
    return model


def main():
    model = build()
    model.optimize()
    print(f"求解状态：{model.getStatus()}")

    # ---- 1. 生成 IIS ----
    iis = model.generateIIS()
    print(f"IIS 节点数={iis.getNNodes()} 耗时={iis.getTime():.2f}s")
    print(f"子问题不可行：{iis.isSubscipInfeasible()}，已证不可约：{iis.isSubscipIrreducible()}")

    # ---- 2. 冲突约束在子 SCIP 里 ----
    subscip = iis.getSubscip()
    print(f"冲突约束（{len(subscip.getConss())} 条）：")
    for cons in subscip.getConss():
        print(f"    ★ {cons.name}（{cons.getConshdlrName()} 约束）")
    print(f"子问题涉及变量：{[var.name for var in subscip.getVars()]}")


if __name__ == "__main__":
    main()
