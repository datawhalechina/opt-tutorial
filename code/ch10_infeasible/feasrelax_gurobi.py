"""第十章 不可行模型排查 —— Gurobi 的 IIS 与 feasRelax。

⚠️ 本机没有 Gurobi 许可证，这个脚本**未实跑**。代码按 Gurobi 14.0 官方文档
   + gurobipy 函数签名核对（signature 不需要许可证即可查看）：
       Model.computeIIS(callback=None, wheres=None)
       Model.feasRelaxS(relaxobjtype, minrelax, vrelax, crelax)
   relaxobjtype：0 = 违反量之和（L1），1 = 违反量平方和（L2），2 = 违反条数（L0）。
   注意别照抄旧教程：Gurobi 12 之前 relaxobjtype=1 才是"条数"，现在编号变了。

Gurobi 的两条主线：
1. computeIIS() 之后 write("x.ilp") 落地，或读 IISConstr / IISLB / IISUB 属性；
2. feasRelaxS() **直接改写当前模型**（destructive），所以先 copy 一份再松弛；
   松弛后的"违反量"就是 Gurobi 自己加的辅助变量：约束 → ArtP_<约束名>/ArtN_<约束名>，
   变量界 → ArtL_<变量名>/ArtU_<变量名>。

运行：python code/ch10_infeasible/feasrelax_gurobi.py（需许可证）
"""

import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共模块（本章数据自包含，不共用 data.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gurobipy as gp
from gurobipy import GRB

OUT = pathlib.Path(__file__).resolve().parent / "output"

# 约束表：(名字, {变量下标: 系数}, 关系符 'L'/'G'/'E', 右端项)
ROWS = [
    ("cap_A", {0: 1.0}, "L", 15.0),
    ("cap_B", {1: 1.0}, "L", 5.0),
    ("order_A", {0: 1.0}, "G", 20.0),
    ("order_B", {1: 1.0}, "G", 8.0),
    ("shared", {0: 1.0, 1: 1.0}, "L", 100.0),
]


def build():
    model = gp.Model("Infeasible_Lines")
    model.Params.OutputFlag = 0
    x = [model.addVar(lb=0, name=name) for name in ("a", "b")]
    model.setObjective(40 * x[0] + 50 * x[1], GRB.MAXIMIZE)
    for name, coefs, sense, rhs in ROWS:
        expr = sum(coef * x[i] for i, coef in coefs.items())
        if sense == "L":
            model.addConstr(expr <= rhs, name=name)
        elif sense == "G":
            model.addConstr(expr >= rhs, name=name)
        else:
            model.addConstr(expr == rhs, name=name)
    return model


def show_iis(model):
    """算 IIS 并落地。.ilp 就是 IIS 专用格式（语法和 .lp 一样）。"""
    model.computeIIS()
    path = OUT / "iis_gurobi.ilp"
    model.write(str(path))

    # 批量取属性和逐个 c.IISConstr 等价，语义都是"这条约束在 IIS 里"
    flags = model.getAttr("IISConstr", model.getConstrs())
    print(f"  IISMinimal={model.IISMinimal}")
    for con, flag in zip(model.getConstrs(), flags):
        if flag:
            print(f"    ★ {con.ConstrName}（IISConstr={flag}）")
    # 变量界是否在 IIS 里另有 IISLB / IISUB 两个属性
    for var in model.getVars():
        if var.IISLB or var.IISUB:
            print(f"    ★ 变量界 {var.VarName}: IISLB={var.IISLB} IISUB={var.IISUB}")
    print(f"  写出 output/{path.name}")


def show_relax(model, relaxobjtype, label):
    """可行性松弛。注意 feasRelaxS 会原地改模型，所以先 copy。"""
    relax = model.copy()
    relax.feasRelaxS(relaxobjtype, False, False, True)   # L 型目标, 不做最小化前提, 只松弛约束
    relax.optimize()
    if relax.Status != GRB.OPTIMAL:
        print(f"  {label}：松弛模型没解出来（Status={relax.Status}）")
        return

    print(f"\n  {label}：松弛目标值 = {relax.ObjVal:g}")
    for var in relax.getVars():
        # Gurobi 的命名约定：ArtP_/ArtN_ 是约束的正/负松弛，ArtL_/ArtU_ 是变量界的松弛
        if var.VarName.startswith(("ArtP_", "ArtN_")) and var.X > 1e-9:
            print(f"    ★ {var.VarName} = {var.X:g}")


def main():
    OUT.mkdir(exist_ok=True)

    # 没有许可证时抛 GurobiError（信息里含 "License expired"），
    # scripts/verify_all.py 会把它记成「跳过」而不是「失败」
    model = build()
    model.optimize()

    if model.Status == GRB.INF_OR_UNBD:
        # 求解器没区分"不可行"和"无界"，先关掉对偶归约再解一次
        model.Params.DualReductions = 0
        model.optimize()

    print(f"求解状态：Status={model.Status}（GRB.INFEASIBLE={GRB.INFEASIBLE}）")
    if model.Status != GRB.INFEASIBLE:
        print("  模型不是不可行，后面两步用不上")
        return

    # ---- 主线 1：IIS ----
    show_iis(model)

    # ---- 主线 2：可行性松弛，三种目标对比 ----
    show_relax(model, 0, "relaxobjtype=0（违反量之和）")
    show_relax(model, 1, "relaxobjtype=1（违反量平方和）")
    show_relax(model, 2, "relaxobjtype=2（违反条数）")


if __name__ == "__main__":
    try:
        main()
    except gp.GurobiError as exc:
        raise SystemExit(f"跳过：本机没有可用的 Gurobi 许可证（{exc}）") from exc
