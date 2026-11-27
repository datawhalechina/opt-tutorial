"""第十一章 根松弛与回调 —— 用 HiGHS（开源）求解，检验它的回调能给什么。

习题同为第 7 章的 CVRP（MTZ 写法）。HiGHS 1.16 的回调接口和别家不一样：
  - 注册方式是订阅事件：h.cbMipSolution += fn（不是 setCallback(fn, data)）
  - 回调函数收到一个 HighsCallbackEvent：e.callback_type / e.message / e.data_out / e.data_in
  - data_out 里有 mip_solution / mip_primal_bound / mip_dual_bound / mip_node_count
  - **没有"节点 LP 松弛"事件**，也没有取 LP 松弛解的接口 → 根松弛界要用单独解 LP 的方式拿

脚本会跑三次求解：一次 LP 松弛、两次同样的 MIP（分别用旧写法和新写法注册回调），
好让"哪种写法真的会触发"有对照。
运行：python code/ch11_callback/rootrelax_highs.py
"""

import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import highspy

from common.data import CVRP
from common.verify import check, show

PROBLEM = "cvrp"


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def build_model(binary):
    """同一个 CVRP 模型，binary=False 时只要 LP 松弛。返回 (model, x, y)。"""
    data = CVRP
    demand, capacity = data["demand"], data["capacity"]
    nodes = range(len(demand))
    customers = range(1, len(demand))
    dist = build_distance(data["coord"])

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    vtype = highspy.HighsVarType.kInteger if binary else highspy.HighsVarType.kContinuous
    x = {
        (i, j): model.addVariable(lb=0, ub=1, type=vtype, name=f"x{i}_{j}")
        for (i, j) in dist
    }
    u = {i: model.addVariable(lb=0, name=f"u{i}") for i in customers}

    model.setObjective(
        sum(dist[i, j] * x[i, j] for (i, j) in dist), highspy.ObjSense.kMinimize
    )

    for i in customers:
        model.addConstr(
            sum(x[i, j] for j in nodes if j != i) == 1, f"out_{i}"
        )
    for j in customers:
        model.addConstr(
            sum(x[i, j] for i in nodes if i != j) == 1, f"in_{j}"
        )
    for i in customers:
        model.addConstr(u[i] >= demand[i], f"load_lb_{i}")
        model.addConstr(u[i] <= capacity, f"load_ub_{i}")
    for i in customers:
        for j in customers:
            if i != j:
                model.addConstr(
                    u[j] >= u[i] + demand[j] - capacity * (1 - x[i, j]),
                    f"mtz_{i}_{j}",
                )
    model.addConstr(
        sum(x[0, j] for j in customers) == sum(x[i, 0] for i in customers),
        "depot_balance",
    )
    return model, x, u


def solve_lp_relaxation():
    model, x, _ = build_model(binary=False)
    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"LP 松弛没解出来，status={model.getModelStatus()}")
    fractional = {
        (i, j): model.variableValue(x[i, j])
        for (i, j) in x
        if 1e-9 < model.variableValue(x[i, j]) < 1 - 1e-9
    }
    return model.getObjectiveValue(), fractional


class MipRecorder:
    """把回调事件记下来：每种事件几次、第一条 MIP 解、以及当时的上/下界。"""

    # 回调类型编号 → 名字（常量在 highspy._core.cb.HighsCallbackType 里，顶层没导出）
    NAMES = {3: "MipSolution", 4: "MipImprovingSolution", 5: "MipLogging", 7: "MipGetCutPool"}

    def __init__(self):
        self.counts = {}
        self.first = None      # (解目标, primal_bound, dual_bound, node)

    def __call__(self, event):
        name = self.NAMES.get(int(event.callback_type), str(event.callback_type))
        self.counts[name] = self.counts.get(name, 0) + 1
        if self.first is None:
            out = event.data_out
            self.first = (
                out.objective_function_value,
                out.mip_primal_bound,
                out.mip_dual_bound,
                out.mip_node_count,
            )


def solve_mip_old_style(x_vars):
    """旧写法：setCallback(五参函数, user_data)。1.16 里它只是注册“转发器”，还得先 startCallback。"""
    model, x, _ = build_model(binary=True)
    hits = {"n": 0}

    def raw_callback(callback_type, message, data_out, data_in, user_data):
        hits["n"] += 1

    model.setCallback(raw_callback, None)
    model.run()
    return model.getObjectiveValue(), hits["n"]


def solve_mip_new_style():
    """新写法：订阅具体事件。回调用 event.data_out 取信息。"""
    model, x, _ = build_model(binary=True)
    recorder = MipRecorder()
    model.cbMipSolution += recorder
    model.cbMipImprovingSolution += recorder
    model.cbMipLogging += recorder
    model.run()
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"没有求到最优解，status={model.getModelStatus()}")

    solution = {f"x{i}_{j}": model.variableValue(x[i, j]) for (i, j) in x}
    info = model.getInfo()
    return model.getObjectiveValue(), solution, recorder, info.mip_node_count


def main():
    lp_obj, fractional = solve_lp_relaxation()
    print(f"[HiGHS] 根 LP 松弛界       = {lp_obj:.6g}")
    print(f"[HiGHS] LP 解里的分数弧    = {len(fractional)} 条")

    old_obj, old_hits = solve_mip_old_style(None)
    mip_obj, solution, recorder, nodes = solve_mip_new_style()

    print(f"[HiGHS] 旧写法 setCallback 触发 = {old_hits} 次（结果是 {old_obj:.6g}）")
    print(f"[HiGHS] 新写法 cbMipSolution 等触发 = {recorder.counts}")
    if recorder.first:
        cand_obj, primal, dual, node = recorder.first
        print(f"[HiGHS] 第一条 MIP 回调: 解={cand_obj:.6g} 上界={primal!r} 下界={dual!r} 节点={node}")
    else:
        print("[HiGHS] 一次 MIP 回调都没触发")
    print(f"[HiGHS] 整数最优           = {mip_obj:.6g}（搜索节点 {nodes} 个）")
    print(f"[HiGHS] 根节点 Gap          = {(mip_obj - lp_obj) / abs(mip_obj) * 100:.2f}%")
    print("[HiGHS] 结论：回调里拿不到节点 LP 松弛解，只能用 mip_dual_bound / 单独解 LP")

    show("HiGHS", mip_obj, solution)
    check(PROBLEM, mip_obj)


if __name__ == "__main__":
    main()
