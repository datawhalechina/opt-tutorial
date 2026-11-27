"""第十一章 根松弛与回调 —— 用 SCIP（PySCIPOpt，开源）求解，用事件处理器看求解过程。

习题同为第 7 章的 CVRP（MTZ 写法）。SCIP 没有"回调函数"这个概念，只有事件处理器：
  - 继承 pyscipopt.Eventhdlr，实现 eventinit（在里面 catchEvent 订阅）和 eventexec
  - model.includeEventhdlr(handler, "名字", "说明") 挂上去
  - 常用事件：FIRSTLPSOLVED / LPSOLVED / NODEFOCUSED / BESTSOLFOUND / NODESOLVED
  - 事件对象只给 event.getNode()，没有 getSol()；解要用 model.getBestSol()
两个坑（本章实测）：
  - 预处理阶段触发的事件（BESTSOLFOUND 等）里读 LP 会打 ERROR 并返回无意义的值 → 先判 getStage()
  - model.getLPObjVal() 返回的是**缩放后**的目标值（SCIP 按目标系数的小数位放大）
运行：python code/ch11_callback/rootrelax_scip.py
"""

import math
import pathlib
import sys

# 让脚本能 import 到 code/common 下的公共数据
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyscipopt import SCIP_EVENTTYPE, SCIP_STAGE, Eventhdlr, Model, quicksum

from common.data import CVRP
from common.verify import check, show

PROBLEM = "cvrp"

EVENT_NAME = {
    SCIP_EVENTTYPE.FIRSTLPSOLVED: "FIRSTLPSOLVED",
    SCIP_EVENTTYPE.LPSOLVED: "LPSOLVED",
    SCIP_EVENTTYPE.NODEFOCUSED: "NODEFOCUSED",
    SCIP_EVENTTYPE.NODEBRANCHED: "NODEBRANCHED",
    SCIP_EVENTTYPE.NODEINFEASIBLE: "NODEINFEASIBLE",
    SCIP_EVENTTYPE.BESTSOLFOUND: "BESTSOLFOUND",
}


def build_distance(coord):
    return {
        (i, j): round(math.dist(coord[i], coord[j]), 2)
        for i in range(len(coord))
        for j in range(len(coord))
        if i != j
    }


def build_model(binary, hide_output=True):
    """同一个 CVRP 模型，binary=False 时只求 LP 松弛。"""
    data = CVRP
    demand, capacity = data["demand"], data["capacity"]
    nodes = range(len(demand))
    customers = range(1, len(demand))
    dist = build_distance(data["coord"])

    model = Model("CVRP")
    if hide_output:
        model.hideOutput()

    vtype = "B" if binary else "C"
    x = {(i, j): model.addVar(vtype=vtype, lb=0, ub=1, name=f"x{i}_{j}") for (i, j) in dist}
    u = {i: model.addVar(vtype="C", lb=0, name=f"u{i}") for i in customers}

    model.setObjective(
        quicksum(dist[i, j] * x[i, j] for (i, j) in dist), sense="minimize"
    )

    for i in customers:
        model.addCons(quicksum(x[i, j] for j in nodes if j != i) == 1, name=f"out_{i}")
    for j in customers:
        model.addCons(quicksum(x[i, j] for i in nodes if i != j) == 1, name=f"in_{j}")
    for i in customers:
        model.addCons(u[i] >= demand[i], name=f"load_lb_{i}")
        model.addCons(u[i] <= capacity, name=f"load_ub_{i}")
    for i in customers:
        for j in customers:
            if i != j:
                model.addCons(
                    u[j] >= u[i] + demand[j] - capacity * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}",
                )
    model.addCons(
        quicksum(x[0, j] for j in customers) == quicksum(x[i, 0] for i in customers),
        name="depot_balance",
    )
    return model, x


def solve_lp_relaxation():
    model, x = build_model(binary=False)
    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"LP 松弛没解出来，status={model.getStatus()}")
    fractional = {
        (i, j): model.getVal(x[i, j]) for (i, j) in x if 1e-9 < model.getVal(x[i, j]) < 1 - 1e-9
    }
    return model.getObjVal(), fractional


class NodeWatcher(Eventhdlr):
    """订阅根节点 LP 与新解事件，把 SCIP 的内部读数记下来。"""

    def __init__(self, x, dist):
        super().__init__()
        self.x = x                            # 变量句柄：事件里靠它读 LP 解
        self.arcs = list(x)
        self.dist = dist
        self.counts = {}
        self.trace = []            # (事件名, 深度, LP 目标(缩放后), 对偶界, 原始界)
        self.root_lp = None        # (缩放后目标, 自算目标, 分数弧条数, 对偶界)
        self.presolv_hits = 0

    def eventinit(self):
        for event_type in EVENT_NAME:
            self.model.catchEvent(event_type, self)

    def eventexec(self, event):
        # 预处理阶段读 LP 会报错，严重时直接崩 → 先看阶段
        if self.model.getStage() != SCIP_STAGE.SOLVING:
            self.presolv_hits += 1
            return

        name = EVENT_NAME.get(event.getType(), str(event.getType()))
        self.counts[name] = self.counts.get(name, 0) + 1
        try:
            depth = event.getNode().getDepth()
        except Exception:
            depth = None

        lp_obj = self.model.getLPObjVal()
        dual = self.model.getDualbound()
        primal = self.model.getPrimalbound()
        if len(self.trace) < 8:
            self.trace.append((name, depth, lp_obj, dual, primal))

        if name == "FIRSTLPSOLVED":
            # getSolVal(None, var) 取"当前 LP 解"；目标值自己按模型算，绕开 SCIP 的目标缩放
            values = {a: self.model.getSolVal(None, self.x[a]) for a in self.arcs}
            own_obj = sum(self.dist[a] * values[a] for a in self.arcs)
            nfrac = sum(1 for a in self.arcs if 1e-6 < values[a] < 1 - 1e-6)
            if self.root_lp is None or own_obj < self.root_lp[1] - 1e-9:
                self.root_lp = (lp_obj, own_obj, nfrac, dual)


def solve_mip_with_events():
    model, x = build_model(binary=True)
    watcher = NodeWatcher(x, build_distance(CVRP["coord"]))
    model.includeEventhdlr(watcher, "NodeWatcher", "记录根 LP 与新解事件")
    model.optimize()
    if model.getStatus() != "optimal":
        raise RuntimeError(f"没有求到最优解，status={model.getStatus()}")

    solution = {f"x{i}_{j}": model.getVal(x[i, j]) for (i, j) in x}
    return model.getObjVal(), solution, watcher, model.getNNodes()


def main():
    lp_obj, fractional = solve_lp_relaxation()
    print(f"[SCIP] 根 LP 松弛界        = {lp_obj:.6g}")
    print(f"[SCIP] LP 解里的分数弧     = {len(fractional)} 条")

    mip_obj, solution, watcher, nodes = solve_mip_with_events()
    print(f"[SCIP] 事件触发次数        = {watcher.counts}")
    print(f"[SCIP] 预处理阶段被跳过的事件 = {watcher.presolv_hits} 次")

    if watcher.root_lp is None:            # 该模型上 FIRSTLPSOLVED 一次都没触发
        print("[SCIP] 这个模型没有触发 FIRSTLPSOLVED，拿不到 LP 松弛解")
        return
    scaled, own_obj, nfrac, _dual = watcher.root_lp
    factor = round(scaled / own_obj)
    print(f"[SCIP] 回调里的 LP 目标值   = {scaled:.6g}（缩放后）")
    print(f"[SCIP] 按 LP 解自己算       = {own_obj:.6g}（缩放因子 {factor}）")
    print(f"[SCIP] 那个 LP 解的分数弧   = {nfrac} 条")
    print("[SCIP] 前几个事件           = ")
    for name, depth, lp_obj_, dual_, primal_ in watcher.trace[:4]:
        print(f"        {name:<13} depth={depth} lp={lp_obj_:.6g} dual={dual_:.6g} primal={primal_:.6g}")
    print(f"[SCIP] 整数最优            = {mip_obj:.6g}（搜索节点 {nodes} 个）")
    print(f"[SCIP] 根节点 Gap           = {(mip_obj - lp_obj) / abs(mip_obj) * 100:.2f}%")

    show("SCIP", mip_obj, solution)
    check(PROBLEM, mip_obj)


if __name__ == "__main__":
    main()
