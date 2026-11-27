"""第十二章 Gurobi 的 PDHG + GPU 路线（本机未实跑）。

⚠️ **本机未实跑（需 NVIDIA GPU + CUDA 版 Gurobi）**：这台机器是 macOS ARM，
没有 NVIDIA GPU；而且本机 Gurobi 许可证已于 2025-07-02 过期，连 `gp.Model()`
都建不出来。所以下面的 `--build` 分支在本机只能跑到抛异常为止。

那这个脚本还留在这里干什么？两件事：

1. `--check-params`：直接读 gurobipy 自带的参数表（`gurobipy/_paramdetails.py`），
   把本章用到的参数名、默认值、取值范围、官方原文打印出来。**不需要许可证**，
   本机实跑过——这是"参数名不是编的"最硬的证据。
2. `--build`：给出完整可抄的建模式样，拿到许可证 + GPU 机器后直接跑。

参数语义（与 `--check-params` 的输出逐字一致）：

* `Method = 6`（常量 `GRB.METHOD_PDHG`）才是 PDHG；`0/1` 是原始/对偶单纯形，
  `2` 是内点（barrier），`3/4` 是并发，`5` 是已废弃的确定性并发单纯形，
  `7` 是非线性 barrier。**只设 `PDHGGPU=1` 不设 `Method=6`，PDHG 根本不会启动。**
* `PDHGGPU`：1 让 PDHG 跑 GPU，0 强制 CPU，默认 -1（多数情况等价 0，
  例外是"Compute Server 的 worker 带 GPU"时默认等价 1）。
* GPU 支持要**单独下载 GPU 版安装包**，而且 Compute Server / Instant Cloud 上不可用。

用法：
    python code/ch12_gpu/pdlp_gurobi.py --check-params      # 本机可跑，不需要许可证
    python code/ch12_gpu/pdlp_gurobi.py --build             # 需要许可证 + GPU 机器
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gen_lp  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
PARAMS = (
    "Method",
    "PDHGGPU",
    "PDHGAbsTol",
    "PDHGRelTol",
    "PDHGConvTol",
    "PDHGIterLimit",
    "PDHGEarlyCrossover",
    "PDHGMIPHeurTime",
    "PDHGMIPHeurWork",
    "PDHGMIPHeurSolutions",
)


def check_params():
    """打印 gurobipy 内置参数表里的 PDHG 条目。不需要许可证。"""
    import gurobipy as gp

    try:
        from gurobipy._paramdetails import param_details
    except ModuleNotFoundError:
        # PDHG 与 _paramdetails 都是 12+ 才有的；11.x 里连 GRB.Param.PDHGGPU 都不存在
        print(f"这个环境的 gurobipy 是 {gp.gurobi.version()}，没有 _paramdetails，"
              "也就没有 PDHG 相关参数。")
        print("本章的 Gurobi 参数表需要在 gurobipy >= 12（实测 14.0.0）的环境里读。")
        return 1

    for name in PARAMS:
        detail = param_details[name]
        values = detail["values"]
        body = detail["description"].replace("\n", " ")
        print(f"{name}  type={values.get('type')} default={values.get('default')} "
              f"range=[{values.get('minimum')}, {values.get('maximum')}]")
        if name == "Method":
            # Method 的取值列表太长，这里只挑关键几行
            for line in body.split("* "):
                if any(key in line for key in ("=PDHG", "=barrier", "=dual simplex",
                                               "=primal simplex")):
                    print(f"    {line.strip().rstrip(',')}")
        else:
            print(f"    {body[:300]}...")


def build_and_solve(scale, gpu):
    """真正调 Gurobi 的部分。本机因许可证过期跑不到这里。"""
    import gurobipy as gp

    mps = OUT / f"{scale}.mps"
    if not mps.exists():
        gen_lp.export_mps(gen_lp.build_scale(scale), mps)
        print(f"{mps.name} 已生成")

    model = gp.read(str(mps))
    model.Params.Method = gp.GRB.METHOD_PDHG         # 6，注意：必须先有这一行
    model.Params.PDHGGPU = 1 if gpu else 0           # 1=GPU, 0=强制 CPU
    model.Params.PDHGEarlyCrossover = 2.0            # 提前开 crossover，尽早拿到基
    model.Params.TimeLimit = 300.0
    model.Params.LogToConsole = 1
    model.optimize()

    if model.SolCount == 0:
        print(f"没拿到解，状态 {model.Status}")
        return
    print(f"目标值 {model.ObjVal:.6f}  耗时 {model.Runtime:.3f} s  "
          f"迭代 {model.IterCount}  状态 {model.Status}")
    return model.ObjVal


def main():
    parser = argparse.ArgumentParser(description="Gurobi 的 PDHG / PDHGGPU 用法")
    parser.add_argument("--check-params", action="store_true", help="只打印参数表（无需许可证）")
    parser.add_argument("--build", action="store_true", help="真正建模型求解（需许可证 + GPU）")
    parser.add_argument("--gpu", action="store_true", help="PDHGGPU=1，本机未实跑")
    parser.add_argument("--scale", default="medium", choices=list(gen_lp.SCALES))
    args = parser.parse_args()

    if args.check_params or not args.build:
        return check_params()
    if not args.gpu:
        print("提示：不加 --gpu 时 PDHGGPU=0，走的是 CPU 上的 PDHG。")

    OUT.mkdir(exist_ok=True)
    try:
        build_and_solve(args.scale, args.gpu)
    except Exception as exc:          # noqa: BLE001 —— 许可证错误的信息要原样给用户看
        print(f"\n没跑起来：{type(exc).__name__}: {exc}")
        print("本机情况：macOS ARM 无 NVIDIA GPU，Gurobi 许可证已过期，"
              "这一分支属于「本机未实跑」。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
