"""第十二章 COPT 的第一阶方法（PDLP）与 GPU 开关。

参数取名依据（都核实过，不是猜的）：

* `LpMethod = 6` 就是"First-order method (PDLP)"。COPT 官方参数文档
  https://guide.coap.online/copt/en-doc/parameter.html 原文列出
  `-1/1/2/3/4/5/6` 六个取值，6 是 PDLP；本机 8.0.1 实测日志也打印
  "Starting PDLP solver using CPU"。
* `GPUMode`：-1 自动、0 强制 CPU、1 标准 GPU、2 高性能 GPU（barrier 专用）。
  官方文档明确写："On platforms such as macOS, only the CPU version of COPT is
  currently available. Even if GPUMode=1 or 2 is set, COPT will still run in CPU
  mode." —— 也就是说本机设 GPUMode=1 不会报错，只是回落 CPU。
* `GPUDevice` 只是多卡时选卡，**单设它不会打开 GPU**，也不会给出任何提示。
* `PDLPTol` / `PDLPPrimalTol` / `PDLPDualTol` / `PDLPIterLimit` / `Crossover`
  都是真实存在的参数名（`setParam` 对不存在的名字会抛
  `AttributeError: Invalid parameter name`，可以用这个方法探测）。

⚠️ 本机是 macOS ARM、没有 NVIDIA GPU：下面 `--gpu` 那条分支**本机未实跑**
（能跑起来，但只会在日志里打印 "GPU mode is not supported" 然后回到 CPU）。

用法：
    python code/ch12_gpu/pdlp_copt.py                 # CPU 上的 PDLP（本机可跑）
    python code/ch12_gpu/pdlp_copt.py --scale large
    python code/ch12_gpu/pdlp_copt.py --gpu           # 需要 NVIDIA GPU + CUDA 版 COPT
"""

import argparse
import pathlib
import sys
import time

import coptpy as cp
from coptpy import COPT

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gen_lp  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
LOG_INTERESTING = ("PDLP", "GPU", "gpu", "crossover", "Crossover", "Simplex", "Barrier")


def status_text(status):
    """把 COPT 的状态码翻成能读的名字。"""
    for name in ("OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNB", "TIMEOUT",
                 "NUMERICAL", "INTERRUPTED"):
        if getattr(COPT, name) == status:
            return name
    return str(status)


def ensure_mps(scale):
    """把生成好的 LP 落成 MPS，COPT 用 model.read() 读回来。"""
    path = OUT / f"{scale}.mps"
    if not path.exists():
        inst = gen_lp.build_scale(scale)
        gen_lp.export_mps(inst, path)
        print(f"{path.name}: {inst.describe()}")
    return path


def solve(mps_path, lp_method, gpu_mode=None, crossover=None, pdlp_tol=None,
          pdlp_iter=None, log_name="copt_pdlp.log"):
    """解一次，返回 (耗时秒, 目标值, 状态)。日志落盘，事后抽关键行。"""
    env = cp.Envr()
    model = env.createModel("pdlp_demo")
    model.setParam("Logging", 1)
    model.setParam("TimeLimit", 300.0)
    log_path = OUT / log_name
    log_path.unlink(missing_ok=True)          # COPT 的日志文件是追加写
    model.setLogFile(str(log_path))
    model.read(str(mps_path))

    model.setParam("LpMethod", lp_method)     # 6 = PDLP
    if gpu_mode is not None:
        model.setParam("GPUMode", gpu_mode)   # 0 CPU / 1 GPU / -1 自动
    if crossover is not None:
        model.setParam("Crossover", crossover)
    if pdlp_tol is not None:
        model.setParam("PDLPTol", pdlp_tol)
    if pdlp_iter is not None:
        model.setParam("PDLPIterLimit", pdlp_iter)

    start = time.perf_counter()
    model.solve()
    seconds = time.perf_counter() - start
    return seconds, model.objval, status_text(model.status), log_path


def show_device(log_path):
    """从日志里挑出"到底跑了什么"的那几行——这是唯一可信的证据。"""
    if not log_path.exists():
        return
    print("    日志里的关键行：")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if any(key in line for key in LOG_INTERESTING):
            print(f"      | {line.strip()}")


def main():
    parser = argparse.ArgumentParser(description="COPT 上 PDLP（CPU/GPU）的用法")
    parser.add_argument("--scale", default="medium", choices=list(gen_lp.SCALES))
    parser.add_argument("--gpu", action="store_true", help="设 GPUMode=1（本机未实跑）")
    parser.add_argument("--pdlp-tol", type=float, default=None, help="PDLPTol，默认 1e-6")
    parser.add_argument("--no-crossover", action="store_true", help="关掉 PDLP 后的 crossover")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    mps = ensure_mps(args.scale)

    # 参考解：1 = dual simplex
    print("\n[参考] LpMethod=1（dual simplex）")
    sec_ref, obj_ref, status_ref, log_ref = solve(mps, 1, log_name="copt_simplex.log")
    print(f"    目标值 {obj_ref:.6f}  耗时 {sec_ref:.3f} s  状态 {status_ref}")

    gpu_mode = 1 if args.gpu else 0
    label = "GPUMode=1（GPU，本机未实跑）" if args.gpu else "GPUMode=0（强制 CPU）"
    print(f"\n[PDLP] LpMethod=6 + {label}")
    sec, obj, status, log_path = solve(
        mps, 6, gpu_mode=gpu_mode,
        crossover=0 if args.no_crossover else None,
        pdlp_tol=args.pdlp_tol,
    )
    gap = abs(obj - obj_ref) / abs(obj_ref)
    print(f"    目标值 {obj:.6f}  耗时 {sec:.3f} s  状态 {status}  "
          f"与单纯形参考解的相对偏差 {gap:.2e}")
    show_device(log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
