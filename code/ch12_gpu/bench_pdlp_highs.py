"""第十二章 CPU 实测：在同一个稀疏 LP 上对比单纯形 / 内点法 / PDLP（HiGHS 1.16）。

这一章的核心事实来自这个脚本的真实输出。它回答一个具体问题：
**一阶方法（PDLP）在 CPU 上到底什么时候比单纯形 / 内点法快，什么时候更慢？**

跑法（仓库根目录）：

    python code/ch12_gpu/bench_pdlp_highs.py             # 四个规模全跑，约 5~6 分钟
    python code/ch12_gpu/bench_pdlp_highs.py --quick     # 只跑 small + medium，几秒钟
    python code/ch12_gpu/bench_pdlp_highs.py --scales large --time-limit 300

结果同时写到 code/ch12_gpu/output/bench.tsv（Tab 分隔，Excel 能直接打开），方便自己画图。

两个口径说明：

* **耗时**取同一配置多次运行的最小值（wall clock，含预处理与求解）。机器上有别的
  负载时这个数会抖，取最小值比取平均更稳定，也更容易复现。
* **参考解**是 simplex 求到的最优值；simplex 触时限时退而用 ipm 的最优值。
  PDLP 的偏差 = |obj - 参考解| / |参考解|，这正是"精度可调到够用就行"要看的数。
"""

import argparse
import csv
import os
import pathlib
import platform
import sys
import time

import highspy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gen_lp  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
SOLVERS = ("simplex", "ipm", "pdlp")
# 低于这个 nnz 的实例跑三遍取最小，更大的只跑一遍（单次就要几十秒）
REPEAT_THRESHOLD = 50_000


class Result:
    __slots__ = ("solver", "status", "objective", "seconds", "iterations", "basis_validity")

    def __init__(self, solver, status, objective, seconds, iterations, basis_validity):
        self.solver = solver
        self.status = status
        self.objective = objective
        self.seconds = seconds
        self.iterations = iterations
        self.basis_validity = basis_validity


def solve(inst, solver, time_limit=120.0, **options):
    """用指定算法解一次，返回结果。options 里放 HiGHS 选项名→值的键值对。"""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.passModel(gen_lp.to_highs_lp(inst))
    if status != highspy.HighsStatus.kOk:
        # passModel 失败不会抛异常，只会让模型变空，必须自己检查
        raise RuntimeError(f"passModel 失败：{status}")

    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", float(time_limit))
    for key, value in options.items():
        h.setOptionValue(key, value)

    start = time.perf_counter()
    h.run()
    seconds = time.perf_counter() - start

    info = h.getInfo()
    iterations = {
        "simplex": info.simplex_iteration_count,
        "ipm": info.ipm_iteration_count,
        "pdlp": info.pdlp_iteration_count,
    }[solver]
    return Result(
        solver=solver,
        status=h.getModelStatus(),
        objective=h.getObjectiveValue(),
        seconds=seconds,
        iterations=iterations,
        basis_validity=info.basis_validity,
    )


def bench_one(inst, solver, repeats, time_limit, **options):
    """跑 repeats 次，返回 (最快的那次结果, 全部耗时)。"""
    runs = [solve(inst, solver, time_limit=time_limit, **options) for _ in range(repeats)]
    times = [r.seconds for r in runs]
    best = min(runs, key=lambda r: r.seconds)
    return best, times


def status_text(status):
    name = status.name
    return {
        "kOptimal": "最优",
        "kTimeLimit": "触时限",
        "kIterationLimit": "触迭代上限",
        "kUnknown": "未证最优",
        "kInfeasible": "不可行",
        "kUnbounded": "无界",
    }.get(name, name)


def fmt_time(seconds):
    if seconds >= 10:
        return f"{seconds:8.1f}"
    return f"{seconds:8.3f}"


def reference_objective(results):
    """参考解：simplex 最优 → ipm 最优 → pdlp 最优，谁先拿到用谁。"""
    for solver in SOLVERS:
        r = results.get(solver)
        if r is not None and r.status == highspy.HighsModelStatus.kOptimal:
            return r.objective, solver
    return None, None


def run_scale(name, repeats_cap, time_limit):
    inst = gen_lp.build_scale(name)
    repeats = 3 if inst.nnz <= REPEAT_THRESHOLD else 1
    repeats = min(repeats, repeats_cap)
    print(f"\n=== {inst.describe()}  (每列 {inst.nnz // inst.cols} 个非零, "
          f"重复 {repeats} 次) ===", flush=True)

    results = {}
    for solver in SOLVERS:
        best, times = bench_one(inst, solver, repeats, time_limit)
        results[solver] = best
        spread = f"  各次 {', '.join(f'{t:.3f}' for t in times)}" if len(times) > 1 else ""
        print(f"  {solver:8s} {status_text(best.status):6s} 迭代 {best.iterations:>7d} 次 "
              f"耗时 {fmt_time(best.seconds)} s{spread}", flush=True)

    ref, ref_solver = reference_objective(results)
    rows = []
    for solver in SOLVERS:
        r = results[solver]
        if ref is None:
            rel = float("nan")
        else:
            rel = abs(r.objective - ref) / max(abs(ref), 1e-30)
        rows.append({
            "scale": name,
            "rows": inst.rows,
            "cols": inst.cols,
            "nnz": inst.nnz,
            "solver": solver,
            "status": status_text(r.status),
            "iterations": r.iterations,
            "seconds": round(r.seconds, 4),
            "objective": r.objective,
            "rel_diff": rel,
            "basis_validity": r.basis_validity,
        })
    if ref is not None:
        print(f"  参考解（{ref_solver}）= {ref:.6f}", flush=True)
        for row in rows:
            print(f"    {row['solver']:8s} 相对偏差 {row['rel_diff']:.2e}  "
                  f"基有效性 {row['basis_validity']}", flush=True)
    return rows


def tolerance_sweep(scale="medium", time_limit=120.0, repeats=3):
    """PDLP 的精度旋钮：放宽容差能省多少迭代？解还能用吗？

    两个旋钮要分开看，这是 12.6 节那个坑的来源：

    * `pdlp_optimality_tolerance`：PDLP 自己的终止容差。
    * `kkt_tolerance`：HiGHS 判定"这个解算不算最优"的门槛。HiGHS 官方 GPU 文档
      建议一上手就把它放宽到 1e-4；只改前一个、不动这一个，迭代数往往纹丝不动。
    """
    inst = gen_lp.build_scale(scale)
    print(f"\n=== 精度扫描：{inst.describe()} ===", flush=True)
    ref = solve(inst, "ipm", time_limit=time_limit)
    print(f"  参考解（ipm）= {ref.objective:.9f}，耗时 {ref.seconds:.3f} s", flush=True)

    rows = []

    def record(label, r):
        rel = abs(r.objective - ref.objective) / abs(ref.objective)
        print(f"  {label:34s} 迭代 {r.iterations:>6d}  耗时 {r.seconds:7.3f} s  "
              f"相对偏差 {rel:.2e}  {status_text(r.status)}", flush=True)
        rows.append({
            "scale": f"{scale}/{label}",
            "rows": inst.rows,
            "cols": inst.cols,
            "nnz": inst.nnz,
            "solver": "pdlp",
            "status": status_text(r.status),
            "iterations": r.iterations,
            "seconds": round(r.seconds, 4),
            "objective": r.objective,
            "rel_diff": rel,
            "basis_validity": r.basis_validity,
        })

    print("  （kkt_tolerance 保持默认 1e-7，只调 PDLP 自己的容差）", flush=True)
    for tol in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        best, _ = bench_one(inst, "pdlp", repeats, time_limit,
                            pdlp_optimality_tolerance=tol)
        record(f"pdlp_tol={tol:g}", best)

    print("  （PDLP 容差用默认值，改 HiGHS 的 kkt_tolerance）", flush=True)
    for kkt in (1e-4, 1e-5, 1e-6):
        best, _ = bench_one(inst, "pdlp", repeats, time_limit, kkt_tolerance=kkt)
        record(f"kkt_tolerance={kkt:g}", best)
    return rows


def write_table(rows, path):
    """写 Tab 分隔的表。不用 .csv：某些企业终端的安全软件会加密 .csv 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scale", "rows", "cols", "nnz", "solver", "status", "iterations",
              "seconds", "objective", "rel_diff", "basis_validity"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="HiGHS 上单纯形 / 内点 / PDLP 的 CPU 对比")
    parser.add_argument("--scales", nargs="+", choices=list(gen_lp.SCALES),
                        default=list(gen_lp.SCALES), help="要跑的规模（默认全部四个）")
    parser.add_argument("--time-limit", type=float, default=120.0, help="每个求解器的时间上限（秒）")
    parser.add_argument("--repeats", type=int, default=3, help="小实例的重复次数上限")
    parser.add_argument("--quick", action="store_true", help="只跑 small + medium")
    parser.add_argument("--sweep", action="store_true", help="跑完主表后额外跑一次精度扫描")
    parser.add_argument("--sweep-only", action="store_true", help="只跑精度扫描（不重跑主表）")
    args = parser.parse_args()

    if args.sweep_only:
        rows = tolerance_sweep(time_limit=args.time_limit)
        write_table(rows, OUT / "bench_sweep.tsv")
        print(f"\n结果已写入 {OUT / 'bench_sweep.tsv'}")
        return

    scales = ["small", "medium"] if args.quick else args.scales
    print(f"HiGHS {highspy.Highs().version()} | {platform.platform()} | "
          f"CPU 核数 {os.cpu_count()}")
    print("耗时＝wall clock（含预处理），小实例取多次最小值；触时限的行只保证"
          f"不优于 {args.time_limit:g} s。")

    rows = []
    for name in scales:
        rows.extend(run_scale(name, args.repeats, args.time_limit))
    if args.sweep:
        sweep_rows = tolerance_sweep(time_limit=args.time_limit)
        rows.extend(sweep_rows)

    write_table(rows, OUT / "bench.tsv")
    print(f"\n结果已写入 {OUT / 'bench.tsv'}")

    optimal = [r for r in rows if r["status"] == "最优"]
    print(f"（共 {len(rows)} 组结果，其中 {len(optimal)} 组证到最优）")


if __name__ == "__main__":
    main()
