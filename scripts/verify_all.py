"""一键校验：把所有能跑的求解器都跑一遍，比对最优目标值是否一致。

用法（在仓库根目录）：
    python scripts/verify_all.py            # 跑全部
    python scripts/verify_all.py lp         # 只跑名字里含 lp 的例题
    python scripts/verify_all.py -v         # 同时打印每个求解器的解

退出码：0 全部通过；1 有例题失败或有求解器缺失导致该例题无任一可运行实现。
"""

import importlib.util
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common.data import EXPECTED  # noqa: E402

TOL = 1e-6

# 这些错误说明"本机没有可用的求解器授权/运行时"，不算代码错误，标记为跳过
UNAVAILABLE_PATTERNS = (
    "License expired",
    "license file",
    "no CPLEX runtime found",
    "Unable to open Gurobi license",
)

# 例题 -> {求解器名: 脚本相对路径}
CASES = {
    "lp_profit": {
        "Gurobi": "code/ch01_lp/profit_gurobi.py",
        "COPT": "code/ch01_lp/profit_copt.py",
        "CPLEX": "code/ch01_lp/profit_cplex.py",
        "HiGHS": "code/ch01_lp/profit_highs.py",
        "SCIP": "code/ch01_lp/profit_scip.py",
    },
    "knapsack": {
        "Gurobi": "code/ch02_knapsack/knapsack_gurobi.py",
        "COPT": "code/ch02_knapsack/knapsack_copt.py",
        "CPLEX": "code/ch02_knapsack/knapsack_cplex.py",
        "HiGHS": "code/ch02_knapsack/knapsack_highs.py",
        "SCIP": "code/ch02_knapsack/knapsack_scip.py",
    },
    "facility": {
        "Gurobi": "code/ch03_milp/facility_gurobi.py",
        "COPT": "code/ch03_milp/facility_copt.py",
        "CPLEX": "code/ch03_milp/facility_cplex.py",
        "HiGHS": "code/ch03_milp/facility_highs.py",
        "SCIP": "code/ch03_milp/facility_scip.py",
    },
    "scheduling": {
        "Gurobi": "code/ch04_techniques/scheduling_gurobi.py",
        "COPT": "code/ch04_techniques/scheduling_copt.py",
        "CPLEX": "code/ch04_techniques/scheduling_cplex.py",
        "HiGHS": "code/ch04_techniques/scheduling_highs.py",
        "SCIP": "code/ch04_techniques/scheduling_scip.py",
    },
    "cvrp": {
        "Gurobi": "code/ch07_cvrp/cvrp_gurobi.py",
        "COPT": "code/ch07_cvrp/cvrp_copt.py",
        "HiGHS": "code/ch07_cvrp/cvrp_highs.py",
        "SCIP": "code/ch07_cvrp/cvrp_scip.py",
    },
    "lp_profit(建模层)": {
        "PuLP": "code/ch08_modeling/pulp_lp.py",
        "Pyomo": "code/ch08_modeling/pyomo_lp.py",
    },
    "facility(建模层)": {
        "PuLP": "code/ch08_modeling/pulp_milp.py",
        "Pyomo": "code/ch08_modeling/pyomo_milp.py",
    },
}


def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(name, path, verbose):
    """返回 (状态, 说明)。状态为 ok / skip / fail。"""
    try:
        module = load(path, f"case_{abs(hash(path))}")
    except ImportError as exc:
        return "skip", f"未安装（{exc.name}）"
    except Exception as exc:                        # 语法错误、导入期报错
        return "fail", f"导入失败：{type(exc).__name__}: {exc}"

    try:
        objective, _ = module.solve()
    except SystemExit:
        return "skip", "求解器不可用"
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        detail = f"{type(exc).__name__}: {str(exc)[:90]}"
        if any(p in str(exc) for p in UNAVAILABLE_PATTERNS):
            return "skip", f"未授权 / 无运行时（{detail}）"
        return "fail", f"运行失败：{detail}"

    expected = EXPECTED.get(name.replace("(建模层)", ""))
    if expected is not None and abs(objective - expected) > TOL:
        return "fail", f"{objective:.6g} ≠ 标准答案 {expected:.6g}"
    return "ok", f"{objective:.6g}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    keyword = args[0] if args else ""

    failures = 0
    results = {}          # (例题, 求解器) -> (状态, 说明)
    print(f"{'例题':<18}{'求解器':<10}{'结果':<8}说明")
    print("-" * 78)
    for name, solvers in CASES.items():
        if keyword and keyword not in name:
            continue
        passed = 0
        for solver, path in solvers.items():
            status, detail = run_case(name, path, verbose)
            results[(name, solver)] = (status, detail)
            mark = {"ok": "通过", "skip": "跳过", "fail": "失败"}[status]
            print(f"{name:<18}{solver:<10}{mark:<8}{detail}")
            if status == "ok":
                passed += 1
            elif status == "fail":
                failures += 1
        if passed == 0:
            print(f"  ⚠️ {name} 没有任何可运行的求解器实现")
            failures += 1
        print("-" * 78)

    # 汇总矩阵：行是例题，列是求解器
    solvers_all = sorted({s for _, s in results})
    print("\n汇总（✓ 通过 / - 跳过 / ✗ 失败）")
    print(f"{'例题':<20}" + "".join(f"{s:<9}" for s in solvers_all))
    for name in dict.fromkeys(n for n, _ in results):
        cells = []
        for solver in solvers_all:
            if (name, solver) not in results:
                cells.append(f"{'·':<9}")
            else:
                cells.append(f"{ {'ok': '✓', 'skip': '-', 'fail': '✗'}[results[(name, solver)][0]]:<9}")
        print(f"{name:<20}" + "".join(cells))

    print("\n全部通过 ✅" if failures == 0 else f"\n有 {failures} 项失败 ❌")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
