"""例题校验工具：统一打印结果、统一比对最优值。

每个求解器脚本都暴露一个 solve() 函数，返回 (目标值, 解字典)。
scripts/verify_all.py 会把所有能跑的求解器都跑一遍，比对目标值是否一致。
"""

from common.data import EXPECTED

TOL = 1e-6


def show(name, objective, solution):
    """打印单个求解器的结果。"""
    print(f"[{name}] 最优目标值 = {objective:.6g}")
    for key, value in solution.items():
        if abs(value) > 1e-9:
            print(f"    {key} = {value:g}")


def check(problem, objective):
    """比对最优目标值，和 data.EXPECTED 里的标准答案对照。"""
    expected = EXPECTED.get(problem)
    if expected is None:
        return True
    if abs(objective - expected) > TOL:
        raise AssertionError(
            f"{problem}: 最优目标值 {objective:.6g} 与标准答案 {expected:.6g} 不一致"
        )
    return True
