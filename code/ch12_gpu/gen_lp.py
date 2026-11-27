"""第十二章 生成中等规模的随机稀疏 LP，供 PDLP / 单纯形 / 内点法对比用。

模型形式（全部约束是 >=，变量有上下界）：

    min  c'x
    s.t. A x >= b
         0 <= x <= u

三个性质保证它是个"干净"的测试例，不依赖运气：

1. **必然可行**：先随机取一个内点 x_feas（每个分量落在 (0.3u, 0.7u)），再令
   b = 0.9 * A x_feas。因为 A 的元素全为正，所以 A x_feas >= b 严格成立。
2. **必然有界**：c > 0 且 x >= 0，目标函数下界为 0，最优解一定存在。
3. **稀疏度可控**：每列恰好 k 个非零，nnz = 列数 * k；并且第 j 列的第一个非零
   强制落在第 (j % 行数) 行，只要列数 >= 行数，每一行都至少有一个非零，
   不会出现"空行 + 正右端项"导致的不可行。

同样的 (rows, cols, k, seed) 一定生成同一个模型——bench_pdlp_highs.py 的
可复现性靠这一条。

用法：
    python code/ch12_gpu/gen_lp.py                     # 打印所有规模的统计
    python code/ch12_gpu/gen_lp.py --scale medium --mps /tmp/medium.mps
"""

import argparse
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

# 四个规模。nnz = cols * nnz_per_col，选值时让 nnz 大约翻 5 倍一档
SCALES = {
    "small": {"rows": 200, "cols": 400, "nnz_per_col": 4},        # nnz = 1 600
    "medium": {"rows": 1000, "cols": 3000, "nnz_per_col": 6},     # nnz = 18 000
    "large": {"rows": 3000, "cols": 12000, "nnz_per_col": 8},     # nnz = 96 000
    "xlarge": {"rows": 8000, "cols": 30000, "nnz_per_col": 8},    # nnz = 240 000
}

DEFAULT_SEED = 7


class Instance:
    """一个生成好的 LP。字段直接用 numpy / scipy，不额外包一层。"""

    def __init__(self, name, A, b, c, upper, seed):
        self.name = name
        self.A = A                # scipy.sparse.csr_matrix (rows x cols)
        self.b = b                # 行下界，行上界是 +inf
        self.c = c                # 目标系数
        self.upper = upper        # 列上界，列下界是 0
        self.seed = seed

    @property
    def rows(self):
        return self.A.shape[0]

    @property
    def cols(self):
        return self.A.shape[1]

    @property
    def nnz(self):
        return self.A.nnz

    def describe(self):
        return f"{self.name}: {self.rows} 行 {self.cols} 列, nnz={self.nnz}"


def build(rows, cols, nnz_per_col=4, seed=DEFAULT_SEED, name=None):
    """按固定种子生成一个稀疏 LP。同样的参数一定得到同样的矩阵。"""
    if cols < rows:
        raise ValueError("本生成器要求 列数 >= 行数，否则可能留下空行")
    if nnz_per_col < 1 or nnz_per_col > rows:
        raise ValueError("nnz_per_col 必须落在 [1, rows]")

    rng = np.random.default_rng(seed)

    # 每列 k 个非零；第一个非零用 j % rows 轮转，保证每行都被覆盖
    col_idx = np.repeat(np.arange(cols), nnz_per_col)
    row_idx = np.empty(cols * nnz_per_col, dtype=np.int64)
    row_idx[0::nnz_per_col] = np.arange(cols) % rows
    for j in range(cols):
        head = j * nnz_per_col
        # 除第一个之外，再从其余行里随机挑 k-1 个互不相同的
        candidates = rng.choice(rows - 1, size=nnz_per_col - 1, replace=False)
        candidates = candidates + (candidates >= row_idx[head])   # 跳过已用的那一行
        row_idx[head + 1: head + nnz_per_col] = candidates

    value = rng.uniform(0.5, 2.0, size=cols * nnz_per_col)
    A = sp.coo_matrix((value, (row_idx, col_idx)), shape=(rows, cols)).tocsr()
    A.sort_indices()

    upper = rng.uniform(1.0, 10.0, size=cols)
    x_feas = rng.uniform(0.3, 0.7, size=cols) * upper
    b = 0.9 * (A @ x_feas)          # A 元素全正、x_feas 全正 ⇒ b > 0 且严格可行
    cost = rng.uniform(0.1, 1.0, size=cols)

    return Instance(name or f"{rows}x{cols}", A, b, cost, upper, seed)


def build_scale(scale, seed=DEFAULT_SEED):
    """按 SCALES 里的名字生成。"""
    if scale not in SCALES:
        raise KeyError(f"未知规模 {scale!r}，可选：{', '.join(SCALES)}")
    return build(name=scale, seed=seed, **SCALES[scale])


def to_highs_lp(inst):
    """转成 highspy.HighsLp（行存储）。

    注意 format_ 和 start_/index_ 必须匹配：csr 矩阵配 kRowwise。
    配错了 passModel 会返回 kError 而不是抛异常，模型静默变空——这是第 12.6 节
    记的一个坑，所以调用方一定要检查 passModel 的返回值。
    """
    import highspy

    if inst.A.format != "csr":
        raise ValueError("to_highs_lp 需要 csr 矩阵")

    lp = highspy.HighsLp()
    lp.num_col_ = inst.cols
    lp.num_row_ = inst.rows
    lp.col_cost_ = inst.c
    lp.col_lower_ = np.zeros(inst.cols)
    lp.col_upper_ = inst.upper
    lp.row_lower_ = inst.b
    lp.row_upper_ = np.full(inst.rows, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = inst.A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = inst.A.indices.astype(np.int32)
    lp.a_matrix_.value_ = inst.A.data
    lp.sense_ = highspy.ObjSense.kMinimize
    return lp


def export_mps(inst, path):
    """写 MPS 文件（cuPDLP-C 的命令行程序只吃 MPS）。返回路径。"""
    import highspy

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.passModel(to_highs_lp(inst))
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"passModel 失败：{status}")
    path.unlink(missing_ok=True)
    h.writeModel(str(path))
    return path


def main():
    parser = argparse.ArgumentParser(description="生成稀疏 LP 测试例")
    parser.add_argument("--scale", choices=list(SCALES), help="只生成/导出某一个规模")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mps", help="同时导出 MPS 到这个路径（需要 --scale）")
    args = parser.parse_args()

    scales = [args.scale] if args.scale else list(SCALES)
    for name in scales:
        inst = build_scale(name, seed=args.seed)
        print(inst.describe() + f", 每列 {SCALES[name]['nnz_per_col']} 个非零, seed={args.seed}")
        if args.mps:
            if not args.scale:
                parser.error("--mps 需要配合 --scale 使用")
            print(f"    MPS 已写入 {export_mps(inst, args.mps)}")


if __name__ == "__main__":
    sys.exit(main())
