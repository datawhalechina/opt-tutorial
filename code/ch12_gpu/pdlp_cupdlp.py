"""第十二章 cuPDLP-C：NVIDIA 开源的一阶 LP 求解器（本机未实跑）。

⚠️ **本机未实跑（需 NVIDIA GPU + CUDA）**：cuPDLP-C 是 CUDA C/C++ 程序，
没有 Python 包，只能编译命令行程序 `plc`。这台 Mac 既没有 NVIDIA GPU 也没有
CUDA Toolkit，所以本脚本的职责是：

1. 用 `gen_lp.py` 生成一个 MPS 文件（cuPDLP-C 的命令行只吃 MPS）；
2. 把官方 README 里的编译/调用命令**原样打印出来**，方便你在 GPU 机器上照抄；
3. 如果 PATH 或环境变量里真的找到了 `plc`，`--run` 会把它跑起来。

以下命令与参数名来自官方仓库（https://github.com/COPT-Public/cuPDLP-C）：

上游 CMake 路径（v0.4.x 的 README，产出 `plc`）：

    export CUDA_HOME=/path-to-cuda        # GPU 模式才需要；CUDA 无官方最低版本号
    cmake -DBUILD_CUDA=ON ..
    cmake --build . --target plc
    ./bin/plc -fname <mpsfile> -nIterLim 5000

COIN-OR onboarding 版路径（2026-10 起，产出在 .coinor-deps/ 下）：

    MODE=gpu bash build.sh

前提按官方 README/INSTALL 原文：C/C++ 编译器、CMake >= 3.24、zlib、Python 3 + NumPy。
HiGHS 有两条路，别混：
  - COIN-OR 打包版（上面的 build.sh）："Bundled HiGHS 1.6.0 is built automatically"，
    内置的 1.6.0 自动构建，不用你自己装；
  - 上游 v0.4.x（CMake 直接构建）：README 要求先装好 HiGHS 并
    `export HIGHS_HOME=/path-to-highs`，否则 configure 失败（FindHiGHSConf.cmake 读它）。
CUDA Toolkit 只在 GPU 模式下需要。VALIDATION.md 记录的实测配置是
CUDA Toolkit 12.8.93 + RTX 4090（driver 580.82.07）。

要点：`-DBUILD_CUDA=ON` 默认是 **OFF**，也就是不加这个开关编出来的是 CPU 版。
官方论文（arXiv:2312.14832）里 GPU 实验用的是 NVIDIA H100 80GB。

顺带一提：HiGHS 从 1.10 起内置的 PDLP 求解器就是 cuPDLP-C（`solver = "pdlp"`），
pip 装下来的 wheel 是 CPU 版——本机跑 `bench_pdlp_highs.py` 时日志里那句
"Solving with cuPDLP-C" 就是它。

用法：
    python code/ch12_gpu/pdlp_cupdlp.py                  # 生成 MPS + 打印命令
    python code/ch12_gpu/pdlp_cupdlp.py --scale large
    python code/ch12_gpu/pdlp_cupdlp.py --run --plc /path/to/plc   # 本机未实跑
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gen_lp  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"

BUILD_STEPS = """\
# 1) 依赖（官方 README/INSTALL 原文）：C/C++ 编译器、CMake>=3.24、zlib、Python3+NumPy
#    内置 HiGHS 1.6.0 自动构建；CUDA Toolkit 只有 GPU 模式才要（无官方最低版本号）
export CUDA_HOME=/usr/local/cuda-12.8     # 换成你自己的路径

# 2) 编译 GPU 版（默认 OFF，不加 -DBUILD_CUDA=ON 编出来的是 CPU 版）
mkdir -p build && cd build
cmake -DBUILD_CUDA=ON \\
      -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \\
      -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" \\
      -DCMAKE_CUDA_FLAGS_RELEASE="-O2 -DNDEBUG" ..
cmake --build . --target plc

# 3) 求解（参数名与默认值来自官方 README 的参数表）
./bin/plc -fname {mps} -nIterLim 5000 -dTimeLim 600 -savesol true -outSol {sol}

# 附：2026-10 起仓库的 COIN-OR onboarding 版改用 build.sh 入口
#     export GPU_ARCH=89        # 默认 native；RTX4090=89, H100=90
#     MODE=gpu bash build.sh    # 产物在 .coinor-deps/cupdlp-gpu-install/bin/plc
"""

PARAM_TABLE = """\
| 参数 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `fname` | 输入的 .mps 文件 | 必填 |
| `out` | 汇总结果 json 路径 | ./solution-sum.json |
| `outSol` | 解 json 路径 | ./solution.json |
| `savesol` | 是否把解写进 json | false |
| `nIterLim` | 迭代上限 | INT_MAX |
| `dPrimalTol` / `dDualTol` / `dGapTol` | 原始/对偶可行性、对偶间隙容差 | 1e-4 |
| `dFeasTol` | 原始/对偶不可行性检查容差 | 1e-8 |
| `dTimeLim` | 时间上限（秒） | 3600 |
| `ifScaling` / `ifRuizScaling` / `ifPcScaling` | 是否用缩放 / Ruiz / Pock-Chambolle 缩放 | true / true / true |
| `eLineSearchMethod` | 0 固定步长、2 自适应（官方表里 1=Malitsky 已划掉，Range 只写 0,2） | 2 |
| `eRestartMethod` | 0 不重启、1 KKT 版本重启（官方原文 "1-KKTversion"；与 GPU 无关） | 1 |
"""


def ensure_mps(scale):
    path = OUT / f"{scale}.mps"
    if not path.exists():
        inst = gen_lp.build_scale(scale)
        gen_lp.export_mps(inst, path)
        print(f"{path.name}: {inst.describe()}")
    return path


def find_plc(explicit=None):
    """找 plc 可执行文件：命令行指定 > 环境变量 CUPDLP_PLC > PATH。"""
    if explicit:
        return explicit
    env = os.environ.get("CUPDLP_PLC")
    if env:
        return env
    return shutil.which("plc")


def main():
    parser = argparse.ArgumentParser(description="cuPDLP-C 的编译与调用（GPU，本机未实跑）")
    parser.add_argument("--scale", default="medium", choices=list(gen_lp.SCALES))
    parser.add_argument("--run", action="store_true", help="找到 plc 才真的执行（本机未实跑）")
    parser.add_argument("--plc", help="plc 可执行文件路径")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    mps = ensure_mps(args.scale)
    solution = OUT / f"{args.scale}_cupdlp.json"

    print("\n一、编译（官方 README，原样照抄）")
    print(BUILD_STEPS.format(mps=mps.name, sol=solution.name))
    print("二、命令行参数（官方 README）")
    print(PARAM_TABLE)

    plc = find_plc(args.plc)
    if not args.run:
        print("三、本机状态：未实跑（需要 NVIDIA GPU + CUDA）。"
              "有了 plc 之后加 --run 即可。")
        return 0
    if not plc:
        print("三、没找到 plc；先按上面的步骤编译，或用 --plc 指定路径。")
        return 1

    cmd = [plc, "-fname", str(mps), "-nIterLim", "5000",
           "-savesol", "true", "-outSol", str(solution)]
    print(f"三、执行：{' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
