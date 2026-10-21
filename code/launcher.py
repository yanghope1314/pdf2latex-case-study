# -*- coding: utf-8 -*-
"""
launcher.py —— 双击启动器（给不想敲命令的人用）

【怎么用】把要转换的文件放进【输入】文件夹，然后双击本文件（或 start.bat），
          结果会出现在【输出】文件夹里。就这么简单。

----------------------------------------------------------------------
【为什么要单独写一个 launcher，不把命令直接写在 .bat 里？】

  Windows 的 .bat 有个绕不开的坑：
    CMD 读取 .bat 文件时，用的是**系统的 ANSI 码页**（中文系统上是 GBK）。
    所以 .bat 里只要出现中文（比如「输入」「输出」这样的文件夹名），
    就会陷入两难：
      · 把 .bat 存成 GBK → 和 Python 输出的 UTF-8 打架，显示乱码
      · 把 .bat 存成 UTF-8 → CMD 把中文路径读成乱码，
                             报"系统找不到指定的路径"

  **解法：让 .bat 里只写英文，所有中文路径交给 Python 处理。**
  Python 处理中文路径是可靠的，不受控制台码页影响。

  所以 start.bat 里只有一句"用 Python 去跑 launcher.py"，没有别的内容。
----------------------------------------------------------------------
【流程】
    1. 检查【输入】文件夹里有没有东西
    2. 没有 → 提示怎么用，然后退出
    3. 有   → 调用 run.py 开始转换，结果写到【输出】
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 本项目根目录
INPUT = ROOT / "输入"                        # 用户把待转换文件放这里
OUTPUT = ROOT / "输出"                       # 转换结果放这里

SUPPORTED = "PDF / Word / PPT / Excel / 图片 / epub / html"


def _has_mineru(python: str) -> bool:
    """探一下某个 Python 解释器装没装 MinerU。

    用 find_spec 只查"有没有这个包"，**不真的 import** ——
    MinerU 会连带加载 PyTorch，真 import 一次要好几秒，太慢。
    """
    try:
        r = subprocess.run(
            [python, "-c",
             "import importlib.util,sys;"
             "sys.exit(0 if importlib.util.find_spec('mineru') else 1)"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:                                   # noqa: BLE001
        return False


def _conda_env_python(name: str = "pdf2latex") -> str | None:
    r"""在当前解释器所属的 conda 里，找名为 name 的环境；找不到返回 None。

    【为什么需要这一步？】
      Windows 上 PATH 里的 "python" 常常是 conda 的**基础环境**，
      而本项目的依赖装在**名为 pdf2latex 的独立环境**里。
      如果只看 PATH，双击后会拿基础环境去跑 —— 缺依赖直接失败。

    【怎么找的？—— 从当前解释器"推算"，不写死盘符】
      start.bat 用的是 PATH 上的 python，它往往是 conda 的基础环境，
      比如 <conda根目录>\python.exe。
      conda 的环境都放在 <conda根目录>\envs\<环境名>\ 下，
      所以目标就是 <conda根目录>\envs\pdf2latex\python.exe。
      这个"根目录"是从 sys.executable 当场算出来的，不是写死的，
      换台电脑装在别的盘符照样能推对。
    """
    cur = Path(sys.executable).resolve()
    parent = cur.parent

    candidates = []
    # 情形一：当前就是基础环境（<根>\python.exe）
    #         → <根>/envs/<name>/python.exe
    candidates.append(parent / "envs" / name / "python.exe")
    candidates.append(parent / "envs" / name / "bin" / "python")
    # 情形二：当前已在某个环境里（<根>\envs\<别的名字>\python.exe）
    #         → 往上退两层就是 conda 根
    candidates.append(parent.parent.parent / "envs" / name / "python.exe")
    # 情形三：已经激活过 conda，直接用它的环境变量
    for var in ("CONDA_ROOT", "CONDA_PREFIX"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(Path(v) / "envs" / name / "python.exe")

    for p in candidates:
        if p.exists() and _has_mineru(str(p)):
            return str(p)
    return None


def find_python() -> str:
    """找到该用哪个 Python 解释器去跑 run.py。

    三级优先，既能在本机双击就跑，也不写死任何盘符路径：

      1. 环境变量 PDF2LATEX_PYTHON —— 显式指定，最省事
      2. 当前解释器**已经装了 MinerU** → 直接用它
      3. 从当前解释器推算出 conda 根目录，去里面找名为 pdf2latex 的环境
         （见 _conda_env_python，全程不写死盘符）
      4. 都不行 → 退回当前解释器，让 run.py 去报"缺依赖"并给出安装提示
    """
    # 1. 用户显式指定
    env = os.environ.get("PDF2LATEX_PYTHON", "").strip()
    if env and Path(env).exists():
        return env

    # 2. 当前解释器就能用
    if _has_mineru(sys.executable):
        return sys.executable

    # 3. 推算 conda 环境位置
    found = _conda_env_python("pdf2latex")
    if found:
        return found

    # 4. 兜底：让 run.py 自己去报缺什么依赖
    return sys.executable


def main() -> int:
    """程序入口。返回 0 表示成功，非 0 表示失败。"""

    # ---- 准备：把控制台切成 UTF-8，否则中文会乱码 ----
    if sys.platform == "win32":
        try:
            # chcp 65001 = 把控制台代码页切成 UTF-8
            subprocess.run("chcp 65001", shell=True,
                           capture_output=True, check=False)
        except Exception:                               # noqa: BLE001
            pass                    # 切不了就算了，不至于因此跑不动

    # 和 common.py 里的日志设置保持一致：两边都是 UTF-8 才不会乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- 打印使用说明 ----
    print()
    print("=" * 62)
    print("   PDF 转 LaTeX   （含 OCR / 公式 / 图片）")
    print("=" * 62)
    print()
    print("   用法：")
    print("     1. 把要转换的文件放进【输入】文件夹")
    print(f"        （支持：{SUPPORTED}）")
    print("     2. 双击本文件（或 start.bat）")
    print("     3. 结果在【输出】文件夹")
    print()

    # ---- 检查【输入】文件夹 ----
    INPUT.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)

    # 排除 ~$ 开头的文件 —— 那是 Office 打开文档时生成的临时锁文件，不是真文档
    files = [p for p in INPUT.iterdir()
             if p.is_file() and not p.name.startswith("~$")]

    if not files:
        print("   [!] 【输入】文件夹是空的")
        print(f"       位置：{INPUT}")
        print("       请先把要转换的文件放进去，再运行一次")
        print()
        return 1

    print(f"   在【输入】里找到 {len(files)} 个文件，开始转换……")
    print("   （第一次跑要加载模型，会慢一些；之后每页约 1-2 秒）")
    print()

    # ---- 调用主程序 run.py 开始干活 ----
    py = find_python()
    cmd = [py, str(ROOT / "run.py"), "--input", str(INPUT), "--out", str(OUTPUT)]
    proc = subprocess.run(cmd, cwd=str(ROOT))

    # ---- 跑完后告诉用户结果在哪儿 ----
    print()
    print("=" * 62)
    if proc.returncode == 0:
        print("   完成！结果在【输出】文件夹：")
        print(f"     {OUTPUT}")
        print()
        print("   每份文档一个子文件夹，里面有：")
        print("     · {文档名}.md    给 AI 大模型读（公式是 LaTeX 代码）")
        print("     · {文档名}.tex   给 LaTeX 用，可以编译成 PDF")
        print("     · images/        从 PDF 里抠出来的图片")
        print()
        print("   如果发现识别错误，可以改 review\\ 里的批注文件，然后执行：")
        print("     python run.py --input 输入 --redo assemble")
    else:
        print(f"   出错了（退出码 {proc.returncode}）")
        print("   把上面的报错信息截图发给开发者")
    print("=" * 62)
    print()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
