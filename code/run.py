# -*- coding: utf-8 -*-
"""
run.py —— 主程序（总指挥，自己不干活，只按顺序叫各个步骤）

    python run.py --input <PDF文件或目录>        # 全部参数见 --help
    python run.py --input <PDF目录> --redo assemble   # 改完批注后从某步接着跑
    python run.py --input <PDF目录> --only analyze --preview   # 只看提示词，不调模型

六个步骤分别写在 steps/ 下的六个文件里：
    ① 预检 → ② 识别 → ③ 组装 → ④ 质检 → ⑤ L2 抽取 → ⑥ LLM 分析

前四步把 PDF 变成 Markdown（Doc / L1），是上次作业的成果。
⑤ 用 config/setting.yaml 里写死的规则，从 L1 抽出结构化证据（L2）——
纯代码、确定性、可复现，不调用任何模型。
⑥ 把 L2 送进大模型做判断，产出结论。

⑤⑥ 都不吃 --input 指定的目录，而是按配置里的「文档关键字」
去 out/ 里找目标文档，所以必须先跑完前四步。

为什么要有 --redo：每一步的结果都会存到硬盘（data/ 目录），
所以"改哪一层，就从哪一层重新跑"，不用整本重做。
这是本工具能人工纠错的基础。详细用法见 python run.py --help；整条流水线的设计见 实验报告/实验报告.tex 第 2 节。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---- 把项目根目录加进模块搜索路径 ----
# 这样无论从哪个目录执行，都能 import 到 steps/ 里的模块
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steps.common import (                                       # noqa: E402
    DATA, LOGS, OUT, REVIEW, STEPS, STEP_TITLES, get_logger, list_pdfs,
    steps_from,
)

# 启动时打印的大字标题（纯装饰，让人知道程序跑起来了）
BANNER = r"""
  ____  ____  _____ _    _   _  _____  _        _    __  __
 |  _ \|  _ \|  ___| |  | | | ||_   _|| |      / \   \ \/ /
 | |_) | | | | |_  | |__| |_| |  | |  | |     / _ \   \  /
 |  __/| |_| |  _| |  __  |\__/  | |  | |___ / ___ \ /  \
 |_|   |____/|_|   |_| |_|     |_|  |_____/_/   \_\_/\_\

  PDF → LaTeX 批量转换 · 含 OCR / 公式 / 图片
"""


def main(argv: list[str] | None = None) -> int:
    """程序入口。返回值 0 表示全部成功，1 表示有失败。

    argv 参数是为了方便测试（可以不传，默认读命令行）。
    """
    # 下面 --out 会修改模块级的 OUT 变量，所以要先声明一下
    global OUT

    # ==================================================================
    # 一、定义有哪些命令行参数
    # ==================================================================
    ap = argparse.ArgumentParser(
        description="PDF → LaTeX 批量转换流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", "-i", required=True, help="PDF 文件或目录")
    ap.add_argument("--out", "-o", default=None, help="产物输出目录（默认 ./out）")
    ap.add_argument("--redo", choices=STEPS, default=None,
                    help="从该步开始重跑（前面的产物复用）")
    ap.add_argument("--only", choices=STEPS, default=None, help="只跑这一步")
    ap.add_argument("--pages", default="all",
                    help="只处理指定页，如 '1-5,8' 或 'all'（默认全部）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="推理设备（默认 cuda 用显卡）")
    ap.add_argument("--lang", default="ch", choices=["ch", "en"],
                    help="文档语言（默认 ch 中文）")
    ap.add_argument("--force", action="store_true", help="忽略已有产物，强制重跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行什么")
    ap.add_argument("--preview", action="store_true",
                    help="只把要送给大模型的提示词存盘，不调用模型"
                         "（配合 --only analyze 使用，方便核对送进去的是什么）")
    args = ap.parse_args(argv)

    log = get_logger()
    print(BANNER)

    # ---- 检查输入 ----
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        log.error("路径不存在：%s", src)
        return 1

    pdfs = list_pdfs(src)
    if not pdfs:
        log.error("该路径下没有可处理的文档（PDF/Word/PPT/Excel/图片）：%s", src)
        return 1

    # ---- 决定这次跑哪几步 ----
    if args.only:
        todo = [args.only]                 # --only：只跑指定的一步
    elif args.redo:
        todo = steps_from(args.redo)       # --redo：从那一步开始往后的都跑
    else:
        todo = list(STEPS)                 # 默认：四步全跑

    # 把这次要做什么告诉用户，避免"跑完了不知道跑了啥"
    log.info("输入    : %s", src)
    log.info("文档数量: %d", len(pdfs))
    log.info("输出    : %s", Path(args.out).resolve() if args.out else OUT)
    log.info("设备    : %s", args.device)
    log.info("将执行  : %s", " → ".join(todo))
    if args.redo:
        log.info("           （前面步骤的产物复用，不重做）")

    # --dry-run：只说不做，让人先确认一遍
    if args.dry_run:
        log.info("")
        log.info("[dry-run] 不实际执行。各步说明：")
        for s in todo:
            log.info("  · %-10s %s", s, STEP_TITLES[s])
        return 0

    # ---- 用户指定了输出目录：把全局 OUT 改掉 ----
    # OUT 定义在 common.py，其他模块直接引用它，所以要逐个改到才生效
    if args.out:
        OUT = Path(args.out).expanduser().resolve()
        OUT.mkdir(parents=True, exist_ok=True)
        import steps.common as _c
        _c.OUT = OUT
        import steps.step3_assemble as _a
        _a.OUT = OUT

    # ---- 按顺序一步步执行 ----
    t_all = time.time()
    done: list[tuple[str, str, float]] = []      # 记录每步的(名字, 结果, 耗时)

    for name in todo:
        t0 = time.time()
        try:
            if name == "preflight":
                from steps.step1_preflight import run as f
                f(src, force=args.force)
            elif name == "convert":
                from steps.step2_convert import run as f
                f(src, device=args.device, lang=args.lang, force=args.force,
                  pages=args.pages)
            elif name == "assemble":
                from steps.step3_assemble import run as f
                f(src, force=args.force)
            elif name == "review":
                from steps.step4_review import run as f
                f(src)
            elif name == "extract":
                from steps.step5_extract import run as f
                # 这一步按 config/setting.yaml 里的「文档关键字」去 out/ 里
                # 找目标文档，不吃 --input 指定的目录。
                # 所以先跑前四步转换 PDF，这一步才有东西可抽。
                f(src)
            elif name == "analyze":
                from steps.step6_analyze import run as f
                f(src, dry_run=bool(args.preview))
            else:
                raise ValueError(f"未知步骤 {name}")
            done.append((name, "成功", time.time() - t0))

        except KeyboardInterrupt:
            log.warning("用户中断")
            done.append((name, "中断", time.time() - t0))
            break

        except Exception as exc:                          # noqa: BLE001
            # ★ 某一步失败就停下，不往下继续 ——
            #   后面步骤依赖前面的结果，硬跑下去会产出一份
            #   "看起来跑完了、其实数据是坏的"结果，比直接报错危险得多。
            import traceback
            log.error("步骤「%s」失败：%s", name, exc)
            log.debug(traceback.format_exc())      # 详细堆栈写进日志文件
            done.append((name, "失败", time.time() - t0))
            break

    # ---- 打印执行汇总 ----
    log.info("")
    log.info("=" * 68)
    log.info("  执行汇总")
    log.info("=" * 68)
    for name, status, dt in done:
        mark = {"成功": "OK  ", "失败": "FAIL", "中断": "STOP"}.get(status, "?   ")
        log.info("  %s  %-10s %-36s %6.1fs", mark, name, STEP_TITLES[name], dt)
    log.info("")
    log.info("  总耗时：%.1fs", time.time() - t_all)
    log.info("  产物目录：%s", OUT)
    log.info("  批注目录：%s", REVIEW)
    log.info("  日志目录：%s", LOGS)
    log.info("=" * 68)

    # 有失败就返回 1，让调用方（比如 launcher.py）知道出错了
    n_fail = sum(1 for _, s, _ in done if s == "失败")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
