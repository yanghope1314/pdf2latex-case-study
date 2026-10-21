# -*- coding: utf-8 -*-
"""
step2_convert.py —— 第②步：识别（把 PDF 里的内容真正读出来）

【做什么】上一步只知道"是扫描件还是文字版"，这一步真正动手读内容：
    版面分析（哪里是标题/正文/表格）、OCR 文字识别、公式转 LaTeX、
    插图抠成图片文件。

【谁干的活】这四件事都是深度学习的活儿，自己写不现实，交给开源工具包
    **MinerU**（版面分析 + PaddleOCR + 公式识别 + 表格识别，一个包全包）。

【为什么用命令行调用，而不是 import 进来】
    MinerU 是重依赖（带 PyTorch 和一堆模型）。import 进来会让本程序启动变慢，
    而且它一旦崩了整条流水线跟着挂。另起进程调用就隔离了 ——
    它崩了不影响已产出的中间结果，重跑也只重跑这一步。

⚠️ MinerU 4.x 有三个坑，本文件都已处理：
   1) `mineru-kit parse` 与 `mineru parse` 的参数含义不同（后者 -p 是页码不是路径），
      所以这里只调 mineru-kit，不混用；
   2) 不加 --pages 只解析前 10 页 —— 长文档会**静默丢内容**，所以写死 --pages all；
   3) 输出可能是 zip 包，collect 之前要先 unpack_zip。

【产物】
    data/{文档名}/02_convert.json   结果索引
    data/{文档名}/mineru/           原始输出（正文 / middle JSON / 图片）
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps.common import (
    ROOT, doc_dir, get_logger, list_pdfs, load_json, safe_name, save_json,
    step_banner,
)

# ----------------------------------------------------------------------
# 模型文件放哪儿 —— 交给环境变量，代码里不写死任何盘符
# ----------------------------------------------------------------------
# 背景：MinerU 4.x 用 MINERU_HOME 定位整个数据目录（模型约 2 GB）。
#       不设的话会下到当前用户的主目录里 —— 在盘空间紧张的本机上，
#       几份模型就能把系统盘撑满，所以强烈建议指到别处。
#       （这里刻意不写出具体的默认路径：本项目不许在代码里出现任何盘符，
#         即便是"某台 Windows 上的默认长什么样"这种示例也不行。）
#
# 【怎么做】
#   你在系统里设过 MINERU_HOME / HF_HOME / MODELSCOPE_CACHE，
#   这里就把它们原样传给 MinerU 子进程 —— 代码不猜、不覆盖。
#   没设的话就不传，让 MinerU 用它自己的默认位置（会自动下载模型）。
#
#   所以想换模型存放位置，设环境变量即可，不用改代码：
#       Windows：系统设置 → 环境变量 → 新建 MINERU_HOME → 目标目录
#
#   ⚠️ 这里刻意不写死任何盘符路径：写死了换台电脑就会指向不存在的目录，
#      而代码「看起来配置好了」，反而更难排查。
_PASSTHROUGH_KEYS = ("MINERU_HOME", "HF_HOME", "MODELSCOPE_CACHE")


def build_mineru_env() -> dict:
    """组装要传给 MinerU 子进程的环境变量。规则见上面注释。"""
    out: dict[str, str] = {}

    for key in _PASSTHROUGH_KEYS:
        current = os.environ.get(key, "").strip()
        if current:
            out[key] = current          # 用户设过 → 原样透传
        # 没设 → 不传这一项，交给 MinerU 默认

    # 优先从魔搭（ModelScope）下载 —— 国内网络比 HuggingFace 稳
    out["MINERU_MODEL_SOURCE"] = os.environ.get("MINERU_MODEL_SOURCE", "modelscope")
    out["PYTHONIOENCODING"] = "utf-8"   # 防止子进程中文乱码
    return out


MINERU_ENV = build_mineru_env()

# 第①步的预检结论 → 翻译成 MinerU 要的 --ocr-mode 参数
OCR_MODE_BY_KIND = {
    "text": "txt",      # 文字版：直接抽文字+公式，不跑 OCR（快且准）
    "mixed": "auto",    # 混合型：交给 MinerU 自己逐页判断
    "scanned": "ocr",   # 扫描件：全量 OCR（慢，但非做不可）
}


def find_kit() -> str | None:
    """找到 mineru-kit 这个可执行文件，返回它的路径；找不到返回 None。

    ⚠️ 坑：在 conda 环境里，可执行文件放在 python.exe 同级的 **Scripts/** 子目录，
    并不在 python.exe 旁边。所以下面要把两种位置都试一遍。
    """
    base = Path(sys.executable).parent
    candidates = [
        base / "mineru-kit.exe", base / "mineru-kit",
        base / "Scripts" / "mineru-kit.exe", base / "Scripts" / "mineru-kit",
        base / "bin" / "mineru-kit",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # 上面都没找到，最后试试系统 PATH 里有没有
    return shutil.which("mineru-kit")


def server_status(kit: str) -> str:
    """查一下 MinerU 的后台服务在不在。

    注：这个只是"看一眼状态"，服务没起也不影响 parse 正常工作，
    所以失败了也无所谓，返回空字符串就行（外面不会因为这个报错）。
    """
    try:
        r = subprocess.run([kit.replace("mineru-kit", "mineru"), "server", "status"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30)
        return (r.stdout or "").strip()
    except Exception:                                       # noqa: BLE001
        return ""


def call_kit(kit: str, pdf: Path, outdir: Path, ocr_mode: str,
             extra: list[str], log, pages: str = "all") -> tuple[int, str]:
    """真正去调用 MinerU 干活。返回 (退出码, 最后几行输出)。

    参数说明：
        kit      —— mineru-kit 的路径
        pdf      —— 要处理的文件
        outdir   —— 结果放哪个目录
        ocr_mode —— txt / auto / ocr，由第①步的预检结论决定
        pages    —— "all" 或 "1-5,8" 这种页码范围

    ⚠️ pages 默认必须是 "all"！
       MinerU 的 --pages 不加的话**只解析前 10 页**，
       一本 200 页的书只会转出前 10 页，而且**不报错**（静默丢内容，踩过）。
    """
    cmd = [
        kit, "parse", str(pdf),
        "-o", str(outdir),
        "--pages", pages,          # ★ 必须显式传，否则只处理前 10 页
        "--ocr-mode", ocr_mode,
        # ★ 用 zip 而不是 markdown：markdown 模式把图片塞成 base64 写在正文里，
        #   而且不产出 middle JSON（第④步的 R6 就没数据可查）。zip 一次给全：
        #   正文 + middle_json.json + images/。两种模式正文实测等价。
        #   ⚠️ -f 是单值参数，写两次只有最后一个生效，不能同时要两种。
        "-f", "zip",
        "-v",                      # 详细输出，出问题时能看到原因
    ] + extra                       # 允许外面追加额外参数
    log.debug("    $ %s", " ".join(cmd))   # 只写进日志文件，不刷屏幕

    # 把模型路径等环境变量传给子进程
    env = dict(os.environ)
    env.update(MINERU_ENV)

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, cwd=str(ROOT))

    # 只留最后 15 行输出 —— 前面的刷屏信息没用，报错原因通常在结尾
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-15:])
    if proc.returncode != 0:
        # 失败了才把 stderr 也带上，方便排查
        tail += "\n[stderr]\n" + "\n".join(
            (proc.stderr or "").strip().splitlines()[-15:])
    return proc.returncode, tail


def unpack_zip(workdir: Path, log) -> int:
    """把 MinerU 产出的 .zip 解压到同一个目录，返回解压出几个文件。

    【为什么要解压？】
      上面用 `-f zip` 让 MinerU 把结果打包成一个 zip，里面有：
          markdown.md        正文（图片用相对路径引用，不是 base64）
          middle_json.json   每页的结构信息（R6 靠它核对页数）
          images/            图片文件
      解压出来后续步骤才能按普通文件读取。

    【出错怎么办】解压失败就记错误、返回 0，让上层按"没有产物"处理 ——
      不静默继续（否则会产出"看起来跑完了但数据是坏的"结果）。
    """
    import zipfile

    zips = sorted(workdir.rglob("*.zip"))
    if not zips:
        return 0

    n = 0
    for z in zips:
        try:
            with zipfile.ZipFile(z) as f:
                f.extractall(workdir)
                n += len(f.namelist())
            log.debug("    解压 %s → %d 个文件", z.name, n)
        except Exception as exc:                            # noqa: BLE001
            log.error("    解压失败 %s：%s", z.name, exc)
            return 0
    return n


def collect_outputs(workdir: Path) -> dict:
    """在 MinerU 的输出目录里，把我们要的东西找出来。

    MinerU 的输出结构在不同版本里会变，所以这里用"递归搜索"而不是写死路径 ——
    不管它怎么改目录层级，都能找到 .md 和图片。
    """
    mds = sorted(workdir.rglob("*.md"))                     # Markdown 正文
    imgs = [p for p in workdir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    middles = sorted(workdir.rglob("*middle*.json"))        # 中间结构文件（备用）
    return {
        "markdown": str(mds[0]) if mds else None,
        "n_markdown": len(mds),
        "images": sorted(str(p) for p in imgs),
        "middle": str(middles[0]) if middles else None,
    }


def stage_input(pdf: Path, ddir: Path, log) -> Path:
    """文件名对 Windows 不友好时，先复制一份干净名字的，再交给 MinerU。

    【为什么需要这一步】
    MinerU 内部拿**输入文件名**去拼它自己的输出目录名，原样照搬 ——
    包括文件名结尾多打的那个空格。
    而 Windows 不允许目录名以空格结尾：
    mkdir 看着成功了，等它往里写 markdown.md 时才炸，报错长这样：

        解析 xxx.pdf 失败: No such file or directory:
        '...\\\\mineru\\\\.xxx \\\\markdown.md'

    这个错发生在 MinerU 内部，本工具改不动它。
    能做的只有一件事：**别把带毛病的文件名递进去**。

    做法：洗过的名字和原名不一样，就复制一份到中间结果目录里再喂。
    代价是多占一份源文件的体积，换 MinerU 能跑通。
    名字本来就干净的文件走原路径，零开销。
    """
    safe = safe_name(pdf.stem)
    if safe == pdf.stem:
        return pdf                                  # 名字本来就干净，直接用
    stage_dir = ddir / "_src"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / f"{safe}{pdf.suffix}"
    if not staged.exists() or staged.stat().st_size != pdf.stat().st_size:
        shutil.copy2(pdf, staged)
    log.info("        文件名带空格或特殊字符，已复制成安全名再交给 MinerU：%s", safe)
    return staged


def run(src: Path, device: str = "cuda", lang: str = "ch",
        force: bool = False, extra: list[str] | None = None,
        pages: str = "all") -> list[dict]:
    """处理 src 下的所有文档。这是给 run.py 调用的入口。

    参数：
        src    —— 文件或文件夹路径
        device —— cuda（用显卡，快）或 cpu（没显卡时用，慢很多）
        lang   —— ch 中文 / en 英文
        force  —— True 表示忽略已有结果重新识别
        pages  —— "all" 或 "1-5,8"，调试时可以只跑几页

    返回：每份文档的识别结果列表。
    """
    log = get_logger()
    step_banner(log, "②", "转换（版面 / OCR / 公式 / 图片）")

    pdfs = list_pdfs(src)
    if not pdfs:
        raise RuntimeError(f"没有找到任何 PDF：{src}")

    # 先确认 MinerU 装好了，免得跑到一半才发现白忙
    kit = find_kit()
    if kit is None:
        log.error("找不到 mineru-kit —— 依赖没装好？")
        log.error("  修复：在你当前用的 Python 环境里执行")
        log.error('        python -m pip install -U "mineru[core]"')
        log.error("  或者先激活装好 MinerU 的 conda 环境再运行本程序")
        raise RuntimeError("找不到 mineru-kit 可执行文件")

    log.info("引擎 : %s", kit)
    log.info("模型 : HF_HOME=%s",
             MINERU_ENV.get("HF_HOME", "（未特别设置，用 MinerU 默认位置）"))
    log.info("共 %d 份 PDF", len(pdfs))

    extra = extra or []
    results, t0 = [], time.time()

    for k, pdf in enumerate(pdfs, 1):
        ddir = doc_dir(safe_name(pdf.stem))
        out_json = ddir / "02_convert.json"

        # 已经有成功结果 → 跳过，不重复花钱花时间
        cached = None if force else load_json(out_json)
        if cached is not None and cached.get("markdown"):
            log.info("  [%d/%d] %-38s 已有结果，跳过", k, len(pdfs), pdf.name[:38])
            results.append(cached)
            continue

        # 读第①步的预检结论，决定用哪种 OCR 模式
        # （读不到就默认按文字版处理，至少不会因为缺文件直接崩）
        pre = load_json(ddir / "01_preflight.json") or {}
        kind = pre.get("kind", "text")
        mode = OCR_MODE_BY_KIND.get(kind, "auto")

        workdir = ddir / "mineru"
        # ★ 强制重跑（--force）时先清空这一份的识别目录，否则 MinerU 的
        #   图片/中间结果会和上一轮混在一起，收集阶段拿到的可能是旧文件。
        if force and workdir.exists():
            shutil.rmtree(workdir)
            log.info("        已清空上一轮的识别目录（--force 重跑不叠加）")
        workdir.mkdir(parents=True, exist_ok=True)

        log.info("  [%d/%d] %-38s %s → --ocr-mode %s",
                 k, len(pdfs), pdf.name[:38], kind, mode)

        t1 = time.time()
        code, tail = call_kit(kit, stage_input(pdf, ddir, log), workdir,
                              mode, extra, log, pages=pages)
        dt = time.time() - t1

        # ---- 失败的情况：把错误记进 JSON，继续处理下一份 ----
        if code != 0:
            log.error("        失败（退出码 %d，%.0fs）", code, dt)
            for ln in tail.splitlines():
                log.error("        %s", ln)
            save_json(out_json, {
                "file": str(pdf), "name": safe_name(pdf.stem), "kind": kind,
                "ocr_mode": mode, "ok": False, "exit_code": code,
                "seconds": round(dt, 1), "error_tail": tail,
            })
            continue

        # ---- 成功的情况：先解压（-f zip 会打成包），再清点产物 ----
        unpack_zip(workdir, log)
        got = collect_outputs(workdir)
        rec = {
            "file": str(pdf), "name": safe_name(pdf.stem),
            "kind": kind, "ocr_mode": mode,
            "ok": bool(got["markdown"]),          # 有 md 才算真成功
            "seconds": round(dt, 1),
            "markdown": got["markdown"],          # 下一步要读这个文件
            "n_images": len(got["images"]),
            "middle": got["middle"],
        }
        save_json(out_json, rec)
        results.append(rec)
        log.info("        OK  %.0fs  图 %d 张", dt, rec["n_images"])

    ok = sum(1 for r in results if r.get("ok"))
    log.info("")
    log.info("转换完成：%d/%d 成功 / 总耗时 %.0fs", ok, len(results), time.time() - t0)
    return results


# 直接运行本文件时走这里
if __name__ == "__main__":
    a = sys.argv
    p = Path(a[1]) if len(a) > 1 else Path(input("PDF 路径: ").strip('"'))
    run(p)
