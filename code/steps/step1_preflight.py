# -*- coding: utf-8 -*-
"""
step1_preflight.py —— 第①步：预检（先摸清这份 PDF 是什么货色）

【做什么】文字版 PDF 的字能直接复制出来（快又准），扫描件本质是一堆照片、
    必须先 OCR（慢且会认错）—— 两者处理路径完全不同。麻烦的是同一个文件夹里
    常常两种混着放，甚至同一份文件前几页是扫描封面、后面是文字正文。
    所以先逐页数"能抽出多少个字"，再决定走哪条路。

【怎么判断】按平均每页净字符数分三档：
    < 30      → 扫描件（扫描页几乎抽不出文字）
    30 ~ 200  → 混合型（有文字层但很稀，多半是扫描时带了隐藏文字层）
    > 200     → 文字版（正常电子排版的 PDF）

【产物】data/{文档名}/01_preflight.json —— 页数、每页字数、判定结果。
    可以打开手工把里面的 kind 改成 scanned，强制走 OCR。

【单独运行】python steps/step1_preflight.py 某文件.pdf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 允许"直接运行本文件"（而不是被 run.py 调用）时也能找到上级目录的模块。
# 判断依据：__package__ 为空说明是当脚本直接跑的，不是被 import 的。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf                # 读写 PDF 的库（能逐页取文字、数图片）

from steps.common import (
    doc_dir, get_logger, list_pdfs, load_json, safe_name, save_json, step_banner,
)

# 两个判断门槛。想调整松紧，改这两个数字就行，别的代码不用动。
SCANNED_CHARS_PER_PAGE = 30    # 低于这个数 ⇒ 当成扫描件
MIXED_CHARS_PER_PAGE = 200     # 低于这个数 ⇒ 当成混合型；再高就是文字版


def inspect_one(pdf: Path) -> dict:
    """检查**一份** PDF，返回一份体检报告（一个字典）。

    输入：PDF 文件路径
    输出：形如下面的字典 ——
        {
          "name": "年报",             文档名
          "pages": 42,                总页数
          "avg_chars_per_page": 850.3, 平均每页多少个字
          "kind": "text",             判定：text 文字版 / mixed 混合 / scanned 扫描件
          "needs_ocr": False,         要不要上 OCR
          "scanned_pages": [3, 7],    哪几页是扫描页（正文里夹的插图页）
          "total_images": 12,         一共多少张图
          "page_detail": [...],       每一页的明细（页码、字数、图片数、尺寸）
        }
    """
    doc = pymupdf.open(pdf)
    pages = []
    try:
        # 一页一页地数：这页有多少字、多少图、多大尺寸
        for i, page in enumerate(doc, 1):
            text = page.get_text("text") or ""

            # 数"净字符数"：先把所有空白（空格、换行、制表符）去掉再数字数。
            # 为什么？因为扫描页常常抽出一大堆空白和换行，
            # 不清理的话会被误判成"有文字"。
            n_chars = len("".join(text.split()))

            n_images = len(page.get_images(full=True))
            pages.append({
                "page": i,
                "chars": n_chars,
                "images": n_images,
                "width": round(page.rect.width, 1),    # 页面宽（磅，72磅=1英寸）
                "height": round(page.rect.height, 1),
            })

        n_page = doc.page_count
        # 顺带记一下 PDF 自带的属性信息，排查问题时有用
        meta = {
            "title": (doc.metadata or {}).get("title", "") or "",
            "producer": (doc.metadata or {}).get("producer", "") or "",  # 用什么软件生成的
            "encrypted": bool(doc.is_encrypted),   # 是否加密（加密的可能读不了）
        }
    finally:
        # 不管上面成不成功，都要把文件关掉，否则文件会被锁住
        doc.close()

    # ---- 算平均字数，给出整体判断 ----
    total_chars = sum(p["chars"] for p in pages)
    avg = total_chars / max(n_page, 1)     # max(...,1) 是防止 0 页时除以零报错

    if avg < SCANNED_CHARS_PER_PAGE:
        kind = "scanned"
    elif avg < MIXED_CHARS_PER_PAGE:
        kind = "mixed"
    else:
        kind = "text"

    # 再单独挑出"扫描页"的页码。
    # 为什么整份判成文字版还要挑？因为正文里常夹着扫描的插图页 / 手写页，
    # 这些页单独需要 OCR。列出来是为了提醒后续步骤注意。
    scanned_pages = [
        p["page"] for p in pages if p["chars"] < SCANNED_CHARS_PER_PAGE
    ]

    return {
        "file": str(pdf),                                    # 原始文件完整路径
        "name": safe_name(pdf.stem),                          # 文档名（洗过，Windows 安全）
        "pages": n_page,                                     # 总页数
        "total_chars": total_chars,                          # 全文总字数
        "avg_chars_per_page": round(avg, 1),                 # 平均每页字数
        "kind": kind,                                        # ★ 核心结论
        "needs_ocr": kind in ("scanned", "mixed"),           # ★ 要不要 OCR
        "scanned_pages": scanned_pages,                      # 疑似扫描的页码
        "total_images": sum(p["images"] for p in pages),     # 图片总数
        "meta": meta,                                        # PDF 属性
        "page_detail": pages,                                # 每页明细
    }


def run(src: Path, force: bool = False) -> list[dict]:
    """处理 src 下的所有文档。这是给 run.py 调用的入口。

    参数：
        src   —— 文件或文件夹路径
        force —— True 表示"忽略上次的结果，全部重新检查一遍"
                 不传（False）的话，已有结果就直接复用，省时间

    返回：每份文档的体检报告组成的列表。
    """
    log = get_logger()
    step_banner(log, "①", "预检", str(src))

    pdfs = list_pdfs(src)
    if not pdfs:
        log.error("没有找到任何 PDF：%s", src)
        return []

    log.info("找到 %d 份 PDF", len(pdfs))
    results = []
    t0 = time.time()

    for k, pdf in enumerate(pdfs, 1):
        # 结果存到 data/{文档名}/01_preflight.json
        out = doc_dir(safe_name(pdf.stem)) / "01_preflight.json"

        # 已有结果且没要求强制重跑 → 直接复用，不重复劳动
        cached = None if force else load_json(out)
        if cached is not None:
            log.info("  [%d/%d] %-40s 复用已有结果（%s）",
                     k, len(pdfs), pdf.name[:40], cached["kind"])
            results.append(cached)
            continue

        # 真正开始检查这一份
        try:
            info = inspect_one(pdf)
        except Exception as exc:                       # noqa: BLE001
            # 单份文件读失败不应该让整批停下，记一笔错误继续处理下一份
            log.error("  [%d/%d] %-40s 读取失败：%s", k, len(pdfs), pdf.name[:40], exc)
            continue

        save_json(out, info)
        results.append(info)

        # 在屏幕上打一行简报：页数、判定类型、平均字数、图片数
        mark = {"text": "文字版", "mixed": "混合型", "scanned": "扫描件"}[info["kind"]]
        log.info("  [%d/%d] %-40s %d页 %s  均%.0f字/页  图%d张%s",
                 k, len(pdfs), pdf.name[:40], info["pages"], mark,
                 info["avg_chars_per_page"], info["total_images"],
                 # 整份是文字版、但个别页是扫描页 → 额外提示一句
                 ("  ⚠ %d页需OCR" % len(info["scanned_pages"]))
                 if info["scanned_pages"] and info["kind"] == "text" else "")

    # ---- 收尾：汇报总计 ----
    dt = time.time() - t0
    n_ocr = sum(1 for r in results if r["needs_ocr"])
    log.info("")
    log.info("预检完成：%d 份 / 耗时 %.1fs", len(results), dt)
    log.info("  文字版 %d 份 · 混合型 %d 份 · 扫描件 %d 份 → 共 %d 份需要 OCR",
             sum(1 for r in results if r["kind"] == "text"),
             sum(1 for r in results if r["kind"] == "mixed"),
             sum(1 for r in results if r["kind"] == "scanned"),
             n_ocr)
    return results


# 直接运行本文件时（python steps/step1_preflight.py）走这里：
# 从命令行取路径，没给就交互式问一句。
if __name__ == "__main__":
    ap_src = sys.argv[1] if len(sys.argv) > 1 else input("PDF 路径: ").strip('"')
    run(Path(ap_src))
