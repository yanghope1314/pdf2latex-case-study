# -*- coding: utf-8 -*-
"""
step4_review.py —— 第④步：质检（把"机器没把握的地方"挑出来给人看）

【做什么】识别一定会出错，而且错得悄无声息 —— 一份 200 页的 PDF 转完，
    你不可能逐页核对。所以这一步用六条简单规则扫一遍，
    把可疑处写成 review/{文档名}.yaml 批注文件，供人核对修改。
    批注文件的两个关键字段是 action 与 corrected：把改好的内容填进 corrected，
    再执行 run.py --redo assemble 就会回灌。判断依据是 corrected 有没有内容，
    不是 action 是不是 fix —— 组装每次都从原始识别结果重来，靠状态字段会在第二轮丢修正。

【六条规则】
    R1  公式大括号不配对        —— 左 { 和右 } 数量对不上
    R2  公式里有可疑命令        —— 出现不像正常 LaTeX 命令的长单词
    R3  行内公式异常长          —— 超过 200 字符，多半把整段话误当成公式
    R4  图片引用指向不存在的文件
    R5  整页无产出              —— 有图却一个字没认出来（OCR 失败的典型症状）
    R6  页数对不上              —— 原文档 200 页，解析结果只有 10 页

    R1~R5 关心"识别得对不对"，R6 关心"内容全不全"。
    内容漏掉不可逆，所以 R6 最要紧。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps.common import (
    DATA, OUT, REVIEW, doc_dir, get_logger, load_json, safe_name, save_json,
    step_banner,
)

try:
    import yaml
    _HAS_YAML = True
except ImportError:                                     # noqa: N816
    _HAS_YAML = False

# 三个常用的"文本模式"，先编译好，后面反复用会快一些
MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)                    # 独立公式块 $$...$$
MATH_INLINE = re.compile(r"(?<!\$)\$([^$\n]{1,400})\$(?!\$)")      # 行内公式 $...$
IMG_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")                    # 图片引用 ![](...)


# ======================================================================
# 五条检查规则 —— 每条都是一个"输入文本、输出可疑点列表"的小函数
# ======================================================================
def r1_brace_mismatch(tex: str) -> list[dict]:
    """R1：检查公式里的大括号配不配对。

    LaTeX 里 { } 必须成对出现，像 \\frac{分子}{分母}。
    如果识别时漏了一个，公式就编译不出来 —— 这是最常见的错误。

    做法：数一数左括号和右括号的个数是否相等。
    """
    hits = []
    for m in MATH_BLOCK.finditer(tex):
        body = m.group(1)
        if body.count("{") != body.count("}"):
            hits.append({
                "type": "formula", "issue": "大括号不配对",
                "found": body.strip()[:120],       # 只取前 120 字，够定位就行
                "hint": f"{{ ×{body.count('{')} vs }} ×{body.count('}')}",
            })
    return hits


def r2_suspect_cmd(tex: str) -> list[dict]:
    """R2：检查公式里有没有"长得可疑"的 LaTeX 命令。

    正常的 LaTeX 命令都很短（\\alpha、\\frac、\\sum）。
    如果冒出一个很长的怪词（比如 \\alphabetax），多半是识别错了。

    做法：找出所有 8 个字母以上的命令名，对照一张"已知正常"的名单，
          不在名单里的就标出来。
    """
    hits = []
    for m in MATH_BLOCK.finditer(tex):
        body = m.group(1)

        # 连续两个以上反斜杠，通常是识别断裂留下的痕迹
        if re.search(r"\\\\{2,}", body):
            hits.append({"type": "formula", "issue": "连续反斜杠（疑似断裂）",
                         "found": body.strip()[:120], "hint": "检查是否漏识别"})

        # 把长命令名挑出来，跟已知名单比对
        for cmd in set(re.findall(r"\\([A-Za-z]{8,})", body)):
            if cmd.lower() not in {
                "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
                "begin", "end", "operatorname", "left", "right", "limits",
                "nolimits", "boldsymbol", "mathbb", "mathcal", "mathrm",
            }:
                hits.append({"type": "formula", "issue": f"可疑命令 \\{cmd}",
                             "found": body.strip()[:120],
                             "hint": "可能是识别错的命令名"})
    return hits


def r3_long_inline(tex: str) -> list[dict]:
    """R3：检查行内公式是不是长得离谱。

    正常的行内公式都很短，比如 $x^2$、$\\alpha$。
    如果一个行内公式超过 200 个字符，多半是识别时把一整段话
    都当成公式圈进去了 —— 那样正文就没了。
    """
    return [
        {"type": "formula", "issue": "行内公式异常长（可能把整段吞进公式）",
         "found": m.group(1)[:120], "hint": f"{len(m.group(1))} 字符"}
        for m in MATH_INLINE.finditer(tex) if len(m.group(1)) > 200
    ]


def r4_missing_image(md: str, base: Path) -> list[dict]:
    """R4：检查正文里引用的图片，文件是不是真的存在。

    因为图片路径在转换过程中被改写了好几次，万一哪步出错，
    正文里就会出现"指向一个不存在的图片"的引用 —— 编译时才发现就晚了。

    参数 base 是正文文件所在目录，用来拼出图片的完整路径去检查。
    """
    hits = []
    for m in IMG_REF.finditer(md):
        rel = m.group(1)
        # 网络图片跳过（那是外链，本机当然没有）
        if rel.startswith(("http://", "https://")):
            continue
        p = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        if not p.exists():
            hits.append({"type": "image", "issue": "图片文件不存在",
                         "found": rel, "hint": "图片可能在转换中丢失"})
    return hits


def r5_empty_page(pre: dict | None, md: str) -> list[dict]:
    """R5：检查是不是"整页什么都没认出来"。

    判断依据：第①步的预检说这份文件里有图，但识别结果却不到 80 个字。
    这几乎可以断定是 OCR 失败了（比如该用 OCR 模式却用了文字模式）。

    参数 pre 是第①步的预检结果（可能没有，所以要先判空）。
    """
    if not pre:
        return []
    img_pages = sum(1 for p in pre.get("page_detail", []) if p.get("images", 0) > 0)
    if img_pages > 0 and len(md.strip()) < 80:
        return [{"type": "page", "issue": "疑似识别失败",
                 "found": f"{pre.get('pages', '?')} 页里 {img_pages} 页含图，但输出不足 80 字",
                 "hint": "该文件可能需要改用 ocr 模式重跑"}]
    return []


def r6_page_coverage(pre: dict | None, conv: dict | None) -> list[dict]:
    """R6：核对"实际解析出的页数"和"原文档页数"对不对得上。

    怎么查：第①步预检数过原文档页数（pre["pages"]），第②步的 middle.json
            里记着实际解析出多少页。两个数一比，还能算出具体缺了哪几页。

    为什么不报警：用户主动用 --pages "1-5" 只跑前几页时页数当然对不上，
            那是故意的。MinerU 用 is_full_document=False 标记这种情况，据此跳过。
            （认不出 middle.json 结构时也跳过 —— 宁可漏报，不可误报。）

    输入：pre = 第①步结果；conv = 第②步结果（含 middle.json 路径）
    输出：可疑点列表（正常是空列表）
    """
    if not pre or not conv:
        return []

    mpath = conv.get("middle")
    if not mpath or not Path(mpath).exists():
        return []          # 没有中间文件，查不了 —— 不误报

    try:
        mj = json.loads(Path(mpath).read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return []

    # 认结构：middle.json 顶层是 {"pages": [{"page_idx": 0, "blocks": [...]}, ...]}
    # 认不出来就跳过 —— 宁可漏报，也不能误报（误报会让批注文件失去可信度）
    pages = mj.get("pages") if isinstance(mj, dict) else None
    if not isinstance(pages, list) or not pages:
        return []

    # 用户自己只跑了一部分页 → 页数对不上是正常的，不报警
    if mj.get("is_full_document") is False:
        return []

    expected = pre.get("pages")
    if not isinstance(expected, int) or expected <= 0:
        return []

    got = len(pages)
    if got >= expected:
        return []          # 页数够，没问题

    # 能拿到 page_idx 的话，把具体缺了哪几页算出来（页码从 1 开始显示）
    got_idx = {pg["page_idx"] for pg in pages
               if isinstance(pg, dict) and isinstance(pg.get("page_idx"), int)}
    if got_idx:
        missing = [i + 1 for i in range(expected) if i not in got_idx]
        detail = ("缺失页码：" + ",".join(str(x) for x in missing[:20])
                  + ("…" if len(missing) > 20 else ""))
    else:
        detail = f"实际只解析出 {got} 页"

    return [{
        "type": "page",
        "issue": f"内容可能缺失：原文档 {expected} 页，只解析出 {got} 页",
        "found": f"预检 {expected} 页 → 解析结果 {got} 页",
        "hint": detail + " ｜ 确认后执行 --redo convert 重跑（该文件可能需要 ocr 模式）",
    }]


# 打印给用户看的规则清单
RULES = "R1大括号 R2可疑命令 R3超长行内公式 R4图片缺失 R5整页无产出 R6页数覆盖"


# ======================================================================
# 合并人工修改 —— 保证你改过的东西不会被下次重跑冲掉
# ======================================================================
def load_manual(name: str) -> dict[str, dict]:
    """读已有的批注文件，把人工填过的内容记下来。

    【为什么必须做这件事？】
      质检每次重跑都会重新生成批注文件。如果直接覆盖：
          你改批注 → 重跑 → 你的修改被冲掉 → 白改了
      这个坑实测确认过（人工加的一条修改在重跑后消失）。

      所以改成**合并**：新扫出来的问题照常列出来，
      但凡是人工动过的（填了 corrected，或改过 action），原样保留。

    【为什么用 found 文本做索引，而不是 id？】
      因为 id 每次重跑都会重新编号（a001、a002…），靠不住；
      而 found 是原文片段，只要原文没变，跨次运行都是稳定的。

    返回：{found 文本: 那条批注} 的字典。
    """
    if not _HAS_YAML:
        return {}
    p = REVIEW / f"{name}.yaml"
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:                                   # noqa: BLE001
        return {}                      # 文件坏了就当没改过，不中断流程
    out: dict[str, dict] = {}
    for it in data.get("items", []):
        key = (it.get("found") or "").strip()
        if key:
            out[key] = it
    return out


def check_one(name: str) -> tuple[list[dict], dict]:
    """对**一份**文档跑完五条规则。

    返回：(可疑点列表, 统计信息)
    """
    ddir = doc_dir(name)
    asm = load_json(ddir / "03_assemble.json") or {}     # 第③步的结果
    pre = load_json(ddir / "01_preflight.json")          # 第①步的结果
    conv = load_json(ddir / "02_convert.json") or {}     # 第②步的结果

    # 没有组装产物就没法质检 —— 明确返回"跳过"，不要假装成功
    md_path = asm.get("markdown")
    if not md_path or not Path(md_path).exists():
        return [], {"ok": False, "reason": "没有组装产物"}

    # 读正文。公式检查看 .tex，图片检查看 .md（各取所长）
    md_file = Path(md_path)
    md = md_file.read_text(encoding="utf-8", errors="replace")
    tex_path = Path(asm.get("tex", ""))
    tex = tex_path.read_text(encoding="utf-8", errors="replace") if tex_path.exists() else md

    # ---- 依次跑五条规则，把结果汇总 ----
    raw: list[dict] = []
    raw += r1_brace_mismatch(tex)
    raw += r2_suspect_cmd(tex)
    raw += r3_long_inline(tex)
    raw += r4_missing_image(md, md_file.parent)
    raw += r5_empty_page(pre, md)
    raw += r6_page_coverage(pre, conv)      # ★ 唯一直接盯"内容有没有漏"的规则

    # 给每条可疑点加上编号和默认状态，变成标准的"批注条目"
    items = []
    for i, h in enumerate(raw, 1):
        items.append({
            "id": f"{name}_a{i:03d}",       # 编号，形如 年报_a001
            "page": None,                    # 页码（目前没细到页，留空）
            "type": h["type"],               # 类型：formula / image / page
            "issue": h["issue"],             # 问题描述
            "found": h.get("found", ""),     # 原文片段（定位用）
            "hint": h.get("hint", ""),       # 给点提示，帮判断
            "corrected": "",                 # ★ 等你填正确内容
            "action": "pending",             # 状态：pending | accept | fix | reject
        })

    stats = {
        "ok": True,
        "name": name,
        "pages": (pre or {}).get("pages"),
        "kind": (pre or {}).get("kind"),
        "n_images": asm.get("n_images", 0),
        "n_math": asm.get("n_math_inline", 0),
        "n_issue": len(items),               # 待核对的问题数
        "seconds": conv.get("seconds"),
    }
    return items, stats


def run(src=None) -> list[dict]:
    """质检所有已组装的文档。这是给 run.py 调用的入口。

    参数 src —— 本次的输入路径，用来限定只质检这一次要处理的文档
    """
    log = get_logger()
    step_banner(log, "④", "质检（生成批注）")

    # ★ 同第③步：只质检本次输入范围内的文档，别把历史缓存也扫进来
    from steps.common import list_docs
    wanted = {safe_name(p.stem) for p in list_docs(src)} if src else None

    # 找出"第③步已经组装过的"文档
    names = [d.name for d in sorted(DATA.iterdir())
             if d.is_dir() and (d / "03_assemble.json").exists()
             and (wanted is None or d.name in wanted)]
    if not names:
        raise RuntimeError("没有可质检的产物 —— 先跑第 ③ 步")

    log.info("质检 %d 份 · 规则：%s", len(names), RULES)
    t0 = time.time()
    summary = []

    for k, name in enumerate(names, 1):
        items, stats = check_one(name)
        if not stats.get("ok"):
            log.warning("  [%d/%d] %-38s 跳过（%s）", k, len(names), name[:38],
                        stats.get("reason"))
            continue

        # ---- 合并人工修改：新问题照常列，人工动过的原样保留 ----
        manual = load_manual(name)
        n_kept = 0
        for it in items:
            prev = manual.get((it.get("found") or "").strip())
            # 只有"人确实动过"的才保留（填了内容 或 改过状态）
            if prev and (prev.get("corrected") or prev.get("action") not in (None, "", "pending")):
                it["action"] = prev.get("action", it["action"])
                it["corrected"] = prev.get("corrected", "")
                n_kept += 1

        # 已经修好的条目也要留住。
        # 原因：万一以后重跑第②步（比如换了参数），.md 会重新生成，
        # 修正内容又没了 —— 留着这些记录才能再回灌一次。
        new_keys = {(it.get("found") or "").strip() for it in items}
        n_archived = 0
        for key, prev in manual.items():
            if key in new_keys:
                continue                  # 这次还扫得出来，上面已经处理过了
            if prev.get("corrected") or prev.get("action") not in (None, "", "pending"):
                prev = dict(prev)
                prev["action"] = "done"   # 已处理，不再列入待办清单
                prev["issue"] = str(prev.get("issue", "")) + "（已处理，保留备用）"
                items.append(prev)
                n_archived += 1

        if n_kept or n_archived:
            log.info("        （保留人工修改 %d 条 / 已修好存档 %d 条）",
                     n_kept, n_archived)

        # ---- 写成 YAML 批注文件 ----
        ypath = REVIEW / f"{name}.yaml"
        payload = {
            "doc": name,
            "source_pdf": (load_json(doc_dir(name) / "01_preflight.json") or {}).get("file"),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            # 这句提示会写在文件开头，方便你过一阵子忘了怎么改时照着做
            "howto": ("改完请把 action 改成 fix，并把改好的内容填进 corrected，"
                      "然后执行：python run.py --input <pdf目录> --redo assemble"),
            "n_issue": len(items),
            "items": items,
        }
        if _HAS_YAML:
            ypath.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8")
        else:
            # 没装 pyyaml 就退回存 JSON（内容一样，只是格式不同）
            import json
            ypath.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        save_json(doc_dir(name) / "04_review.json", stats)
        summary.append(stats)

        # ✓ 表示没发现问题，⚠ 表示有待核对项
        flag = "✓" if stats["n_issue"] == 0 else "⚠"
        log.info("  [%d/%d] %s %-38s %s页 %d图 %d公式 → %d 处待核",
                 k, len(names), flag, name[:38], stats["pages"],
                 stats["n_images"], stats["n_math"], stats["n_issue"])

    # ---- 收尾汇报 ----
    total_issue = sum(s["n_issue"] for s in summary)
    log.info("")
    log.info("质检完成：%d 份 / 共 %d 处待核 / 耗时 %.1fs",
             len(summary), total_issue, time.time() - t0)
    if total_issue:
        log.info("批注文件在：%s", REVIEW)
        log.info("改完后执行：python run.py --input <pdf目录> --redo assemble")
    return summary


# 直接运行本文件时走这里
if __name__ == "__main__":
    run(None)
