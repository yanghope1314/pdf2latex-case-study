# -*- coding: utf-8 -*-
"""
step5_extract.py —— 第⑤步：L2 抽取（把 L1 原文变成能直接送模型的证据）

【这一步解决什么问题？】用大白话说：
    上一步已经拿到了 Markdown 全文（L1），一份 20 页论文约 4 万字符。
    但这 4 万字直接喂给大模型有三个问题：
      ① 里面大半是噪音 —— 版权行、页码、参考文献、表格残骸
      ② 没法核对 —— 模型说"P080 里有这个数"，你怎么确认它没编？
      ③ 太贵 —— 长上下文中段模型注意力最弱，会漏掉关键限定
    这一步负责按**预先定好的规则**从原文里抽出结构化证据（L2），
    抽出多少、抽哪几句，全由 config/setting.yaml 决定，
    运行时不需要人再标注任何东西。

【什么是 L1 和 L2】
    L1 = out/{文档名}/{文档名}.md，全文照搬，一段不改。
    L2 = 本步骤的产物。按"分析单元"把 L1 切块，再按核心问题逐条匹配，
         每条证据带单元编号（P080、H012）和所属小节名。

【L2 真正的价值不是"省"，是"逼出没有"】
    ★ 省 token 是附带的好处，不是目的。真正要紧的是第三条：

      规则匹配不上，就是匹配不上。**"CP5 代价与风险：命中 0 段"
      本身就是一条硬结论**，它会被原样交给模型，
      模型据此判"这一格证据不足"，而不是去全文里找点什么凑数。

      实测踩过的坑：摩根士丹利那份论文的风险一节写得非常完整
      （黑箱、公平性、责任归属、错误成本一条不缺），
      但 8 条风险句**没有一条点名那两家银行**。
      规则给的答案是"泛述"，模型据此判了"证据不足"——
      这个结论如果靠人通读全文，一定会被"写得挺全"带过去。

【为什么不能交给模型做这一层】
    规则匹配必须**可复现**：同样的输入跑两次，输出必须一模一样。
    模型做不到 —— 调一次温度不一样、措辞就不一样。
    这一层一旦不可复现，后面所有结论都不可追责。

【产物】
    out/{文档名}/03_l2.json    L2 结构化证据（这一步的唯一产物）
      · 元信息        原文字符数、单元数、案例元信息
      · 核心问题[]    每个问题抽到了哪几句（带编号+小节名）
      · 量化陈述[]    每个数字，并标注它有没有 基线/口径/时间窗
      · 风险句[]      每条风险句，并标注有没有点名案例主体
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from steps.common import (                                          # noqa: E402
    DATA, OUT, get_logger, load_json, save_json, step_banner,
)

# ---- 本步骤专用的路径 ----
SETTING_FILE = ROOT / "config" / "setting.yaml"


# ----------------------------------------------------------------------
# 一、配置
# ----------------------------------------------------------------------
def load_setting(path: Path = SETTING_FILE) -> dict:
    """读 config/setting.yaml。

    规则、核心问题、分析单元全部来自那里。这个文件是唯一的配置来源，
    本模块里不许再出现第二套关键词 —— 否则改配置不改代码，
    或者改代码不改配置，两边对不上的时候没人知道该信哪个。
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError("这一步需要 PyYAML。修复：pip install PyYAML") from None
    if not path.exists():
        raise RuntimeError(f"找不到配置文件：{path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("案例", "分析单元", "核心问题", "模型"):
        if key not in cfg:
            raise RuntimeError(f"配置缺少必需段：「{key}」")
    return cfg


# ----------------------------------------------------------------------
# 二、切分：L1 → 分析单元
# ----------------------------------------------------------------------
# 单元编号规则：标题记 H001、H002…，段落记 P001、P002…
# 编号是稳定坐标，后面所有证据都靠它回指原文。

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")


def split_units(md_text: str, unit_cfg: dict) -> tuple[list[dict], str]:
    """把 L1 文本切成分析单元。

    输入：md_text = L1 全文；unit_cfg = 配置里的「分析单元」段
    返回：单元列表（见下），以及最后一个小节名

    一个单元长这样：
        id    稳定编号，H001（标题）/ P001（段落）…
        kind  heading / para
        text  正文
        sec   所属小节（继承最近一次见到的标题）

    ★ 为什么编号要这么设计
      编号是后面每一条结论的坐标 —— 报告里写的 [P080]
      就是靠它回到原文。编号必须**只由内容和顺序决定**，
      不能依赖时间戳或随机数，否则同一个输入两次跑出来的报告对不上。

    ★ 为什么要带 sec（小节名）
      光有编号不够 —— "命中 3 段"这三个位置分别在讲什么，
      光看编号连不起来。小节名是把零散证据串回结构的唯一线索。

    ⚠️ 空行必须当段落边界
      漏了这一句的后果很隐蔽：整节正文会被并成一个单元，
      实测段落数从 125 掉到 29、平均长度 1400 字符 ——
      表面上"切了段"，实际等于没切，分析精度全丢。
    """
    min_chars = int(unit_cfg.get("最短字符数", 0))
    keep_heading = bool(unit_cfg.get("标题单列", True))

    units: list[dict] = []
    cur_sec = ""
    h_i = p_i = 0
    buf: list[str] = []

    def flush() -> None:
        """把缓冲区里的行拼成一个段落单元。"""
        nonlocal p_i, buf
        text = " ".join(x.strip() for x in buf).strip()
        buf = []
        if len(text) < min_chars:          # 太短的当噪声丢掉（页码、版权行…）
            return
        p_i += 1
        units.append({"id": f"P{p_i:03d}", "kind": "para",
                      "text": text, "sec": cur_sec})

    for raw in md_text.splitlines():
        m = _HEADING.match(raw)
        if m:
            flush()
            title = m.group(2).strip()
            cur_sec = title
            if keep_heading:
                h_i += 1
                units.append({"id": f"H{h_i:03d}", "kind": "heading",
                              "text": title, "sec": title})
        elif not raw.strip():
            # 空行 = 段落边界，必须断（漏了会整节并成一段，详见本函数 docstring）
            flush()
        else:
            buf.append(raw)
    flush()
    return units, cur_sec


# ----------------------------------------------------------------------
# 三、规则匹配
# ----------------------------------------------------------------------
def compile_rules(problem: dict) -> dict:
    """把一个核心问题的规则编译成正则。

    关键词转成"不区分大小写的字面匹配"，
    正则原样编译并加上 IGNORECASE。

    为什么不用 `re.escape` 处理关键词再统一走正则：
    两种写法各有用途 —— 关键词表里会有 `secondary data` 这种含空格的短语，
    转义后当正则也能匹配，但可读性差，且遇到用户误写正则元字符时行为诡异。
    分开处理更可控。
    """
    rules = problem.get("规则") or {}
    kws = [k.lower() for k in (rules.get("关键词") or [])]
    pats = [re.compile(p, re.IGNORECASE) for p in (rules.get("正则") or [])]
    return {"关键词": kws, "正则": pats}


def match_sentences(unit: dict, rules: dict, max_sents: int = 3
                    ) -> tuple[list[tuple[list[str], str]], int]:
    """在一个段落里逐句跑规则，返回 (命中句列表, 命中总次数)。

    输入：unit = 一个段落单元；rules = compile_rules 编好的规则
          max_sents = 一段最多留几句
    返回：命中句 = [(命中了哪些词, 句子原文)]，以及本段命中总次数

    ★ 为什么按句返回，不按段返回
      段落动辄上千字符，把整段塞进 L2 会让压缩比变成负数 ——
      实测第一版就是这么做的：L2 体积是原文的 **173%**，
      一个比原文还大的"中间层"就没有存在的意义了。
      按句返回后，一段里与核心问题无关的部分（综述、举例、过渡）
      不会被带进去，证据也更接近"这一点上材料说了什么"。

    一段最多留 max_sents 句：同一段里连着三句都在讲这个问题，
    再多就是啰嗦，不是新信息。
    """
    hits: list[tuple[list[str], str]] = []
    total = 0
    for sent in _SENT_SPLIT.split(unit["text"]):
        if len(sent.strip()) < 15:
            continue
        low = sent.lower()
        kw_hits, n = [], 0
        for kw in rules["关键词"]:
            c = low.count(kw)
            if c:
                kw_hits.append(kw)
                n += c
        for pat in rules["正则"]:
            c = len(pat.findall(sent))
            if c:
                kw_hits.append(pat.pattern[:30])
                n += c
        if n:
            hits.append((kw_hits, sent.strip()))
            total += n
    hits.sort(key=lambda t: -len(t[0]))       # 命中种类多的排前面
    return hits[:max_sents], total


def unit_hit_count(unit: dict, rules: dict) -> int:
    """只算命中次数，不返回句子。用于统计"多少段沾了这个边"。"""
    total = 0
    low = unit["text"].lower()
    for kw in rules["关键词"]:
        total += low.count(kw)
    for pat in rules["正则"]:
        total += len(pat.findall(unit["text"]))
    return total


# ----------------------------------------------------------------------
# 四、量化陈述抽取 —— L2 里最有价值的一块
# ----------------------------------------------------------------------
# 三个"有没有"的判据。目的是把"披着精确外衣的套话"和"真的量过的数"分开。

_NUM = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|"
    r"minutes?|hours?|days?|months?|years?|reports?|meetings?|clients?|"
    r"customers?|studies|points?|times?)?",
    re.IGNORECASE,
)

# 有基线：说明"和什么比"。
# 注意 `[^.]{0,40}` 这一段不能省：
# 实际语料里 "reduces the average meeting duration **by** about 30 minutes"，
# reduce 和 by 中间隔了五个词，写成 `reduc\w+\s+by` 会漏掉全篇最关键的那个数字。
_BASELINE = re.compile(
    r"(reduc\w+[^.]{0,40}\s+by|increas\w+[^.]{0,40}\s+by|"
    r"from\s+.{1,30}\s+to\s+|compared\s+(?:to|with)|versus|baseline|"
    r"grew\s+from|doubled|shorten\w+|saved|improve\w+[^.]{0,30}\s+by|"
    r"up\s+from|down\s+from)", re.IGNORECASE)

# 有口径：说明"怎么统计的"
_MEASURE = re.compile(
    r"(average|mean|median|approximately|around|about|roughly|approx\.?|"
    r"annual|annually|monthly|daily|weekly|per\s|share|rate|percentage|"
    r"estimated|estimat\w+|on\s+average)", re.IGNORECASE)

# 有时间窗：说明"测量区间有多长"。
# ⛔ 刻意**不收** minutes/hours/days —— 它们在这类语料里几乎总是**量级**
# 而不是区间："缩短了 30 分钟"里的 30 minutes 是效果大小，
# 不是"在多长时间内测出来的"。把它当时间窗会把最典型的孤证判成达标。
_WINDOW = re.compile(
    r"(19|20)\d{2}|\b(annual|annually|monthly|daily|weekly|quarterly)\b|"
    r"\byears?\b|\b(over\s+the\s+past|last)\s+\d+|\bsince\s+(19|20)\d{2}|"
    r"\bbetween\s+(19|20)\d{2}", re.IGNORECASE)

# 这些句子里的数字不是业绩：检索日期、DOI、表格行、
# 图表交叉引用（"Table 1 illustrates…"）、模型版本号（GPT-4 的 4）、
# 以及转换残留的 HTML 标签和参考文献编号（"(2025, 12 11)"、"G21, M15, O33"）。
_NOISE = re.compile(
    r"(retrieved\s+\d|https?://|doi\.?[:\s]|arxiv|\[\d+\]|\|\s*-+|^\s*\||"
    # 转换产物里的残留行
    r"!\[[^\]]*\]\(|\[\[|<[a-z/][^>]{0,80}>|"
    r"\bGPT-?\d|jel\s+classification|"
    # 表图：只要句子里出现 Table N / Figure N 就整句不算。
    # 不去区分它是题注、引出语还是正文指代 ——
    # 在案例分析里，"Table 3 shows…" 这句话里的数字是表号不是论断，
    # 逐个枚举写法（illustrates/below/provides/in…）永远补不全，
    # 一刀切反而更稳也更可解释。
    r"\b(table|figure|fig|chart|exhibit)s?\s*\d+\b|"
    # 页码残留：正文里被 PDF 页码插进来的 "719) artificial intelligence…"
    r"^\s*\d{1,4}\s*\)|"
    # 引文检索日期。两种写法都要拦：
    #   "(2025, 12 11)."  ← 句子切分后剩下的裸片段，最难认
    #   "Retrieved 12 11, 2025"
    r"\((19|20)\d{2}\s*,\s*\d{1,2}\s*[,)]|"
    r"\b(19|20)\d{2}\s*,\s*\d{1,2}\s*\d{1,2}\b)",
    re.IGNORECASE)

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")

# 参考文献 / 附录小节：里面的数字全是卷期页码，不是论断。
# 靠小节名判断比逐句认引文格式可靠 —— 引文格式五花八门，小节名是稳定的。
_REF_SECTION = re.compile(
    r"^\s*(references?|bibliograph\w*|works\s+cited|参考文献|附录|appendi\w*)",
    re.IGNORECASE)


def extract_quantities(units: list[dict]) -> list[dict]:
    """把全文的量化陈述逐条抽出来，并标出它有没有基线/口径/时间窗。

    这是 L2 相对 L1 最有增量的地方：
    L1 里有这些数字，但它们是散落在正文里的普通字符；
    L2 把它们提出来、排好队、并且给每个数字打上"这句话
    有没有交代它是量出来的"的标记。

    判据是逐句做的，不是逐段 —— 一个段里前句有基线、
    后句没有，不能让后句沾前句的光。
    """
    out = []
    for u in units:
        if u["kind"] != "para":
            continue
        if _REF_SECTION.match(u["sec"] or ""):
            continue                      # 参考文献小节，整段不算
        for sent in _SENT_SPLIT.split(u["text"]):
            if _NOISE.search(sent):
                continue                      # 检索日期 / DOI / 表格行
            # 兜底：又短又几乎全是数字标点的，多半是切碎后剩下的引文残片。
            # 纯正则列不完（日期格式五花八门），用"字母占比"一刀切更稳。
            letters = sum(c.isalpha() for c in sent)
            if len(sent) < 45 and letters / max(len(sent), 1) < 0.5:
                continue
            # 去掉数字尾部的标点。"2030," 若不清掉尾巴，
            # 就躲开了下面那道"纯年份"过滤，被当成业绩数字留下。
            nums = [m.group(0).strip(" .,;:()") for m in _NUM.finditer(sent)]
            # 纯四位数是年份（1956、2024），是时间标记不是业绩，
            # 一律不算量化陈述。"1956 年的达特茅斯会议"该丢。
            nums = [n for n in nums if n and not re.fullmatch(r"\d{4}", n)]
            if not nums:
                continue
            out.append({
                "单元": u["id"],
                "小节": u["sec"],
                "数字": nums,
                "句子": sent.strip()[:300],
                "有基线": bool(_BASELINE.search(sent)),
                "有口径": bool(_MEASURE.search(sent)),
                "有时间窗": bool(_WINDOW.search(sent)),
            })
    return out


# ----------------------------------------------------------------------
# 五、风险句的"是否点名案例主体"
# ----------------------------------------------------------------------
# 支撑一个具体的判断：风险综述写得很全，
# 但一条也没落到被研究的那两个机构头上 —— 前者是"泛述"，后者才是"披露"。


def mark_risk_context(units: list[dict], subjects: list[str]) -> list[dict]:
    """给每条风险句标上：附近有没有出现案例主体名。

    主体名从配置里给（这里从文件名与正文首段推断的兜底词表也可，
    但写死主体名会让程序换个案例就失效，所以做成可配置）。
    """
    out = []
    pat = re.compile("|".join(re.escape(s) for s in subjects), re.IGNORECASE) \
        if subjects else None
    for u in units:
        if u["kind"] != "para":
            continue
        for sent in _SENT_SPLIT.split(u["text"]):
            if not re.search(r"\b(risk|risks|threat|concern)\w*", sent,
                             re.IGNORECASE):
                continue
            out.append({
                "单元": u["id"],
                "小节": u["sec"],
                "句子": sent.strip()[:300],
                "点名案例主体": bool(pat.search(sent)) if pat else None,
            })
    return out


# ----------------------------------------------------------------------
# 六、主流程
# ----------------------------------------------------------------------
def run(src: Path, cfg: dict | None = None) -> int:
    log = get_logger()
    step_banner(log, "⑤", "L2 抽取（代码规则：切分 → 匹配 → 抽量化陈述）")

    cfg = cfg or load_setting()
    case = cfg["案例"]
    log.info("案例：%s（按文件名关键字「%s」匹配）", case["名称"], case["文档关键字"])

    # ---- 找到待分析的文档 ----
    # 用配置里的关键字去 out/ 里找，而不是拿 --input 的路径当唯一目标：
    # 一次可能转换好几篇，但本次分析只针对配置指定的那一篇。
    hits = [d for d in sorted(OUT.iterdir())
            if d.is_dir() and case["文档关键字"] in d.name]
    if not hits:
        raise RuntimeError(
            f"在 {OUT} 里找不到含「{case['文档关键字']}」的转换结果。\n"
            f"  先跑前四步把 PDF 转成 Markdown，再跑这一步。")
    if len(hits) > 1:
        log.warning("匹配到 %d 个目录，取第一个：%s", len(hits), hits[0].name)
    doc_dir = hits[0]

    md = next(iter(doc_dir.glob("*.md")), None)
    if md is None:
        raise RuntimeError(f"{doc_dir.name} 里没有 .md —— ③ 组装那一步没跑？")

    md_text = md.read_text(encoding="utf-8", errors="replace")
    log.info("L1 原文：%s（%d 字符）", md.name, len(md_text))

    # ---- 切分 ----
    units, _ = split_units(md_text, cfg["分析单元"])
    n_head = sum(1 for u in units if u["kind"] == "heading")
    n_para = len(units) - n_head
    unit_name = cfg["分析单元"]["名称"]
    log.info("分析单元=%s → 标题 %d 个 + 段落 %d 个 = %d 个单元",
             unit_name, n_head, n_para, len(units))

    # ---- 逐核心问题匹配 ----
    mcfg = cfg["模型"]
    cap = int(mcfg.get("每问题最多证据句", 10))
    clip = int(mcfg.get("每句最多字符", 300))
    problems_out = []
    for p in cfg["核心问题"]:
        rules = compile_rules(p)
        scored = []          # (句子命中分, 段落, 命中词, 句子)
        n_units = 0          # 沾过边的段落数
        n_hits = 0           # 命中总次数
        for u in units:
            if u["kind"] != "para":
                continue
            c = unit_hit_count(u, rules)
            if c:
                n_units += 1
                n_hits += c
                sents, _ = match_sentences(u, rules)
                for kw, sent in sents:
                    # 句子得分 = 命中种类数（种类比频次更能说明"讲到了这一块"）
                    scored.append((len(kw), u, kw, sent))

        # 分数高的排前面；同分按单元编号，保证同样输入两次跑出同样结果
        scored.sort(key=lambda t: (-t[0], t[1]["id"]))

        ev, seen_sent = [], set()
        for _, u, kw, sent in scored:
            key = sent[:80]
            if key in seen_sent:       # 同一句话被多条规则命中时只留一次
                continue
            seen_sent.add(key)
            ev.append({
                "单元": u["id"],
                "小节": u["sec"],
                "命中词": kw[:5],
                "原句": sent[:clip],
            })
            if len(ev) >= cap:
                break

        problems_out.append({
            "id": p["id"], "标题": p["标题"], "问题": p["问题"],
            "判据": (p.get("判据") or "").strip(),
            "缺口问法": (p.get("缺口问法") or "").strip(),
            "命中段落数": n_units,
            "命中次数": n_hits,
            "证据": ev,
        })
        log.info("  %s %-8s 命中 %3d 段 / %4d 次 → 取 %d 句作证据",
                 p["id"], p["标题"], n_units, n_hits, len(ev))

    # ---- 量化陈述与风险句 ----
    # 全都抽出来后要排序再截断，不能按文档顺序取前 N 条 ——
    # 按顺序取会优先留下前半部分，而最有信息量的往往是结论段的数字。
    quantities = extract_quantities(units)
    quantities.sort(key=lambda q: (-sum((q["有基线"], q["有口径"], q["有时间窗"])),
                                   q["单元"]))
    quantities = quantities[:int(mcfg.get("量化陈述最多条", 24))]

    subjects = case.get("主体") or []
    risks = mark_risk_context(units, subjects) if subjects else []
    risks = risks[:int(mcfg.get("风险句最多条", 10))]

    n_good = sum(1 for q in quantities
                 if sum((q["有基线"], q["有口径"], q["有时间窗"])) >= 2)
    log.info("量化陈述：%d 条（其中 %d 条同时具备 基线/口径/时间窗 中的两项以上）",
             len(quantities), n_good)
    if risks:
        named = sum(1 for r in risks if r["点名案例主体"])
        log.info("风险句：%d 句（其中 %d 句点名了案例主体，其余为泛述）",
                 len(risks), named)

    # ---- 出 L2 ----
    payload = {
        "元信息": {
            "案例": case["名称"],
            "文档": doc_dir.name,
            "副标题": (case.get("副标题") or "").strip(),
            "所属行业": case.get("所属行业", ""),
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "分析单元": unit_name,
            "标题数": n_head, "段落数": n_para,
            "L1字符数": len(md_text),
        },
        "核心问题": problems_out,
        "量化陈述": quantities,
        "风险句": risks,
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    save_json(doc_dir / "03_l2.json", payload)

    # ---- 抽取率：这一步值不值，就看这个数 ----
    # 口径要说清楚：**只算从原文里抽出来的文本**，
    # 不含配置里的问题描述、判据、缺口问法 ——
    # 那些不是文档内容，算进来会把压缩比做低或做高，得不出可比���数。
    # 真正送进模型的体积由第⑥步实测报告（那才是最终口径）。
    extracted = sum(len(e["原句"]) for p in problems_out for e in p["证据"])
    extracted += sum(len(q["句子"]) for q in quantities)
    extracted += sum(len(r["句子"]) for r in risks)
    log.info("")
    log.info("=" * 68)
    log.info("  L1 原文            %6d 字符", len(md_text))
    log.info("  抽取出的文本量     %6d 字符  （%.1f%%）",
             extracted, extracted / len(md_text) * 100 if md_text else 0)
    log.info("  L2 文件（含配置文本）%5d 字符  —— 是中间记录，不是送模型的量",
             len(body))
    log.info("  产物：%s", doc_dir / "03_l2.json")
    log.info("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT))