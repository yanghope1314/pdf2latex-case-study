# -*- coding: utf-8 -*-
"""
step6_analyze.py —— 第⑥步：LLM 分析（把 L2 证据变成结论）

【这一步解决什么问题？】用大白话说：
    上一步按规则抽出了证据（L2），但规则只会数"这段话里有没有某个词"，
    它判断不了"材料在这个问题上支持不支持某种说法"。
    这一步把 L2 交给大模型做那一层判断，产出结论和依据单元编号。

【大模型和规则的分工在哪】
    ★ 能交给代码的，绝不交给模型：

      代码（⑤）    页码是查的、命中是数的、
                   "有没有基线/口径/时间窗"是标的、
                   "风险句有没有点名主体"是判的
      模型（⑥）    只做最后一层：读证据 → 下判断 → 说理由 → 指回编号

    这么分是因为模型对长上下文的中段注意力最弱。
    让它读 4 万字原文，它会漏掉第 30 页的一条关键限定；
    让它读 8000 字抽好的证据，它不会。

【★ 模型看不到 L1 全文，只能看到第⑤步给的 L2】
    这是本步骤最要紧的一条设计，三个理由：

      一、量   原文 4 万多字符，证据不到 1 万（实测 27.7%）
      二、准   长上下文里模型漏条件是常态
      三、可核 每条结论带 [P080] 这样的编号，能回原文某个段落核对

    代价必须说清楚：**证据漏了的东西，模型也分析不出来。**
    所以⑤的规则命中率不是细节，它直接决定这一步的能力上限。
    改⑤的规则比调模型参数有用得多。

【两道保险（都是实测踩出来的）】
    ① 输出必须开 response_format=json_object。
       提示词里写死 schema 不够 —— 提示词一长（实测 3000+ token）
       模型就会改写成散文 Markdown，schema 全丢。
       ⚠️ 开了这个参数，prompt 里必须出现小写 "json" 字样，否则接口 400。

    ② 调完之后校验：模型引用的每个单元编号都必须真实存在。
       不校验的话，模型随口编一个 [P999] 也能混进报告 ——
       而整份报告的可信度正是建立在"每个编号都能核对"上。

    ★ 失败不闷头出报告：引文对不上就报错停下、逐条列出。
       一份看着可核、实际核不上的报告，比没有更糟。

【产物】
    analysis/{文档名}_分析.md              最终报告（人读）
    analysis/{文档名}_分析.json            结构化结果 + 无效引用清单
    analysis/{文档名}_模型原始输出.json    模型原话存档，便于复现与答辩
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from steps.common import OUT, get_logger, step_banner            # noqa: E402
from steps.step5_extract import load_setting                       # noqa: E402

REPORT_DIR = ROOT / "analysis"          # 分析结论放这儿
SETTING_FILE = ROOT / "config" / "setting.yaml"


# ----------------------------------------------------------------------
# 一、调用大模型
# ----------------------------------------------------------------------
def call_llm(mcfg: dict, messages: list[dict], log, dry_run: bool = False) -> str:
    """调用 DeepSeek，返回模型输出的文本。

    输入：mcfg = 配置里的「模型」段；messages = 对话消息
    返回：模型输出的文本（str）

    ★ 密钥怎么处理
      只从环境变量 DEEPSEEK_API_KEY 读进内存，
      **不落盘、不打印、不写日志、不进异常信息**。
      请求失败时只报状态码和响应前若干字 ——
      那里面可能带账户与限流信息，不适合原样倒出来。

    ★ 为什么要带重试
      网络抖动、限流、5xx 都不是"重来一次就好"以外的解释。
      但 400/401/403 是请求本身有问题（密钥错、schema 不合法），
      重试没有意义，直接报错把响应正文带回去。
    """
    if dry_run:
        return ""

    import requests

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        # ★ 这份代码要交给别人跑，缺 key 时要把"怎么办"直接打出来，
        #   而不是只丢一句"环境变量未设置"。密钥永远只从环境变量读。
        log.error("第⑥步要调用大模型，但环境变量 DEEPSEEK_API_KEY 没设置。")
        log.error("  Windows 命令提示符:  set DEEPSEEK_API_KEY=sk-你自己的key    然后重跑")
        log.error('  Windows PowerShell :  $env:DEEPSEEK_API_KEY="sk-你自己的key"  然后重跑')
        log.error("  第⑥步可以单独补跑:   python run.py --input <PDF目录> --only analyze")
        log.error("  不想调模型也没关系: 前五步的产物已经落盘 ——")
        log.error("     out/{文档名}/{文档名}.md 与 .tex 是转换结果，03_l2.json 是抽出的证据。")
        log.error("  密钥只放在环境变量里，不要写进任何文件。")
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY（第⑥步需要调用模型）")

    payload = {
        "model": mcfg["模型名"],
        "messages": messages,
        "temperature": float(mcfg.get("温度", 0.2)),
        "max_tokens": int(mcfg.get("最大输出token", 3000)),
    }
    # 强制 JSON 输出模式。实测 DeepSeek 接受这个参数（HTTP 200）。
    # 为什么提示词里已经写死了格式还要开这个：
    # 短任务两种模式都听话，但提示词一长（实测 3000+ token 的证据包）
    # 模型就会改写成散文——返回一篇 Markdown 分析，schema 全丢。
    # 提示词约束是软约束，这里才是硬约束。
    if mcfg.get("JSON模式", True):
        payload["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            r = requests.post(
                mcfg["接口"],
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=180,
            )
            dt = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                u = data.get("usage") or {}
                log.info("  模型返回 %.1fs · prompt %d tok / completion %d tok",
                         dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                return (data["choices"][0]["message"]["content"] or "").strip()
            last = f"HTTP {r.status_code}"
            log.warning("  第 %d 次尝试失败：%s", attempt, last)
            if r.status_code in (400, 401, 403):
                raise RuntimeError(
                    f"模型调用被拒绝（{last}）。响应前 200 字：{r.text[:200]}")
            time.sleep(2 * attempt)
        except RuntimeError:
            raise
        except Exception as exc:                       # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            log.warning("  第 %d 次尝试异常：%s", attempt, last)
            time.sleep(2 * attempt)
    raise RuntimeError(f"连续 3 次调用失败，最后一次：{last}")


# ----------------------------------------------------------------------
# 二、拼提示词
# ----------------------------------------------------------------------
_SYSTEM = """你是一名案例分析员。你的任务是根据给定的**证据**回答若干个核心问题。

铁律：
1. 只依据给出的证据作答。证据里没有的信息，一律写"证据中未见"。
   绝对不要用你对这家公司、这个行业或这 papers 的先验知识来补充。
2. 每条结论后面用方括号标出依据的单元编号，例如 [P080]。
   编号必须来自证据里出现过的，不能编。
3. 如果某个核心问题的证据不足，明确写"证据不足"并说明缺什么，
   不要用常识补齐。
4. 判断要区分"材料说了"和"材料证明了"。材料主张不等于材料证实。

关于"倾向"这一栏（重要，判错这一栏整个分析就没意义）：
  倾向 = 这一格上的表现，是把**这份材料当作分析对象**时的加分项还是减分项。
  · 正面：材料在这一格给出了可核、可信、对分析有用的东西。
  · 负面：材料在这一格有缺陷 —— 只有主张没有证据、避而不谈、
          或者把泛述包装成针对本案例的披露。

最要紧的一条判据：
  **缺口本身是不是缺陷，要看该不该由这个材料来填。**
  · 风险一节全是教科书式泛述，一条也没落到被研究的两家机构头上 —— 这是**减分项**，
    因为案例分析本来就需要知道它们承担了什么风险。
  · 论文没写自己的定价策略 —— 这**不是**减分项，因为那不在它的论述范围内。
  分不清的时候，往"中性"上靠，并在"倾向理由"里说清你为什么这么判。"""

_JSON_SCHEMA = """{
  "总体评价": "两到三句话，说清这份材料作为分析对象的成色",
  "核心问题": [
    {"id": "CP1",
     "结论": "一句话回答这个问题",
     "判定": "充分 | 部分 | 证据不足",
     "倾向": "正面 | 负面 | 中性",
     "倾向理由": "这一格是加分还是减分，一句话，必须指向证据",
     "依据": ["P080", "P012"],
     "理由": "两三句话，必须落在给出的证据上"}
  ],
  "正负汇总": {
     "正面": ["CP2", "CP3"],
     "负面": ["CP4", "CP5"],
     "中性": []
  },
  "核心发现": ["三到五条最有价值的发现，每条都要有单元编号"],
  "存疑之处": ["一到三条你认为证据本身有问题的地方"],
  "方法局限": ["一到三条本次分析的局限（不是你分析对象的局限）"]
}"""


def build_prompt(l2: dict, cfg: dict) -> str:
    """把 L2 渲染成送给模型的文本。

    输入：l2 = 第⑤步产出的 L2 JSON；cfg = 配置
    返回：发给模型的 user 消息文本

    ⚠️ 这段文本里必须出现小写 "json" 四个字。
      开了 response_format=json_object 之后，DeepSeek 会检查这一点，
      只写在 system 消息里不算，会直接 400 拒绝。
      （真踩过：schema 结构本身不含 "json" 二字，接口直接报错。）

    渲染顺序按"先结论素材、后佐证"排：先给量化表和风险句，
    再给各问题的证据句。模型读到最后面时注意力最强，
    所以把最该被看见的东西放在中后段。
    """
    meta = l2["元信息"]
    out = [
        "# 案例材料分析任务",
        "",
        f"案例：{meta['案例']}",
        f"副标题：{meta['副标题']}",
        f"行业：{meta['所属行业']}",
        f"文档分析单元：{meta['分析单元']}（标题 {meta['标题数']} 个 / 段落 {meta['段落数']} 个）",
        "",
        "以下是从材料中用代码规则抽取出的证据。"
        "你没有看到材料全文，只能依据这些证据回答。",
        "",
        "## 一、量化陈述（每条已标注它有没有交代基线 / 口径 / 时间窗）",
        "",
    ]
    qs = l2.get("量化陈述") or []
    if qs:
        out.append("编号 | 数字 | 基线 | 口径 | 时间窗 | 原句")
        for q in qs:
            marks = ("✓" if q["有基线"] else "✗",
                     "✓" if q["有口径"] else "✗",
                     "✓" if q["有时间窗"] else "✗")
            sent = q["句子"].replace("|", "/")
            out.append(f'{q["单元"]} | {"、".join(q["数字"][:4])} | '
                       f'{marks[0]} {marks[1]} {marks[2]} | {sent[:150]}')
    else:
        out.append("（无）")

    out += ["", "## 二、风险类语句（标注是否点名了案例主体）", ""]
    rs = l2.get("风险句") or []
    if rs:
        for r in rs:
            tag = "点名主体" if r["点名案例主体"] else "泛述（未点名本案例主体）"
            out.append(f'[{r["单元"]}] （{tag}）{r["句子"][:150]}')
    else:
        out.append("（无）")

    out += ["", "## 三、各核心问题的证据", ""]
    for p in l2["核心问题"]:
        out.append(f'### {p["id"]} {p["标题"]}')
        out.append(f'问题：{p["问题"]}')
        if p["判据"]:
            out.append(f'判断标准：{p["判据"]}')
        out.append(f'（代码规则在全文中命中 {p["命中段落数"]} 段 / {p["命中次数"]} 次）')
        if not p["证据"]:
            out.append(f'**证据不足。** {p["缺口问法"]}')
        else:
            for e in p["证据"]:
                sent = e["原句"].replace("\n", " ")
                out.append(f'[{e["单元"]}]（{e["小节"][:40]}）{sent}')
        out.append("")

    # 注意这句里的 "json" 是小写且必须出现在**用户消息**里 ——
    # DeepSeek 开启 response_format=json_object 时会检查这一点，
    # 只写在 system 消息里不算，会直接 400 拒绝。
    # （踩过：schema 结构本身不含 "json" 二字，接口报
    #  "Prompt must contain the word 'json' in some form"。）
    out += ["", "## 四、输出要求", "",
            "请只输出一个 **json** 对象（严格符合下面的字段结构），",
            "不要任何标题、列表、解释文字或 ``` 包裹。字段如下：", "",
            _JSON_SCHEMA]
    return "\n".join(out)


# ----------------------------------------------------------------------
# 三、解析模型输出
# ----------------------------------------------------------------------
def parse_json(text: str, log) -> dict:
    """从模型输出里取出 JSON。

    输入：text = 模型返回的原始文本；log = 日志器
    返回：解析出来的 dict

    ★ 为什么还要三层兜底
      就算开了 response_format=json_object、提示词也写死了 schema，
      模型仍可能裹上 ```json 代码块、或者前后加两句说明。
      三层依次是：直接解析 → 抓第一个 {...} 块 → 抓 ```json 块。

    ★ 为什么解析失败要把原文存盘再报错
      丢了原文等于丢了一次排障线索。
      出问题时人得看得见模型到底说了什么，
      而不是只看到一句"解析失败"。
    """
    t = text.strip()
    for cand in (t,
                 re.search(r"\{.*\}", t, re.S).group(0) if re.search(r"\{.*\}", t, re.S) else "",
                 re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S).group(1)
                 if re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S) else ""):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            log.debug("这一版不是合法 JSON：%s", exc)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bad = REPORT_DIR / "模型原始输出.txt"
    bad.write_text(text, encoding="utf-8")
    raise RuntimeError(
        f"模型输出无法解析为 JSON，原始输出已存到 {bad.name}，请人工看一眼。")


# ----------------------------------------------------------------------
# 四、渲染成 Markdown
# ----------------------------------------------------------------------
def render_md(res: dict, l2: dict) -> str:
    meta = l2["元信息"]
    crit = {p["id"]: p for p in l2["核心问题"]}

    # 正负倾向用符号标出来，一眼能看出哪几格是减分项
    MARK = {"正面": "＋ 正面", "负面": "－ 负面", "中性": "○ 中性"}

    L = [f'# {meta["案例"]} · 大数据案例分析', "",
         f'> {meta["副标题"]}', "",
         f'行业：{meta["所属行业"]}　|　分析单元：{meta["分析单元"]}'
         f'（段落 {meta["段落数"]} 个）',
         f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}　|　'
         f'分析引擎：LLM（证据由代码规则抽取）', "",
         "---", "", "## 总体评价", "", res.get("总体评价", "（无）"), ""]

    # ---- 正负汇总：先给结论，再展开 ----
    zf = res.get("正负汇总") or {}
    if zf:
        L += ["## 正负汇总", "",
              "| 倾向 | 核心问题 |", "|---|---|"]
        for k in ("正面", "负面", "中性"):
            ids = zf.get(k) or []
            if not ids:
                continue
            names = "、".join(
                f'{i} {crit.get(i, {}).get("标题", "")}'.strip() for i in ids)
            L.append(f'| {MARK[k]} | {names} |')
        L.append("")

    L += ["## 逐个核心问题", ""]
    L += ["| 问题 | 证据判定 | 倾向 | 一句话结论 |", "|---|---|---|---|"]
    for item in res.get("核心问题", []):
        pid = item.get("id", "")
        name = crit.get(pid, {}).get("标题", pid)
        L.append(f'| {pid} {name} | {item.get("判定", "")} '
                 f'| {item.get("倾向", "")} | {item.get("结论", "")} |')
    L.append("")
    for item in res.get("核心问题", []):
        pid = item.get("id", "")
        name = crit.get(pid, {}).get("标题", pid)
        q = crit.get(pid, {}).get("问题", "")
        L += [f'### {pid} {name}', "",
              f'**问题**：{q}', "",
              f'**结论**：{item.get("结论", "")}', "",
              f'**证据判定**：{item.get("判定", "")}　'
              f'**倾向**：{item.get("倾向", "")}', ""]
        if item.get("倾向理由"):
            L += [f'**倾向理由**：{item["倾向理由"]}', ""]
        if item.get("理由"):
            L += ["**理由**：", "", item["理由"], ""]
        ev = item.get("依据") or []
        if ev:
            L += [f'**依据单元**：{"、".join(ev)}', ""]

    for key, title in (("核心发现", "核心发现"),
                       ("存疑之处", "存疑之处"),
                       ("方法局限", "本次分析的局限")):
        items = res.get(key) or []
        if not items:
            continue
        L += [f"## {title}", ""]
        L += [f"{i}. {t}" for i, t in enumerate(items, 1)]
        L.append("")

    L += ["---", "",
          "## 附：证据来源说明", "",
          "本报告的所有判断都建立在代码抽取的证据之上，"
          "**模型没有看到原文全文**。",
          "每个方括号编号（如 `[P080]`）对应原文的一个段落单元，"
          "可回转换产物 `out/{文档名}/{文档名}.md` 按顺序核对。", "",
          "两栏判定的区别：",
          "**证据判定**说材料给的东西够不够（充分／部分／证据不足）；"
          "**倾向**说这一格是加分还是减分。", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------
# 五、主流程
# ----------------------------------------------------------------------
def run(src: Path, dry_run: bool = False) -> int:
    """第⑥步的入口。返回 0 = 通过，抛异常 = 失败。

    输入：src = 项目根（保留这个参数是为了跟其余五步同一个调用形状，
                本步骤实际不吃它）；dry_run = True 时只存提示词不调模型
    返回：0

    做的事，按顺序：
        1. 读配置，按「文档关键字」去 out/ 里找目标文档
           （不吃 --input 的目录 —— 本步骤只处理配置指定的那一篇）
        2. 读第⑤步的 03_l2.json，没有就报错并提示先跑⑤
        3. 拼提示词 → 调模型 → 解析（解析失败会带纠正消息重试）
        4. ★ 校验模型引用的每个单元编号是否真实存在
        5. 落盘：报告 md、结构化 json、模型原话存档

    ★ 第 4 步为什么不能省
      模型可能引用一个根本不存在的单元编号。
      不校验的话报告看起来"有依据"，实际那条依据点不开 ——
      比没有依据更糟，因为它会让读者停止怀疑。
      所以有一个对不上就报错停下、逐条列出，不闷头出报告。

    --preview 的用法（不调模型、不花钱，调试提示词时用）：
        python run.py --input <任意目录> --only analyze --preview
      提示词落在 logs/prompt_preview.txt。
    """
    log = get_logger()
    step_banner(log, "⑥", "LLM 分析（L2 证据 → 结论）")

    cfg = load_setting(SETTING_FILE)
    mcfg = cfg["模型"]
    case = cfg["案例"]

    hits = [d for d in sorted(OUT.iterdir())
            if d.is_dir() and case["文档关键字"] in d.name]
    if not hits:
        raise RuntimeError(f"在 {OUT} 里找不到含「{case['文档关键字']}」的转换结果")
    doc_dir = hits[0]
    l2_path = doc_dir / "03_l2.json"
    if not l2_path.exists():
        raise RuntimeError(
            f"{l2_path.name} 不存在 —— 先跑第⑤步。\n"
            "  python run.py --only extract")
    l2 = json.loads(l2_path.read_text(encoding="utf-8"))

    prompt = build_prompt(l2, cfg)
    l1_chars = l2["元信息"]["L1字符数"]
    log.info("案例：%s", case["名称"])
    log.info("L2 证据：%d 字符（原文 %d 字符，占 %.1f%%）",
             len(prompt), l1_chars, len(prompt) / l1_chars * 100 if l1_chars else 0)
    log.info("模型：%s（温度 %s，最多 %d tok）",
             mcfg["模型名"], mcfg.get("温度"), mcfg.get("最大输出token"))

    if dry_run:
        out = ROOT / "logs" / "prompt_preview.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(prompt, encoding="utf-8")
        log.info("[dry-run] 不实际调用模型。提示词已存到 %s", out)
        log.info("          提示词 %d 字符", len(prompt))
        return 0

    log.info("调用模型中……")
    # 调用 + 解析 一起重试。
    # 只重试"调用失败"不够 —— 实测有次 HTTP 200、内容也拿到了，
    # 但模型把 schema 改写成了 Markdown，解析直接失败。
    # 那种情况要带一条纠正消息再问一次，比直接报错有用得多。
    t0 = time.time()
    res = None
    msgs = [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt}]
    for attempt in range(1, 4):
        raw = call_llm(mcfg, msgs, log, dry_run)
        try:
            res = parse_json(raw, log)
            break
        except RuntimeError as exc:
            if attempt >= 3:
                raise
            log.warning("  第 %d 次的输出解析不了（%s），带纠正消息重试",
                        attempt, str(exc)[:60])
            msgs.append({"role": "assistant", "content": raw[:1500]})
            msgs.append({"role": "user", "content":
                         "上面的回答不是合法 JSON。请严格按下面这份 schema 重新输出，"
                         "只输出 JSON 本身，不要任何标题、解释或代码块标记：\n"
                         + _JSON_SCHEMA})
    log.info("耗时 %.1fs", time.time() - t0)

    # 原始输出先落盘再往下走。
    # 解析之后的校验/渲染如果自己有 bug，原始输出已经保住了，
    # 不必为调一个键名错误再烧一次 API 调用。
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r'[\\/:*?"<>|]', "_", doc_dir.name)
    (REPORT_DIR / f"{stamp}_模型原始输出.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 校验：模型引用的单元编号必须真实存在 ----
    # 不校验的话，模型随口编一个 [P999] 也能混进报告，
    # 而整份报告的可信度正是建立在"每个编号都能核对"上。
    # 证据项的键是「单元」（不是 id）—— 单元编号 H001/P001 才是坐标。
    valid = {u["单元"] for p in l2["核心问题"] for u in p["证据"]}
    valid |= {q["单元"] for q in l2.get("量化陈述") or []}
    valid |= {r["单元"] for r in l2.get("风险句") or []}
    valid |= {p["id"] for p in l2["核心问题"]}
    cited, bad = set(), []
    for item in res.get("核心问题", []):
        for ref in item.get("依据") or []:
            cited.add(ref)
            if ref not in valid:
                bad.append(f'{item.get("id")} 引用了不存在的单元 {ref}')
    for t in (res.get("核心发现") or []) + (res.get("存疑之处") or []):
        for ref in re.findall(r"\[([A-Z]\d{3})\]", str(t)):
            cited.add(ref)
            if ref not in valid:
                bad.append(f"发现里引用了不存在的单元 {ref}")

    # ---- 落盘 ----
    (REPORT_DIR / f"{stamp}_分析.json").write_text(
        json.dumps({"分析": res, "元信息": l2["元信息"],
                    "引用单元数": len(cited),
                    "无效引用": bad}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (REPORT_DIR / f"{stamp}_分析.md").write_text(render_md(res, l2), encoding="utf-8")

    log.info("")
    log.info("=" * 68)
    log.info("  模型引用单元 %d 个，其中无效 %d 个", len(cited), len(bad))
    for b in bad:
        log.error("  ✗ %s", b)
    log.info("  产物：analysis/%s_分析.md · %s_分析.json", stamp, stamp)
    log.info("=" * 68)
    if bad:
        raise RuntimeError(
            f"模型引用了 {len(bad)} 个不存在的单元编号 —— "
            "报告里的依据无法核对，先看 analysis/ 下的 json。")
    return 0


if __name__ == "__main__":
    sys.exit(run(ROOT, dry_run="--dry-run" in sys.argv))