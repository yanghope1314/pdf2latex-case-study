# -*- coding: utf-8 -*-
"""
step3_assemble.py —— 第③步：组装（把识别结果整理成能直接交的文件）

【这一步解决什么问题？】用大白话说：
    上一步 MinerU 已经认出了内容，但它给的 Markdown 有两个毛病：
      ① 图片路径指向它自己的临时文件夹，文件一挪走图就全断了
      ② 只有 Markdown，没有我们要的 .tex（LaTeX）文件
    这一步就负责收拾这两件事，产出三个东西：
      · 图片复制到 out/{文档名}/images/，并把正文里的引用路径改对
        （★ 撞名时改名而不是覆盖 —— 覆盖会让一张图悄悄顶掉另一张）
      · 一份干净的 .md  —— 给 AI 大模型读
      · 一份 .tex       —— 可以编译成 PDF 的 LaTeX 源文件

【还有个重要职责：把人工批注"焊"回去】
    你在 review/ 里改的批注（第④步生成的），是在这一步被真正写进结果的。
    见函数 apply_corrections 的说明 —— 这是整个工具"可以人工纠正"的最后一环。

【Markdown → LaTeX 用什么转换？】
    优先用 pandoc（如果电脑上装了）—— 它是专门干这个的，
    处理表格、嵌套列表比我们自己写的规则稳得多。
    没装就退回本文件内置的轻量转换器（能覆盖标题/公式/图片/表格/粗斜体这些常见结构）。
    ⚠️ 用哪个转换器会在日志里说明，不会悄悄降级让你不知道。

【产物】
    out/{文档名}/{文档名}.md      给大模型读（公式是 LaTeX 代码）
    out/{文档名}/{文档名}.tex     给 LaTeX 用
    out/{文档名}/images/          从 PDF 里抠出来的图片
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps.common import (
    OUT, REVIEW, doc_dir, get_logger, load_json, safe_name, save_json,
    step_banner,
)

# YAML 是批注文件的格式。没装这个库的话，批注功能就跳过（其他照常跑）。
try:
    import yaml
    _HAS_YAML = True
except ImportError:                                     # noqa: N816
    _HAS_YAML = False


# ======================================================================
# 一、把人工批注写回正文 —— "可以人工纠正"的最后一环
# ======================================================================
def apply_corrections(md: str, name: str, log) -> tuple[str, int]:
    """读 review/{文档名}.yaml，把人工改好的内容替换进正文。

    输入：md = 正文文本；name = 文档名
    输出：(改好的正文, 成功替换了几条)

    ★ 为什么必须有这个函数？
      没有它的话，你在批注文件里辛辛苦苦改完，再执行"重新组装"，
      程序会拿 MinerU 的原始识别结果重新生成一遍 —— 你的改动全丢了。
      等于白改。

    批注文件的格式约定（和第④步生成的文件对应）：
        found:     "识别错的原文片段"   ← 用来定位，别动它
        corrected: "你改成的正确内容"   ← 你把正确答案填这里
    """
    if not _HAS_YAML:
        return md, 0
    ypath = REVIEW / f"{name}.yaml"
    if not ypath.exists():        # 没有批注文件说明用户没改过，正常
        return md, 0

    try:
        data = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    except Exception as exc:                            # noqa: BLE001
        # 批注文件被改坏了（比如缩进错了）不应该让整批任务挂掉
        log.warning("      批注文件解析失败（%s），跳过回灌", exc)
        return md, 0

    n_ok = n_skip = 0
    for it in data.get("items", []):
        corr = (it.get("corrected") or "").strip()
        found = (it.get("found") or "").strip()

        # ★ 判断"这条要不要应用"的依据是【corrected 填没填】，
        #   而不是【action 是不是 fix】。
        #   原因：本函数每次都从 MinerU 原始输出重新生成正文，
        #   所以修正必须"每次都重放"。但第二轮跑质检时，那条记录会被
        #   标记成 done（已处理）—— 如果只认 action=fix，第二轮修正就丢了（踩过）。
        if not corr or not found:
            continue

        if found and found in md:
            md = md.replace(found, corr)     # 定位到了，替换
            n_ok += 1
        else:
            # 定位不到就明确报警，绝不静默跳过 ——
            # 静默失败比直接报错更危险：你以为改上了，其实没改。
            log.warning("      批注 %s 未匹配到原文（found 片段已变？），跳过",
                        it.get("id", "?"))
            n_skip += 1

    if n_ok or n_skip:
        log.info("      回灌批注：成功 %d 条 / 跳过 %d 条", n_ok, n_skip)
    return md, n_ok


# ======================================================================
# 二、把 base64 内嵌图片还原成独立的图片文件
# ======================================================================
# 匹配 Markdown 里这种写法： ![](data:image/png;base64,iVBORw0KGgo...)
# "base64"是一种把图片二进制数据转成纯文字的编码方式。
B64_IMG = re.compile(
    r"!\[[^\]]*\]\(data:image/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)\)"
)


def extract_base64_images(md: str, img_dir: Path) -> tuple[str, int]:
    """把正文里内嵌的 base64 图片解码成真实图片文件，并改写引用。

    输入：md = 正文；img_dir = 图片存到哪个目录
    输出：(改好的正文, 解出来几张图)

    【为什么要做这件事？】
      MinerU 遇到某些 PDF 时，会把图片直接"塞"进 Markdown 文本里（内嵌 base64）。
      好处是单个文件自包含，坏处是**体积爆炸**：
      实测 2 页就能撑到 400 KB，200 页会变成几百 MB 的文本文件 ——
      传也传不动，AI 也读不了。所以统一抠出来存成独立文件。
    """
    import base64

    ext_map = {"jpeg": "jpg", "jpg": "jpg", "png": "png",
               "gif": "gif", "webp": "webp", "bmp": "bmp"}
    counter = {"n": 0}      # 用字典是因为下面嵌套函数里要能改到它

    def _repl(m: re.Match) -> str:
        """每匹配到一张内嵌图片就调用一次：解码 → 存盘 → 换掉原文。"""
        fmt = ext_map.get(m.group(1).lower(), "png")
        raw = re.sub(r"\s+", "", m.group(2))     # base64 里可能夹换行，先去掉
        counter["n"] += 1
        fn = f"embedded_{counter['n']:04d}.{fmt}"    # 命名：embedded_0001.png
        try:
            (img_dir / fn).write_bytes(base64.b64decode(raw))
        except Exception:                                   # noqa: BLE001
            return m.group(0)         # 解不开就原样留着，别把正文弄坏
        return f"![]({fn})"           # 引用改成本地文件名

    return B64_IMG.sub(_repl, md), counter["n"]


# ======================================================================
# 三、LaTeX 文件的"外壳"模板
# ======================================================================
# 下面这段是 LaTeX 文件的固定开头，作用是告诉 LaTeX：
# 用什么纸型、要引入哪些功能包（数学公式、插图、表格、超链接…）。
# __IMGDIR__ 和 __TITLE__ 是占位符，后面会被替换成真实值。
TEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
% 由 pdf2latex 自动生成 —— 公式为 LaTeX 数学，图片为独立文件
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}   % 数学公式支持
\usepackage{graphicx}                   % 插图支持
\usepackage{booktabs}                   % 三线表（好看的表格）
\usepackage{longtable}                  % 跨页长表格
\usepackage{geometry}                   % 页面边距
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}   % 超链接
\geometry{margin=2.5cm}
\graphicspath{{__IMGDIR__/}}            % 图片默认去哪找
\setlength{\parindent}{0pt}             % 首行不缩进
\setlength{\parskip}{0.6em}             % 段落之间留点空
\title{__TITLE__}
\date{}
\begin{document}
\maketitle
"""

TEX_POSTAMBLE = "\n\\end{document}\n"   # LaTeX 文件的固定结尾


# ======================================================================
# 四、内置的 Markdown → LaTeX 转换器
#     （电脑上没装 pandoc 时用这个兜底）
# ======================================================================
def _esc_text(s: str) -> str:
    """把普通文字里的 LaTeX 特殊字符转义，防止它们被当成命令。

    【为什么需要？】LaTeX 里 # % & _ ~ ^ \\ 这些符号有特殊含义。
    比如正文里写"折扣率是 50%"，那个 % 会让 LaTeX 把后面整行当注释吞掉。
    所以要在前面加个反斜杠，变成 \\% 表示"我就是个百分号"。

    ★ 关键：$...$ 里面的数学公式**不动** —— 那里本来就是 LaTeX 代码。
    """
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]

        # 遇到 $ 说明进了数学区，一路复制到配对的 $ 为止，中间原样保留
        if ch == "$":
            j = s.find("$", i + 1)
            if j == -1:               # 没有配对的 $，剩下的原样输出
                out.append(s[i:]); break
            out.append(s[i:j + 1]); i = j + 1; continue

        # 遇到反斜杠说明是 LaTeX 命令（如 \alpha），整条命令原样保留
        if ch == "\\":
            j = i + 1
            while j < n and s[j].isalpha():   # 命令名是连续的字母
                j += 1
            out.append(s[i:j]); i = j; continue

        # 需要转义的五个特殊字符
        if ch in "#%&_":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(ch)            # 普通字符原样输出
        i += 1
    return "".join(out)


def _inline(s: str) -> str:
    """把行内的 Markdown 标记换成 LaTeX 写法。

    **加粗**  → \\textbf{加粗}
    *斜体*    → \\emph{斜体}
    `代码`    → \\texttt{代码}
    """
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    return s


def _md_table_to_tex(rows: list[str]) -> str:
    """把 Markdown 表格转成 LaTeX 的 tabular 表格。

    输入：Markdown 表格的每一行，形如  | 姓名 | 分数 |
    输出：LaTeX 表格代码

    Markdown 表格长这样（第二行是横线，只用来分隔表头，不是真数据）：
        | 姓名 | 分数 |
        |------|------|
        | 张三 | 95   |
    """
    # 按 | 拆开每一行，得到一个个格子
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]

    # 丢掉 |---|---| 这条分隔线
    body = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x or "-") for x in c)]
    if not body:
        return ""

    # 各行列数可能不齐，按最宽的一行补齐（短的用空字符串填）
    ncol = max(len(r) for r in body)
    body = [r + [""] * (ncol - len(r)) for r in body]

    lines = [r"\begin{center}", r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule"]
    for k, r in enumerate(body):
        lines.append(" & ".join(_inline(_esc_text(c)) for c in r) + r" \\")
        if k == 0:
            lines.append(r"\midrule")      # 表头下面画一条线
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def md_to_tex(md: str, img_rel: str) -> str:
    """把整篇 Markdown 正文逐行转成 LaTeX。返回 LaTeX 正文部分（不含外壳）。

    处理顺序是"一行一行看，认出是什么结构就用对应的 LaTeX 写法"：
        独立公式块 $$...$$  →  equation 环境
        表格   |...|        →  tabular
        图片   ![](...)     →  figure + includegraphics
        标题   ### 标题      →  \\section / \\subsection …
        列表   - 项目 / 1. 项目 → itemize / enumerate
        其他               →  普通段落
    """
    out: list[str] = []
    lines = md.splitlines()
    i, n = 0, len(lines)
    in_math_block = False

    while i < n:
        ln = lines[i]

        # ---- 独立公式块：$$ ... $$ ----
        if ln.strip().startswith("$$"):
            if not in_math_block:
                in_math_block = True
                body = ln.strip().strip("$")
                # 情况 A：一行就写完了，如 $$E=mc^2$$
                if body and ln.strip().endswith("$$") and len(ln.strip()) > 4:
                    out.append("\\begin{equation*}\n" + body + "\n\\end{equation*}")
                    in_math_block = False
                else:
                    # 情况 B：公式跨了好几行，逐行收集到结尾的 $$ 为止
                    buf = [body] if body else []
                    i += 1
                    while i < n and not lines[i].strip().endswith("$$"):
                        buf.append(lines[i]); i += 1
                    if i < n:
                        buf.append(lines[i].strip().rstrip("$"))
                    out.append("\\begin{equation*}\n" + "\n".join(buf).strip()
                               + "\n\\end{equation*}")
                    in_math_block = False
            i += 1
            continue

        # ---- 表格：本行以 | 开头，且下一行是 |---|---| 分隔线 ----
        if ln.strip().startswith("|") and i + 1 < n and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            out.append(_md_table_to_tex(tbl))
            out.append("")
            continue

        # ---- 图片：![](路径) ----
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln.strip())
        if m:
            src = m.group(1)
            name = Path(src).name        # 只取文件名，路径由 graphicspath 负责
            out.append("\\begin{figure}[htbp]\n  \\centering\n"
                       f"  \\includegraphics[width=0.85\\textwidth]{{{img_rel}/{name}}}\n"
                       "\\end{figure}")
            i += 1
            continue

        # ---- 标题：# 的个数决定层级 ----
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(lvl, "paragraph")
            title = _inline(_esc_text(m.group(2)))
            # 四级及以下 LaTeX 没有对应命令，用星号版（不编号）
            out.append(f"\\{cmd}*{{{title}}}" if lvl >= 4 else f"\\{cmd}{{{title}}}")
            i += 1
            continue

        # ---- 无序列表：- 项目 / * 项目 ----
        if re.match(r"^\s*[-*+]\s+", ln):
            items = []
            while i < n and re.match(r"^\s*[-*+]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*+]\s+", "", lines[i]))
                i += 1
            out.append("\\begin{itemize}")
            out += [r"  \item " + _inline(_esc_text(x)) for x in items]
            out.append("\\end{itemize}")
            continue

        # ---- 有序列表：1. 项目 ----
        if re.match(r"^\s*\d+[.)]\s+", ln):
            items = []
            while i < n and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+[.)]\s+", "", lines[i]))
                i += 1
            out.append("\\begin{enumerate}")
            out += [r"  \item " + _inline(_esc_text(x)) for x in items]
            out.append("\\end{enumerate}")
            continue

        # ---- 剩下的都当普通段落 ----
        if ln.strip():
            out.append(_inline(_esc_text(ln)))
        else:
            out.append("")            # 空行 → 保留空行，让 LaTeX 知道分段
        i += 1

    return "\n".join(out)


def try_pandoc(md_path: Path, tex_path: Path, img_rel: str, log=None) -> bool:
    """尝试用 pandoc 转换。成功返回 True，失败/没装返回 False。

    ⚠️ 失败一定要吭声！如果悄悄退回内置转换器，
       用户根本不知道自己拿到的是次一等的产物（踩过）。
    """
    exe = shutil.which("pandoc")
    if exe is None:
        return False                  # 没装 pandoc，这是正常情况，不用报警
    try:
        subprocess.run(
            [exe, str(md_path), "-f", "markdown+tex_math_dollars",
             "-t", "latex", "-s", "--wrap=none", "-o", str(tex_path)],
            check=True, capture_output=True, timeout=180,
        )
        return True
    except Exception as exc:                            # noqa: BLE001
        if log:
            log.warning("      pandoc 失败（%s），改用内置转换器", exc)
            if isinstance(exc, subprocess.CalledProcessError):
                # pandoc 自己的报错信息很有用，打出来方便排查
                err = (exc.stderr or b"").decode("utf-8", "replace").strip()
                if err:
                    log.warning("      pandoc stderr: %s", err[:200])
        return False


# ======================================================================
# 五、把图片收拢到输出目录，并把正文里的引用改对
# ======================================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def stage_images(md: str, img_root: Path, img_dir: Path,
                 img_rel: str) -> tuple[str, int]:
    """把图片全部落到 img_dir，并把正文里的引用路径改写正确。

    输入：md = 正文；img_root = MinerU 输出目录（图片散落在它下面）；
          img_dir = 图片复制到哪儿；img_rel = 正文里引用图片的前缀
    输出：(改写好的正文, 一共收了多少张图)

    三步顺序固定：先解 base64 内嵌图，再复制真实图片文件，最后统一改写引用
    （改写必须在文件都落盘之后，才能按"引用→文件名"的对照表来改）。

    ★ 撞名要改名，绝不能直接覆盖 —— MinerU 可能把图片放在多个子目录下，
      不同子目录里有同名文件；按文件名平铺复制的话，后一张会盖掉前一张，
      一张图的内容被悄悄替换、正文引用还指向同一个文件，从结果上完全看不出来。
    """
    # 1. 解 base64 内嵌图片（文件名形如 embedded_0001.png）
    md, n_b64 = extract_base64_images(md, img_dir)

    # 2. 复制真实图片文件
    #
    # ★ 只搬"正文真正引用到的"图片。
    #   第②步用 zip 模式时，MinerU 会把表格切图也一并放进 images/
    #   （page_14_table_4.jpg 之类）—— 那是它内部用来识别表格的中间产物，
    #   表格已经转成 Markdown 表格了，这些切图正文里根本没引用。
    #   全搬过去只会在输出目录里堆一堆用不上的文件。
    # 注意：这里只取了**一个**捕获组，所以是 group(1)（图片路径）
    refs = {m.group(1) for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md)}
    refs_rel = {r.lstrip("./") for r in refs}
    refs_base = {Path(r).name for r in refs}

    used = {f.name for f in img_dir.iterdir() if f.is_file()}   # 已被 base64 占用的名字
    name_map: dict[str, str] = {}        # 原引用写法 → 实际落盘的文件名
    copied = 0

    for p in sorted(img_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        if p.relative_to(img_root).as_posix() not in refs_rel and p.name not in refs_base:
            continue                     # 正文没引用它，跳过

        dst_name = p.name
        if dst_name in used:                        # 撞名 → 加父目录名做区分
            prefix = p.parent.name or "sub"
            dst_name = f"{prefix}_{p.stem}{p.suffix}"
            n = 2
            while dst_name in used:                 # 还撞就继续编号
                dst_name = f"{prefix}_{n}_{p.stem}{p.suffix}"
                n += 1

        used.add(dst_name)
        shutil.copy2(p, img_dir / dst_name)
        copied += 1

        # 记下两种可能的引用写法，供下面改写时匹配
        name_map[p.relative_to(img_root).as_posix()] = dst_name
        name_map.setdefault(p.name, dst_name)       # 同名时保留第一个

    # 3. 改写正文里的图片引用：先按"完整相对路径"匹配（撞名时靠它区分），
    #    匹配不上再按文件名兜底（兼容正文只写了文件名的情况）。
    def _fix_ref(m: re.Match) -> str:
        ref = m.group(2)                      # ★ group(1) 是方括号里的说明文字，
                                              #   group(2) 才是圆括号里的图片路径
        if ref.startswith("data:"):           # base64 在上面第 1 步已处理
            return m.group(0)
        new = name_map.get(ref.lstrip("./")) or name_map.get(Path(ref).name)
        if new is None:
            return f"![{m.group(1)}]({img_rel}/{Path(ref).name})"   # 兜底：保持原行为
        return f"![{m.group(1)}]({img_rel}/{new})"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _fix_ref, md)

    return md, copied + n_b64


# ======================================================================
# 六、主流程
# ======================================================================
def run(src: Path, force: bool = False) -> list[dict]:
    """组装所有已成功识别的文档。这是给 run.py 调用的入口。

    参数：
        src   —— 本次的输入路径（用来限定"只处理这一次要处理的文档"）
        force —— 预留参数，保持和其他步骤接口一致

    返回：每份文档的组装结果列表。
    """
    log = get_logger()
    step_banner(log, "③", "组装（Markdown → LaTeX）")

    # ★ 只处理**本次 --input 里指定的**文档。
    #   以前是直接把 data/ 下所有缓存都捞出来处理，导致换个输入目录跑的时候，
    #   上次跑过的文档又被重新生成一遍，输出目录里混进无关内容（踩过）。
    from steps.common import DATA, list_docs
    wanted = {safe_name(p.stem) for p in list_docs(src)} if src else None

    # 把"第②步成功了的"文档挑出来
    convs = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        if wanted is not None and d.name not in wanted:
            continue                      # 不在本次范围内，跳过
        rec = load_json(d / "02_convert.json")
        if rec and rec.get("ok"):
            convs.append(rec)

    if not convs:
        raise RuntimeError("没有可组装的产物 —— 先跑第 ② 步（假设第 ② 步成功了？）")

    use_pandoc = shutil.which("pandoc") is not None
    log.info("转换器：%s", "pandoc" if use_pandoc else "内置轻量转换器")
    log.info("待组装 %d 份", len(convs))

    results = []
    t0 = time.time()

    for k, rec in enumerate(convs, 1):
        name = rec["name"]
        md_src = Path(rec["markdown"])
        if not md_src.exists():
            log.error("  [%d/%d] %-38s Markdown 不在了", k, len(convs), name[:38])
            continue

        # 输出结构：out/{文档名}/{文档名}.md  +  out/{文档名}/images/
        # （以前用 out/{文档名}_images 这种"兄弟目录"，导致 md 里图片引用
        #   得写成 ../ 才对，质检还会误报"图片不存在" —— 踩过）
        out_dir = OUT / name
        img_dir = out_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        # ★ 每次组装前先清空 images/ —— 重跑必须幂等。
        #   MinerU 每次输出的图片文件名可能不一样（同一张图这次叫
        #   page_10_image_4.jpg，上次叫 images_4_page_10_image_4.jpg），
        #   不清空就会把上一轮的图留在目录里，越跑越多、正文只引用其中一张，
        #   看上去像"图片莫名变多了"。实测踩过：连跑几次后 images/ 里堆了 5 个文件。
        if img_dir.exists():
            n_old = sum(1 for f in img_dir.rglob("*") if f.is_file())
            shutil.rmtree(img_dir)
            if n_old:
                log.info("        images/ 清掉上一轮的 %d 个文件（重跑不叠加）", n_old)
        img_dir.mkdir(parents=True, exist_ok=True)
        IMGREL = "images"

        # ---- ① 图片搬到 out/，并改写正文里的引用路径 ----
        md = md_src.read_text(encoding="utf-8", errors="replace")
        md, copied = stage_images(md, md_src.parent, img_dir, IMGREL)

        # ---- ② 回灌人工批注（必须在写出 .md 之前做）----
        md, n_fix = apply_corrections(md, name, log)

        md_out = out_dir / f"{name}.md"
        md_out.write_text(md, encoding="utf-8")

        # ---- ③ 生成 .tex ----
        tex_out = out_dir / f"{name}.tex"
        done = False
        # ⚠️ 这里**不能**写成 `if use_pandoc and not force`。
        #    --force 的含义是"忽略已有产物重跑"，跟用哪个转换器毫无关系。
        #    之前多写了个 not force，结果一加 --force 就悄悄退回内置转换器（踩过）。
        if use_pandoc:
            done = try_pandoc(md_out, tex_out, IMGREL, log)
        if not done:
            # 用内置转换器
            body = md_to_tex(md, IMGREL)
            # ⚠️ 这里用 replace 而不是 Python 的 % 格式化：
            #    LaTeX 注释符号本身就是 %，会和 % 占位符打架（踩过）。
            head = (TEX_PREAMBLE
                    .replace("__IMGDIR__", IMGREL)
                    .replace("__TITLE__", _esc_text(name)))
            tex_out.write_text(head + body + TEX_POSTAMBLE, encoding="utf-8")

        # ---- ④ 统计一下成果，记进 JSON ----
        n_math = len(re.findall(r"\$[^$]+\$", md))          # 公式个数（粗略统计）
        n_tbl = len(re.findall(r"^\s*\|", md, re.M))        # 表格行数（粗略统计）
        info = {
            "name": name, "ok": True,
            "markdown": str(md_out), "tex": str(tex_out),
            "images_dir": str(img_dir), "n_images": copied,
            "n_math_inline": n_math, "n_table_lines": n_tbl,
            "n_corrections_applied": n_fix,                 # 应用了几条人工批注
            "converter": "pandoc" if (use_pandoc and done) else "builtin",
        }
        save_json(doc_dir(name) / "03_assemble.json", info)
        results.append(info)
        log.info("  [%d/%d] %-38s 图%d 公式%d 表%d行 → %s",
                 k, len(convs), name[:38], copied, n_math, n_tbl,
                 "pandoc" if info["converter"] == "pandoc" else "内置")

    log.info("")
    log.info("组装完成：%d 份 / 耗时 %.1fs → %s", len(results), time.time() - t0, OUT)
    return results


# 直接运行本文件时走这里（src=None 表示不限定范围，处理 data/ 下全部）
if __name__ == "__main__":
    run(None)
