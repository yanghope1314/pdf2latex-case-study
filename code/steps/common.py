# -*- coding: utf-8 -*-
"""
common.py —— 公用小工具（四个步骤都要用到的东西，统一放这里）

【这个文件是干什么的？】用大白话说：
    四个步骤文件都要做三件同样的事 ——
      ① 知道文件该存到哪个文件夹
      ② 把运行过程记下来（出错了能查）
      ③ 把中间结果存成文件（方便人工检查和修改）
    与其在四个文件里各写一遍，不如都放这里，写一次到处用。

【给非计算机专业的读者】把它想成"流水线的传送带"：
    第①步把结果写进 data/ 文件夹，第②步从 data/ 读出来接着干。
    因为每一步的结果都落了盘，所以任何一步都能单独重跑，不用从头再来 ——
    这正是本工具"可以人工纠正"的基础。

【和其他文件的关系】
    step1_preflight.py  预检   ┐
    step2_convert.py    转换   ├─ 四个步骤文件，都从这里 import 公用函数
    step3_assemble.py   组装   │
    step4_review.py     质检   ┘
    run.py              总指挥，按顺序调用上面四个
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------
# 一、文件夹都在哪儿 —— 全部以"本文件所在位置"为基准往上找
# ----------------------------------------------------------------------
# 这样写的好处：整个项目拷到别的电脑、别的盘符，都不用改代码。
# ROOT 就是 pdf2latex/ 这个项目根目录。
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"      # ★ 中间结果（JSON）。人可以打开看、可以直接改
LOGS = ROOT / "logs"      # ★ 运行日志。每次跑都会生成一个带时间戳的 .log
REVIEW = ROOT / "review"  # ★ 人工批注（YAML）。改完能回灌进最终结果
OUT = ROOT / "out"        # 最终产物：.md / .tex / images/

# 这四个文件夹如果不存在就建出来。
# 也就是说：删掉它们没关系，程序一跑会自动重建（所以它们不必提交/保留）。
for _d in (DATA, LOGS, REVIEW, OUT):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 二、日志 —— 让程序"说话"，说给屏幕和文件两个地方听
# ----------------------------------------------------------------------
_LOGGER: logging.Logger | None = None


def get_logger(name: str = "pdf2latex") -> logging.Logger:
    """拿到一个日志对象。**同一个程序里只会创建一次**，之后都复用。

    它同时往两个地方写字：
      · 屏幕        —— 让你跑的时候能看见进度（只显示 INFO 以上）
      · logs/*.log  —— 存下来备查（连 DEBUG 细节都记）

    为什么要记文件？因为 OCR 跑一晚上、或者报错一闪而过的时候，
    有日志文件才能回头查"当时到底发生了什么"。

    【中文乱码的坑】Windows 控制台默认不是 UTF-8，中文打印出来会是乱码。
      这里强制把输出切成 UTF-8。
      配套约定：调用方（start.bat / launcher.py）也要先执行 chcp 65001，
      两边编码对上才不会乱码。Git Bash 本身就是 UTF-8，天然没问题。
      ⚠️ 如果你在没 chcp 的裸 CMD 里直接敲 python run.py，中文会乱码 ——
         所以推荐双击 start.bat 或 launcher.py，别手敲命令。
    """
    global _LOGGER
    if _LOGGER is not None:      # 已经建过了，直接复用，别重复挂处理器
        return _LOGGER

    # 把标准输出切成 UTF-8，加 errors="replace" 是防止个别怪字符直接让程序崩掉
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)   # 这个级别是"闸门"，处理器上再各自过滤
    lg.propagate = False         # 不要再往上层传，否则会重复打印一遍
    if lg.handlers:              # 防御：万一别的模块已经配过了
        _LOGGER = lg
        return lg

    # 每行日志长这样：  14:23:05 | INFO    | 找到 3 份 PDF
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    # 处理器①：打印到屏幕（只显示 INFO 以上，DEBUG 太啰嗦）
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    # 处理器②：写进文件，文件名带时间戳，这样每次跑都有独立的一份
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOGS / f"run_{stamp}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)   # 文件里记得更全，方便排查
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    _LOGGER = lg
    return lg


def step_banner(log: logging.Logger, idx: str, title: str, extra: str = "") -> None:
    """在屏幕上打一条醒目的分界线，标明"现在开始跑第几步"。

    纯粹是为了让长年累月跑批处理的人一眼看清进度，没有技术含量。
    效果：
        ====================================================================
          [②] 转换（版面 / OCR / 公式 / 图片）  —— 某个PDF目录
        ====================================================================

    （示例里刻意不写具体盘符：本项目不许代码里出现任何盘符。）
    """
    log.info("")
    log.info("=" * 68)
    log.info("  [%s] %s%s", idx, title, ("  —— " + extra) if extra else "")
    log.info("=" * 68)


# ----------------------------------------------------------------------
# 三、中间结果的读写 —— "可以人工纠正"就是靠这两个函数
# ----------------------------------------------------------------------
# Windows 禁止出现在文件名里的字符（不含全角标点，中文标题里的"："是合法的）
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows 的保留设备名：拿这些名字建文件夹会指向设备而不是磁盘
_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


def safe_name(raw: str) -> str:
    """把文档名洗成一个能安全当文件夹名的字符串。

    【为什么要洗】
    本工具的中间产物是**按文档名建文件夹**存的
    （data/文档名/01_preflight.json），所以文件名里任何一个
    Windows 不认的字符，都会让后面某一步写文件时莫名其妙地失败。

    实测踩过的坑：一份 PDF 叫《…去哪儿：用AI重构旅行 .pdf》，
    文件名结尾多打了一个空格。mkdir 不报错、目录也建出来了，
    但 Win32 会把路径末尾的空格**静默吃掉**，
    于是下一步往那个目录里写 JSON 时直接 FileNotFoundError，
    报错信息里看不出真实原因，只能回头猜文件名。

    【洗三样东西】
      1. 首尾空白，以及**结尾的空格和点**（Win32 会静默吞掉它们）
      2. Windows 禁止的字符 < > : " / \\ | ? * 和控制字符 —— 换成下划线
      3. Windows 保留设备名（CON / PRN / AUX / NUL / COM1…）—— 加前缀避让
    """
    name = _ILLEGAL_CHARS.sub("_", str(raw)).strip()
    # 结尾的空格和点必须去掉：它们建得出来、读不回来，是最难查的一类坑
    name = name.rstrip(" .")
    if not name:
        return "未命名文档"                    # 洗完啥都不剩，给个兜底名
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        name = "_" + name
    return name[:120]                          # 留点余量，别顶着 260 字符上限


def doc_dir(doc_name: str) -> Path:
    """取某一份文档专属的中间结果文件夹，顺便建出来。

    例如文档叫《年报》，就返回 data/年报/，
    第①②③④步的结果分别存在里面：01_preflight.json、02_convert.json …
    一份文档一个文件夹，互不干扰。

    这里再洗一次名字（safe_name）：调用方大多已经洗过了，
    但建文件夹这一步是整个流程里最先碰到文件系统的地方，
    多一道保险比事后翻日志便宜。
    """
    d = DATA / safe_name(doc_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> None:
    """把 Python 的数据存成 JSON 文件。

    两个细节是特意这么写的：
      · ensure_ascii=False —— 中文原样存，不转成 \\u4e2d 这种转义码，
        这样你用记事本打开就能直接看懂。
      · indent=2          —— 缩进两格，让文件有层次、方便人眼阅读和手改。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_json(path: Path):
    """读回之前存的 JSON。

    文件不存在就返回 None（而不是报错）。
    调用方看到 None 就知道"这一步还没跑过"，于是决定要不要重新跑。
    """
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# 四、四个步骤的名字和顺序 —— 顺序就是依赖关系
# ----------------------------------------------------------------------
# 列表的先后顺序 = 执行的先后顺序。
# 第②步要用第①步的结果，所以①必须排在②前面。
# 命令行的 --redo 参数就是靠这个列表决定"从哪一步开始重跑"。
STEPS = ["preflight", "convert", "assemble", "review", "extract", "analyze"]

# 每一步的中文说明，打印给用户看（命令行里敲的仍是上面那个英文名）
STEP_TITLES = {
    "preflight": "预检（页数 / 有无文本层 / 是否扫描件）",
    "convert": "转换（版面分析 → OCR → 公式 → 图片）",
    "assemble": "组装（Markdown → LaTeX）",
    "review": "质检（生成批注文件）",
    "extract": "L2 抽取（代码规则：切分 → 匹配 → 量化陈述）",
    "analyze": "LLM 分析（L2 证据 → 结论）",
}


def steps_from(name: str) -> list[str]:
    """给定一个步骤名，返回"从它开始到结束"的所有步骤。

    例：steps_from("assemble") → ["assemble", "review"]
    用途：--redo assemble 表示"前两步的结果还能用，从组装开始往后重跑"。
    """
    if name not in STEPS:
        raise ValueError(f"未知步骤 {name!r}，可选：{STEPS}")
    return STEPS[STEPS.index(name):]


# ----------------------------------------------------------------------
# 五、找出"要处理哪些文件"
# ----------------------------------------------------------------------
# 这些后缀名的文件都交给 MinerU 处理。
# （清单抄自 MinerU 4.x 自己的 filetypes.py，它支持的不止 PDF）
DOC_EXTENSIONS: frozenset[str] = frozenset({
    # 文档
    "pdf",
    # 图片
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "jp2",
    # Office（含老二进制格式）
    "doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf",
    # OpenDocument
    "odt", "ods", "odp",
    # 其他
    "epub", "ofd", "html", "htm", "shtml",
})


def list_docs(src: Path) -> list[Path]:
    """找出待处理的文件清单。

    输入：一个文件夹路径，或者单个文件路径
    输出：符合条件的文件列表（按文件名排好序）

    三个细节：
      · 递归查找 —— 子文件夹里的文件也算，不用你一个个搬出来
      · 跳过 ~$ 开头的文件 —— 那是 Office 打开文档时生成的临时锁文件，
        不是真文档，当正常文件处理会报错（踩过）
      · 按文件名排序 —— 保证每次跑的顺序都一样，便于对照排查
    """
    src = Path(src)
    # 情况一：直接给了一个文件
    if src.is_file():
        return [src] if src.suffix.lower().lstrip(".") in DOC_EXTENSIONS else []

    # 情况二：给了一个文件夹，递归把里面所有支持的文件捞出来
    out = []
    for p in src.rglob("*"):
        if not p.is_file() or p.name.startswith("~$"):
            continue
        if p.suffix.lower().lstrip(".") in DOC_EXTENSIONS:
            out.append(p)
    return sorted(out)


# 兼容旧名字：以前只支持 PDF，函数叫 list_pdfs；现在支持多种格式，改名 list_docs。
# 留着这行是为了老代码/老命令仍然能用，不影响任何功能。
list_pdfs = list_docs
