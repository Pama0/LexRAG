"""法条引用关系抽取器 - 从条文文本中识别跨条文引用"""
import re
from dataclasses import dataclass, field
from typing import Optional

from legacy.rag.parser import cn_num_to_int


@dataclass
class Citation:
    """一条法条引用关系"""
    source_article: str               # 引用方条文号（中文），如 "二十"
    source_article_int: int           # 引用方条文号（数字），如 20
    target_article: str               # 被引用条文号（中文），如 "十八"
    target_article_int: int           # 被引用条文号（数字），如 18
    citation_type: str                # "internal" 本法引用 / "external" 外部法规引用
    citation_verb: str                # 动词：依照/参照/按照/适用/依据/准用/比照/援引
    target_law: Optional[str] = None  # 外部法规名（仅 B 类）
    context: str = ""                 # 引用所在原文片段


# ==================== 正则模式 ====================

# 引转动词
_VERB = r"(依照|参照|按照|适用|依据|准用|比照|援引)"

# A 类：本法引用 — "依照本法第X条" / "依据本办法第Y条"
_SELF_REF = r"(本法|本条例|本办[法规及]*|本规定|本细则|本章)"
_ARTICLE_CN = r"第[一二三四五六七八九十百零]+条"
# 被引用条文后可能跟"第X款"、"第X项"、范围描述等
_ARTICLE_SUFFIX = r"[^，。；\n]{0,30}"

PATTERN_INTERNAL_EXPLICIT = re.compile(
    rf"{_VERB}\s*{_SELF_REF}\s*({_ARTICLE_CN}{_ARTICLE_SUFFIX})"
)

# A 类省略自指：条文中直接 "依照第X条" 而省略 "本法"
# 需排除：外部法规引用（前面有《》）和已匹配的显式引用
PATTERN_INTERNAL_IMPLICIT = re.compile(
    rf"{_VERB}\s*({_ARTICLE_CN}{_ARTICLE_SUFFIX})"
)

# B 类：外部法规引用 — "依照《XXX》第X条"
PATTERN_EXTERNAL = re.compile(
    rf"{_VERB}\s*《([^》]+)》[^。]{{0,15}}?({_ARTICLE_CN}{_ARTICLE_SUFFIX})"
)

# 从被引用条文中拆分出多个条文号，如 "第五十三条、第五十六条"
_SPLIT_ARTICLES = re.compile(rf"({_ARTICLE_CN})")


def _parse_target_article(target_text: str) -> list[tuple[str, Optional[int]]]:
    """从被引用文本中提取所有条文号

    例: "第五十三条、第五十六条" → [("五十三", 53), ("五十六", 56)]
    例: "第十八条" → [("十八", 18)]
    """
    matches = _SPLIT_ARTICLES.findall(target_text)
    results = []
    for article_str in matches:
        # 去掉 "第" 和 "条"
        cn_num = article_str[1:]  # 去掉 "第"
        if cn_num.endswith("条"):
            cn_num = cn_num[:-1]
        num = cn_num_to_int(cn_num)
        results.append((cn_num, num))
    return results


def extract_citations(
    text: str,
    file_name: str = "",
    current_article: Optional[str] = None,
) -> list[Citation]:
    """从条文文本中抽取所有引用关系

    Args:
        text: 条文完整文本
        file_name: 所属法规文件名，用于区分内部/外部引用
        current_article: 当前条文号（中文），如 "二十"。若为 None 则从文本开头提取

    Returns:
        Citation 列表
    """
    citations = []

    # 提取当前条文号
    if current_article is None:
        m = re.match(rf"({_ARTICLE_CN})", text.strip())
        if m:
            current_article = m.group(1)[1:]  # 去掉 "第"
            if current_article.endswith("条"):
                current_article = current_article[:-1]

    if not current_article:
        return citations

    source_int = cn_num_to_int(current_article)
    if source_int is None:
        return citations

    # 预处理：去除章节上下文行和条文号标头，防止跨行误匹配
    # 例: "【法规名】 第四章 行政处罚的管辖和适用\n第二十二条　内容..."
    # →  "内容..."
    clean_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 跳过章节上下文行（以 【 开头）
        if line.startswith("【"):
            continue
        # 去除条文号标头（如 "第二十二条　" 或 "第二十二条："）
        header_match = re.match(rf"^{_ARTICLE_CN}[\s：:　]*", line)
        if header_match:
            line = line[header_match.end():]
        if line:
            clean_lines.append(line)
    clean_text = "\n".join(clean_lines)

    # 已匹配的 span，用于去重（避免隐式模式重复匹配显式已匹配的内容）
    matched_spans = []

    # ---- A 类显式：依照本法第X条 ----
    for m in PATTERN_INTERNAL_EXPLICIT.finditer(clean_text):
        verb = m.group(1)
        target_text = m.group(3)
        for cn_num, num in _parse_target_article(target_text):
            if num is None:
                continue
            citations.append(Citation(
                source_article=current_article,
                source_article_int=source_int,
                target_article=cn_num,
                target_article_int=num,
                citation_type="internal",
                citation_verb=verb,
                context=_truncate_context(clean_text, m.start(), m.end()),
            ))
        matched_spans.append((m.start(), m.end()))

    # ---- A 类隐式：依照第X条（省略 "本法"）----
    for m in PATTERN_INTERNAL_IMPLICIT.finditer(clean_text):
        # 跳过已被显式模式覆盖的
        if any(s <= m.start() < e for s, e in matched_spans):
            continue
        # 跳过前面有《的（那是外部引用）
        prefix = clean_text[max(0, m.start() - 15):m.start()]
        if "《" in prefix:
            continue
        verb = m.group(1)
        target_text = m.group(2)
        for cn_num, num in _parse_target_article(target_text):
            if num is None:
                continue
            citations.append(Citation(
                source_article=current_article,
                source_article_int=source_int,
                target_article=cn_num,
                target_article_int=num,
                citation_type="internal",
                citation_verb=verb,
                context=_truncate_context(clean_text, m.start(), m.end()),
            ))

    # ---- B 类：依照《XXX》第X条 ----
    for m in PATTERN_EXTERNAL.finditer(clean_text):
        verb = m.group(1)
        target_law = m.group(2)
        target_text = m.group(3)
        for cn_num, num in _parse_target_article(target_text):
            if num is None:
                continue
            citations.append(Citation(
                source_article=current_article,
                source_article_int=source_int,
                target_article=cn_num,
                target_article_int=num,
                citation_type="external",
                citation_verb=verb,
                target_law=target_law,
                context=_truncate_context(clean_text, m.start(), m.end()),
            ))

    # 过滤自引用（source == target 必然是误提取）
    citations = [c for c in citations if c.source_article_int != c.target_article_int]

    return citations


def _truncate_context(text: str, start: int, end: int, max_len: int = 80) -> str:
    """截取匹配附近的上下文"""
    ctx_start = max(0, start - 10)
    ctx_end = min(len(text), end + 20)
    ctx = text[ctx_start:ctx_end].replace("\n", " ").strip()
    if len(ctx) > max_len:
        ctx = ctx[:max_len] + "..."
    return ctx
