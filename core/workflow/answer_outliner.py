"""AnswerOutliner：据【教学维度词表 + 书的 TOC】给"讲清楚"问题定讲解骨架。

explain 专用。结构【自上而下】来自固定教学维度词表（是什么/作用/组成/原理/适用·边界/关系）
+ 书的目录章节（撑"组成"高度），【绝不从召回派生】——召回片段只用来判这个主题该讲哪几维、
每维填什么子查询，碎细节（FIL_PAGE_UNDO_LOG 之类）不会被提成顶层小节。
出 list[Dimension]（label∈词表、query=检索子查询）。空/失败 → []，由 qa.explain 落 agent 兜底。
设计见 docs/superpowers/specs/2026-06-21-explain-lecturer-synthesis-design.md。
"""
import logging

from llama_index.core.llms import LLM

from core.workflow.query_dimension import Dimension, DimensionSet

logger = logging.getLogger(__name__)

# 固定教学维度词表：模型只能选用，不得自创；不在词表里的 label 一律过滤掉。
_ALLOWED_LABELS = {"是什么", "作用", "组成", "原理", "适用·边界", "关系"}

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
_OUTLINE_PROMPT = """你是讲师备课助手。下面给出一个用户想"讲清楚/讲透"的问题、知识库里宽召回到的相关片段，以及（若有）这本书的目录章节。请像备课的老师一样，【从固定教学维度词表里】挑选这个问题该讲的几个维度，定出讲解骨架。

固定教学维度词表（label 只能原样取自下面，不得自创，更不得把召回片段里的碎细节当成维度）：
- 是什么：概念的定义与定位
- 作用：解决什么问题、为什么需要（动机先行）
- 组成：由哪些部件/子结构构成
- 原理：部件怎么协作、工作机制
- 适用·边界：什么场景用、何时当心
- 关系：与相邻概念的联系或对比

定维度规则：
- 高度由问题决定：宽/入门问题（如"讲懂MySQL"）选靠前维度（是什么/作用/组成）、停在高处别下钻到部件内部细节；具体/深问题（如"MVCC怎么实现"）聚焦被问那一维（原理）下钻、前置维度一句带过。
- "组成"维度优先参考下面给出的【目录章节】——书的顶层章节就是部件高度；没给目录时按通用教学常识定组成。
- 每个维度配一个能独立检索的 query（含问题的主体技术实体，别只写"作用"这种裸词）。
- 维度数量自适应：原子概念 1~2 个即可，宽主题最多 {max} 个；按上面词表顺序排列。

只返回 JSON，不要其它任何内容：
{"dimensions":[{"label":"是什么","query":"……"},{"label":"组成","query":"……"}]}

问题：{query}

目录章节（可能为空）：
{toc}

召回片段：
{passages}"""


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class AnswerOutliner:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        query: str,
        passages: list[str],
        toc_hint: list[str] | None = None,
        max_items: int = 8,
    ) -> list[Dimension]:
        toc_text = "、".join(toc_hint) if toc_hint else "（无干净目录，按通用教学常识定\"组成\"）"
        prompt = (
            _OUTLINE_PROMPT.replace("{query}", query)
            .replace("{passages}", "\n---\n".join(passages) or "（无）")
            .replace("{toc}", toc_text)
            .replace("{max}", str(max_items))
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            data = DimensionSet.model_validate_json(text)
            dims = [
                Dimension(label=d.label.strip(), query=d.query.strip())
                for d in data.dimensions
                if d.label.strip() in _ALLOWED_LABELS and d.query and d.query.strip()
            ][:max_items]
            logger.info(
                "outline: 列出 %d 个教学维度：%s",
                len(dims), " | ".join(d.label for d in dims),
            )
            return dims
        except Exception as exc:
            logger.warning("outline 解析失败，返回空（explain 将落 agent 兜底）：%s", exc)
            return []
