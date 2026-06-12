"""book 知识库 RAG workflow：judge_query → retrieve → synthesize。

judge_query 步骤先规范化 query（纠错、缩写展开），再判定是否够明确：宽泛则
自动收窄改写，最多 MAX_ROUNDS 轮，再进入检索。规范化对所有 query 生效，明确的
query 也用纠错后的版本检索。指代/缺上下文类问题由 Agent 消解;workflow 仅作兜底检测未消解的指代,不自行消解，
不在此处理。
"""
import json
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

MAX_ROUNDS = 2

_JUDGE_PROMPT = """你是检索 query 处理器，对下面的 query 依次做三步：先规范化，再降噪，再判定该 query 能否直接进入检索。
要求：如果问题已经足够清晰适合检索,以下三步皆可不对问题修改,不要强行改写

第一步 规范化（始终执行，只改形式不改意图）：
- 纠正错别字、明显的同音/形近字错误（如"装饰起"→"装饰器"），不确定时保留原词。
- 统一全半角、大小写。
- 仅展开毫无歧义的常见技术缩写（如 K8s→Kubernetes）。
规范化只修形式，严禁改变用户意图或新增用户没提到的话题。

第二步 降噪（去除口语化、礼貌性、请求词、无信息量的词，保留关键词语，实体，技术名词）                                                                                                                                                                                                                          
如（原始问题:小E,我想请问一下,MySQL有哪些锁啊?                                                                                                                                                                                                                                                                 
改写后的查询:MySQL有哪些锁）                                                                                                                                                                                                                                                                                   
降噪只删与检索无关的冗余措辞，严禁删除任何技术限定词、修饰语、实体、版本号或承载意图的词（如"聚簇""行级""全文""有哪些""区别""第3章"）。判据：一个词删掉后若会改变检索命中，就必须保留。                                                                                                                        
反例：「MySQL聚簇索引和二级索引的区别」不可降成「MySQL 索引」——删掉"聚簇""二级""区别"会毁掉意图。

第三步 判定该 query 能否直接进入检索（基于规范化和降噪后的 query）：
【可以】可确定指向具体的技术概念/章节/问题，能检索到精准、集中的内容。
特征：仅检索问题即可
返回 {"category":"retrievable","rewritten_query": "处理后的 query"}

【不可以】归入以下四类之一：

- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索（多为指代不明）。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
  返回 {"category":"missing_info","rewritten_query": "处理后的 query","clarify_reason": "需澄清的原因，如'这个索引'指代不明"}

- ambiguous（角度不定）：话题已具体、能集中命中，但用户想要的维度/立场未给，有多个合理答法不知道选哪个。
  特征：答案就一个主题，但有几种角度/立场可选。
  如「Vue和React哪个好」(缺选型维度)「Redis做缓存好吗」(缺评判角度)
  返回 {"category":"ambiguous","rewritten_query": "处理后的 query"，“ambiguous_reason”: "角度不定的原因，比如vue和React哪个好问题缺少评价好的维度"}

- pending_split （需要拆分）：问题显式包括多个实体。或话题大到要覆盖文档一整片内容，检索会命中大量分散结果。
  特征：答案需要罗列并列子项才完整。
  如「讲讲MySQL」「讲讲功能A和功能B」
  返回 {"category":"pending_split","rewritten_query": "处理后的 query","split_reason": "需要拆分的原因，如MySQL需要罗列子项，功能A和功能B需要分开拆解"}

- other（其他无法直接检索的情况，以上三类均不符合）
  返回 {"category":"other","rewritten_query": "处理后的 query"}
  
【不可以】归类的优先级：先判断信息是否不足，再判断问题是否角度不定，再判断问题是否需要拆分，均不符合则为other

对照：
  「怎么优化MySQL」→ pending_split（优化是一整片：索引/查询/配置/架构）
  「MySQL大表查询慢怎么优化」→ ambiguous（场景已具体，仍有索引/分区/改SQL几个角度）
  「Vue和React哪个好」→ ambiguous（缺"好"的维度，虽然两个实体，但仍为ambiguous）
  「Vue和React的区别」→ pending_split（不缺维度，需要拆分）

category 仅为[retrievable|pending_split|missing_info|ambiguous|other]不允许有其他词，rewritten_query 始终返回处理后的 query，结果只返回 JSON，不要其他任何内容。

query：{query}"""


class JudgeEvent(Event):
    query: str
    round: int = 0

class PreProcessEvent(Event):
    query: str
    book_titles: Optional[list[str]] = None
    allow_clarify: bool = True

class ClarifyEvent(Event):
    query: str
    clarify_reason: Optional[str] = None

class SplitEvent(Event):
    query: str

class AssumeEvent(Event):
    query: str


class RetrieveEvent(Event):
    query: str
    book_titles: Optional[list[str]] = None

class SynthesizeEvent(Event):
    query: str
    nodes: list

class ClarifyResult:
    """workflow 因 query 需澄清而终止时的载荷,区别于正常的 Response。"""

    def __init__(self, query: str, clarify_reason: str):
        self.query = query
        self.clarify_reason = clarify_reason


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 代码块围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()



class BookRagWorkflow(Workflow):
    def __init__(self, index_manager, llm: LLM, similarity_top_k: int = 5, **kw):
        super().__init__(**kw)
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k

    async def _preprocess_query(self, query: str) -> tuple[str, str, str]:
        """对query进行先规范化，再降噪，再判定明确性的预处理。并根据query的不足进行分流处理。

        解析失败一律当作 clear=True 并用原 query，绝不阻塞检索。
        """
        resp = await self.llm.acomplete(_JUDGE_PROMPT.replace("{query}", query))
        try:
            data = json.loads(_strip_fences(str(resp)))
            category = str(data["category"])
            rewritten = str(data.get("rewritten_query") or query).strip() or query
            clarify_reason=str(data.get("clarify_reason",""))
            return category, rewritten,clarify_reason
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return "clear", query,""

    def _make_filters(self, book_titles: Optional[list[str]]):
        if not book_titles:
            return None
        return MetadataFilters(filters=[
            MetadataFilter(
                key="book_title",
                operator=FilterOperator.IN,
                value=list(book_titles),
            ),
        ])

    async def _retrieve_nodes(self, query: str, book_titles: Optional[list[str]]):
        index = self.index_manager.get_index()
        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=self._make_filters(book_titles),
        )
        return await retriever.aretrieve(query)

    @step
    async def start(self, ev: StartEvent) -> PreProcessEvent:
        return PreProcessEvent(
            query=ev.query,
            book_titles=getattr(ev, "book_titles", None),
            allow_clarify=getattr(ev, "allow_clarify", True),
        )

    @step
    async def preprocess(self, ev: PreProcessEvent) -> "ClarifyEvent | RetrieveEvent | SplitEvent | AssumeEvent":
        category, q, clarify_reason = await self._preprocess_query(ev.query)
        match category:
            case "clear":
                return RetrieveEvent(query=q, book_titles=ev.book_titles)
            case "pending_split":
                return SplitEvent(query=q)
            case "missing_info":
                if ev.allow_clarify:
                    return ClarifyEvent(query=q, clarify_reason=clarify_reason)
                # 预算耗尽：不再澄清，降级为尽力检索（用 cleaned query）
                return RetrieveEvent(query=q, book_titles=ev.book_titles)
            case "ambiguous":
                return AssumeEvent(query=q)
            case "unclear":
                return RetrieveEvent(query=q, book_titles=ev.book_titles)
        return RetrieveEvent(query=q, book_titles=ev.book_titles)

    @step
    async def clarify(self, ev: ClarifyEvent) -> StopEvent:
        return StopEvent(result=ClarifyResult(
            query=ev.query,
            clarify_reason=ev.clarify_reason,
        ))

    @step
    async def split(self, ev: SplitEvent) -> StopEvent:
        return StopEvent(result=ClarifyResult(
            query=ev.query,
            clarify_reason=ev.clarify_reason,
        ))

    async def assume(self, ev: AssumeEvent) -> StopEvent:

    @step
    async def retrieve(self, ev: RetrieveEvent) -> "SynthesizeEvent | StopEvent":
        nodes = await self._retrieve_nodes(ev.query, ev.book_titles)
        if not nodes:
            return StopEvent(result=Response(response="", source_nodes=[]))
        return SynthesizeEvent(query=ev.query, nodes=nodes)

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> StopEvent:
        synthesizer = get_response_synthesizer(llm=self.llm)
        response = await synthesizer.asynthesize(query=ev.query, nodes=ev.nodes)
        return StopEvent(result=response)
