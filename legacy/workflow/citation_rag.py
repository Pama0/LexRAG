"""引用链 RAG 工作流 — 检索后自动扩展被引用条文"""
import logging

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.workflow import step, Context, Workflow, StartEvent, StopEvent
from llama_index.core.indices.vector_store.retrievers.auto_retriever.auto_retriever import (
    VectorIndexAutoRetriever,
)

from legacy.rag.citation_graph import CitationGraph
from legacy.rag.vector_store_info import VECTOR_STORE_INFO
from legacy.rag.auto_retriever_prompt import LEGAL_AUTO_RETRIEVER_PROMPT

logger = logging.getLogger(__name__)


def _generate_file_name_candidates(name: str) -> list[str]:
    """为 LLM 生成的 file_name 生成可能的正确变体

    常见问题:
    - 缺少 .docx 后缀: '行政处罚法' → '行政处罚法.docx'
    - 缺少 '中华人民共和国' 前缀: '行政处罚法.docx' → '中华人民共和国行政处罚法.docx'
    """
    candidates = []
    # 补 .docx 后缀
    if not name.endswith(".docx"):
        candidates.append(name + ".docx")
    # 补 '中华人民共和国' 前缀
    base = name.replace(".docx", "")
    if not base.startswith("中华人民共和国"):
        candidates.append("中华人民共和国" + base + ".docx")
    return candidates

# 引用扩展的 prompt 前缀
CITATION_RAG_PROMPT = """\
以下是检索到的法律条文。其中：
- [直接检索] 表示与用户查询直接相关的条文
- [引用扩展] 表示被直接检索条文引用的相关条文，用于补充上下文

请基于以上条文回答用户问题。如果回答中引用了被扩展的条文，请说明其引用来源。

"""


class CitationRAGWorkflow(Workflow):
    """引用链 RAG 工作流

    流程：
    1. AutoRetriever 检索 → 得到 top_k nodes
    2. 对每个 node 查 CitationGraph，获取被引用条文
    3. 按 file_name + article_no_int 精确获取被引用的 node
    4. 合并原始结果 + 引用扩展结果（去重）
    5. 带标记的 Response Synthesis
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLM,
        citation_graph: CitationGraph,
        expand_depth: int = 1,
        max_expansions: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.index = index
        self.llm = llm
        self.citation_graph = citation_graph
        self.expand_depth = expand_depth
        self.max_expansions = max_expansions

        self.auto_retriever = VectorIndexAutoRetriever(
            index=index,
            vector_store_info=VECTOR_STORE_INFO,
            llm=llm,
            prompt_template_str=LEGAL_AUTO_RETRIEVER_PROMPT,
            similarity_top_k=5,
            max_top_k=20,
        )

    @step
    async def query_with_citation(self, ctx: Context, ev: StartEvent) -> StopEvent:
        question = ev.get("query")
        query_bundle = QueryBundle(query_str=question)

        # ---- 第一步：AutoRetriever 检索 ----
        nodes = []
        try:
            spec = await self.auto_retriever.agenerate_retrieval_spec(
                query_bundle=query_bundle
            )
            filter_list = [(f.key, f.operator.value, f.value) for f in spec.filters]
            logger.info(f"AutoRetriever 提取: query='{spec.query}', filters={filter_list}")

            retriever, spec_query_bundle = self.auto_retriever._build_retriever_from_spec(spec)
            nodes = retriever.retrieve(spec_query_bundle)
        except Exception as e:
            logger.warning(f"AutoRetriever 失败: {e}")

        # 空结果回退策略
        if not nodes:
            has_file_name_filter = any(
                f.key == "file_name" for f in getattr(spec, "filters", [])
            )
            if has_file_name_filter:
                # 尝试修正 file_name（LLM 生成的文件名常与实际不匹配）
                file_name_filters = [f for f in spec.filters if f.key == "file_name"]
                other_filters = [f for f in spec.filters if f.key != "file_name"]
                original_name = file_name_filters[0].value if file_name_filters else ""

                # 生成候选文件名变体
                candidates = _generate_file_name_candidates(original_name)

                for candidate in candidates:
                    try:
                        trial_filters = other_filters + [
                            MetadataFilter(key="file_name", value=candidate)
                        ]
                        trial_spec = spec.model_copy(update={"filters": trial_filters})
                        retriever, spec_query_bundle = self.auto_retriever._build_retriever_from_spec(trial_spec)
                        trial_nodes = retriever.retrieve(spec_query_bundle)
                        if trial_nodes:
                            logger.info(f"file_name 修正成功: '{original_name}' → '{candidate}'")
                            nodes = trial_nodes
                            break
                    except Exception:
                        continue

                # 所有修正都失败，去掉 file_name 只保留其他过滤
                if not nodes and other_filters:
                    logger.info("file_name 修正失败，去掉 file_name 重试")
                    try:
                        new_spec = spec.model_copy(update={"filters": other_filters})
                        retriever, spec_query_bundle = self.auto_retriever._build_retriever_from_spec(new_spec)
                        nodes = retriever.retrieve(spec_query_bundle)
                    except Exception as e2:
                        logger.warning(f"重试也失败: {e2}")

            if not nodes:
                logger.warning("AutoRetriever 空结果，退回纯向量检索")
                nodes = self.index.as_retriever(similarity_top_k=5).retrieve(query_bundle)

        if not nodes:
            return StopEvent(result="未检索到相关法律条文。")

        # ---- 第二步：引用链扩展 ----
        expanded_nodes = self._expand_citations(nodes)

        # ---- 第三步：合并 + 去重 + 标记 ----
        all_nodes = self._merge_nodes(nodes, expanded_nodes)
        logger.info(f"引用链扩展: 原始 {len(nodes)} 条, 扩展 {len(expanded_nodes)} 条, 合并后 {len(all_nodes)} 条")

        # ---- 第四步：带标记的 Response Synthesis ----
        # 给节点文本加上 [直接检索] / [引用扩展] 标记
        for n in all_nodes:
            is_exp = n.node.metadata.get("is_citation_expansion", False)
            if is_exp:
                source = n.node.metadata.get("citation_source", "")
                n.node.text = f"[引用扩展·{source}]\n{n.node.text}"
            else:
                n.node.text = f"[直接检索]\n{n.node.text}"

        from llama_index.core import PromptTemplate
        synthesis_prompt = PromptTemplate(CITATION_RAG_PROMPT + "{context_str}")
        result = get_response_synthesizer(
            text_qa_template=synthesis_prompt,
        ).synthesize(
            query=question,
            nodes=all_nodes,
        )
        return StopEvent(result=result)

    def _expand_citations(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """从检索结果中提取被引用条文，精确获取对应 node"""
        # 收集所有需要扩展的 (file_name, article_int) 对
        to_expand: dict[tuple[str, int], Citation] = {}

        for node in nodes:
            file_name = node.node.metadata.get("file_name", "")
            article_int = node.node.metadata.get("article_no_int")
            if not file_name or article_int is None:
                continue

            # BFS 扩展引用链
            citations = self.citation_graph.expand(
                file_name=file_name,
                article_ints=[article_int],
                depth=self.expand_depth,
            )
            for cite in citations:
                key = (file_name, cite.target_article_int)
                if key not in to_expand:
                    to_expand[key] = cite

        if not to_expand:
            return []

        # 限制扩展数量
        if len(to_expand) > self.max_expansions:
            logger.info(f"引用扩展数 {len(to_expand)} 超过限制 {self.max_expansions}，截断")
            to_expand = dict(list(to_expand.items())[:self.max_expansions])

        # 按 file_name 分组批量查询
        expanded = []
        for (file_name, article_int), cite in to_expand.items():
            try:
                fetched = self.index.as_retriever(
                    similarity_top_k=1,
                    filters=MetadataFilters(
                        filters=[
                            MetadataFilter(key="file_name", value=file_name),
                            MetadataFilter(key="article_no_int", value=article_int),
                        ]
                    ),
                ).retrieve(QueryBundle(query_str=""))
                for n in fetched:
                    # 标记为引用扩展
                    n.node.metadata["is_citation_expansion"] = True
                    n.node.metadata["citation_source"] = f"第{cite.source_article}条引用"
                    expanded.append(n)
            except Exception as e:
                logger.debug(f"引用扩展查询失败 {file_name} 第{article_int}条: {e}")

        return expanded

    def _merge_nodes(
        self,
        direct_nodes: list[NodeWithScore],
        expanded_nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """合并原始和扩展节点，去重，标记来源"""
        seen: set[tuple[str, int]] = set()
        result = []

        # 原始节点标记
        for n in direct_nodes:
            file_name = n.node.metadata.get("file_name", "")
            article_int = n.node.metadata.get("article_no_int")
            n.node.metadata["is_citation_expansion"] = False
            key = (file_name, article_int) if article_int else (n.node.node_id,)
            if key not in seen:
                seen.add(key)
                result.append(n)

        # 扩展节点
        for n in expanded_nodes:
            file_name = n.node.metadata.get("file_name", "")
            article_int = n.node.metadata.get("article_no_int")
            key = (file_name, article_int) if article_int else (n.node.node_id,)
            if key not in seen:
                seen.add(key)
                result.append(n)

        return result
