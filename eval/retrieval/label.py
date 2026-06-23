"""阶段一：给 golden 可答类 query 标相关 chunk（pooling + LLM 二元判定），冻结到
eval/retrieval/dataset/golden.retrieval.jsonl。调 LLM，一次性；人工抽检后即用。

只标 retrievable/pending_split/other/ambiguous；missing_info/out_of_scope 跳过
（本无相关 chunk）。pooling = vector top-N ∪ bm25 top-N，judge 分批 0/1 判定。

运行（项目根）：python -m eval.retrieval.label
"""
import asyncio
import json
import os

from openai import AsyncOpenAI

from configs.embedding import configure_embedding
from configs.llm import configure_llm, deepseek_api_key
from core.rag.data_loader import RAGIndexManager
from core.retrieval.retrieve import make_retriever
from eval.config import CHROMA_COLLECTION, CHROMA_DIR, DATASET_DIR

POOL_N = 30          # 每路候选深度
JUDGE_BATCH = 10     # 每批判定 chunk 数
CHUNK_TRUNC = 600    # judge 时 chunk 正文截断字数

GOLDEN = os.path.join(DATASET_DIR, "golden.jsonl")
LABEL_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "golden.retrieval.jsonl"
)
LABEL_CATEGORIES = {"retrievable", "pending_split", "other", "ambiguous"}

_JUDGE_PROMPT = """判断每个检索片段是否与问题【直接相关】（能用于回答问题）。

问题：{question}

片段（按序号）：
{chunks}

只输出 JSON：键为片段序号(字符串)，值为 1(相关)或 0(不相关)，覆盖全部序号。
不要任何解释。例：{{"0": 1, "1": 0}}"""


def _chunk_text(node) -> str:
    return (node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")) or ""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def merge_pool(ranked_lists: list[list]) -> tuple[list[str], dict[str, object]]:
    """各路有序 NodeWithScore 列表 → (保序去重的 node_id 列表, node_id→node)。"""
    ordered_ids: list[str] = []
    id2node: dict[str, object] = {}
    for nodes in ranked_lists:
        for nws in nodes:
            nid = nws.node.node_id
            if nid not in id2node:
                id2node[nid] = nws.node
                ordered_ids.append(nid)
    return ordered_ids, id2node


def parse_judgement(text: str, idx_to_id: dict[int, str]) -> set[str]:
    """LLM JSON {局部序号: 0|1} → 判 1 且序号在范围内的 chunk_id 集合。容错。"""
    data = json.loads(_strip_fences(text))
    out: set[str] = set()
    for key, val in data.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue
        if idx in idx_to_id and int(val) == 1:
            out.add(idx_to_id[idx])
    return out


async def _judge_batch(gen, question, batch_ids, id2node) -> set[str]:
    """对一批 chunk 调 LLM 判 0/1。batch_ids：本批 chunk_id 列表。"""
    idx_to_id = {i: cid for i, cid in enumerate(batch_ids)}
    chunks = "\n".join(
        f"[{i}] {_chunk_text(id2node[cid])[:CHUNK_TRUNC]}" for i, cid in idx_to_id.items()
    )
    prompt = _JUDGE_PROMPT.format(question=question, chunks=chunks)
    resp = await gen.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens=400,
    )
    return parse_judgement(resp.choices[0].message.content, idx_to_id)


async def _build_pool(query, idx, hybrid) -> tuple[list[str], dict[str, object]]:
    """vector top-N ∪ bm25 top-N 候选池（bm25 复用 HybridRetriever 内部路径）。"""
    dense = await make_retriever("vector").retrieve(
        query, index_manager=idx, book_titles=None, top_k=POOL_N
    )
    await hybrid._ensure_bm25(idx)
    sparse = hybrid._bm25_search(query, None, POOL_N)
    return merge_pool([dense, sparse])


async def _label_one(gen, query, idx, hybrid) -> list[str]:
    ids, id2node = await _build_pool(query, idx, hybrid)
    relevant: set[str] = set()
    for start in range(0, len(ids), JUDGE_BATCH):
        batch = ids[start:start + JUDGE_BATCH]
        relevant |= await _judge_batch(gen, query, batch, id2node)
    return [cid for cid in ids if cid in relevant]   # 保 pooling 序


async def main() -> None:
    with open(GOLDEN, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    configure_llm()
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
    hybrid = make_retriever("hybrid")
    gen = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=deepseek_api_key)

    out_rows: list[dict] = []
    zero_hit: list[str] = []
    for r in rows:
        q = r["user_input"]
        cat = r.get("category", "")
        if cat not in LABEL_CATEGORIES:
            out_rows.append({"user_input": q, "category": cat,
                             "relevant_chunk_ids": [], "skipped": True})
            continue
        try:
            rel = await _label_one(gen, q, idx, hybrid)
        except Exception as exc:  # noqa: BLE001 — 单条失败不中断，标 skipped
            print(f"[warn] 判定失败，跳过：{q[:30]} | {type(exc).__name__}: {exc}")
            out_rows.append({"user_input": q, "category": cat,
                             "relevant_chunk_ids": [], "skipped": True})
            continue
        skipped = not rel
        if skipped:
            zero_hit.append(q)
        out_rows.append({"user_input": q, "category": cat,
                         "relevant_chunk_ids": rel, "skipped": skipped})
        print(f"[{cat}] {q[:40]} → {len(rel)} 相关")

    os.makedirs(os.path.dirname(LABEL_OUT), exist_ok=True)
    with open(LABEL_OUT, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    labeled = sum(1 for r in out_rows if not r["skipped"])
    print(f"\n已写 {LABEL_OUT}：{labeled}/{len(out_rows)} 条有相关标注")
    if zero_hit:
        print(f"[抽检] {len(zero_hit)} 条零命中（已标 skipped），建议人工核对：")
        for q in zero_hit:
            print(f"  - {q[:50]}")


if __name__ == "__main__":
    asyncio.run(main())
