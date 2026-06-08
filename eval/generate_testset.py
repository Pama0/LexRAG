"""从 book chroma 切片用 ragas TestsetGenerator 生成测试集草稿。

走"复用已切块"路线：把 chroma 片段包成 LangChain Document，喂 generate_with_chunks。
产出 dataset/testset.draft.jsonl，人工校验后另存为 testset.jsonl 供 run_eval 使用。
运行（项目根目录）：python -m eval.generate_testset --size 50
"""
import argparse
import asyncio
import json
import os

from langchain_core.documents import Document as LCDocument


def chunks_to_langchain(documents: list[str], metadatas: list[dict]) -> list[LCDocument]:
    """把 chroma 的 (正文, 元数据) 逐条包成 LangChain Document，跳过空文本。"""
    out: list[LCDocument] = []
    for text, meta in zip(documents, metadatas):
        if not text or not text.strip():
            continue
        out.append(LCDocument(
            page_content=text,
            metadata={
                "book_title": (meta or {}).get("book_title", ""),
                "chapter": (meta or {}).get("chapter", ""),
                "page": (meta or {}).get("page", ""),
                "file_path": (meta or {}).get("file_path", ""),
            },
        ))
    return out


def load_book_chunks() -> list[LCDocument]:
    """从项目 chroma 全量取出 book 切片并转 LangChain Document。

    直接用 chromadb 读原始文本+元数据，绕开 RAGIndexManager——后者在 collection
    有数据时会急切构建 VectorStoreIndex（需要全局 embed_model），而此处只读不检索。
    """
    import chromadb

    from eval.config import CHROMA_COLLECTION, CHROMA_DIR

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    chunks = chunks_to_langchain(data["documents"], data["metadatas"])
    print(f"从 chroma 加载 {len(chunks)} 条 book 切片")
    return chunks


async def generate(size: int) -> None:
    from ragas.testset import TestsetGenerator
    from ragas.testset.persona import Persona
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

    from eval.config import DATASET_DIR, TESTSET_DRAFT_PATH, make_eval_embeddings, make_eval_llm

    chunks = load_book_chunks()
    if not chunks:
        raise SystemExit("chroma 无 book 切片，先入库（python main.py 入库流程）再生成测试集")

    gen_llm = make_eval_llm()
    gen_emb = make_eval_embeddings()

    personas = [
        Persona(
            name="tech_reader",
            role_description="正在阅读技术书的工程师，针对书中具体的技术概念、机制、章节提出有据可查的问题",
        ),
    ]

    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb, persona_list=personas)

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=gen_llm), 0.6),
        (MultiHopSpecificQuerySynthesizer(llm=gen_llm), 0.4),
    ]
    # 中文 prompt 适配
    for query, _ in distribution:
        prompts = await query.adapt_prompts("chinese", llm=gen_llm)
        query.set_prompts(**prompts)

    print(f"开始生成测试集（{size} 条）……")
    dataset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=size,
        query_distribution=distribution,
    )
    eval_dataset = dataset.to_evaluation_dataset()

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(TESTSET_DRAFT_PATH, "w", encoding="utf-8") as f:
        for sample in eval_dataset.to_list():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"草稿已写入 {TESTSET_DRAFT_PATH}（共 {len(eval_dataset.to_list())} 条）")
    print("⚠️ 人工校验后另存为 testset.jsonl，再跑 run_eval。")


def main():
    parser = argparse.ArgumentParser(description="生成 book RAG 测试集草稿")
    parser.add_argument("--size", type=int, default=50, help="测试集条数")
    args = parser.parse_args()
    asyncio.run(generate(args.size))


if __name__ == "__main__":
    main()
