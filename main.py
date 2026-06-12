"""book 知识库助手 CLI 入口。

组装 core 组件 + book 工具 + 主 agent，进入交互式对话。
（Web 服务入口见 api/main.py：python -m uvicorn api.main:app）
"""
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from configs.embedding import configure_embedding
from configs.llm import configure_llm
from core.rag.data_loader import RAGIndexManager
from core.workflow.doc_query_service import DocQueryService

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")


async def run() -> None:
    configure_embedding()
    llm = configure_llm()

    index_manager = RAGIndexManager(
        persist_dir=CHROMA_DIR,
        collection_name="book_knowledge",
    )

    service = DocQueryService(index_manager=index_manager, llm=llm)
    await service.chat()


if __name__ == "__main__":
    asyncio.run(run())
