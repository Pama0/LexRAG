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
from core.agent.agent import BookAgent
from core.rag.data_loader import RAGIndexManager
from core.tools.book_tools import create_book_search_tool, create_list_books_tool

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

    tools = [
        create_book_search_tool(index_manager, llm),
        create_list_books_tool(index_manager),
    ]
    agent = BookAgent(tools=tools, llm=llm)
    await agent.chat()


if __name__ == "__main__":
    asyncio.run(run())
