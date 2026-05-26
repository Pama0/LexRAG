# CLAUDE.md

LLM 学习/实验项目，技术书籍知识库助手（上传 PDF + RAG 问答），使用智谱 AI GLM 作为底层 LLM。

## Environment

- Python 3.12+，虚拟环境 `.venv`
- 激活：`.venv\Scripts\activate` (PowerShell) / `source .venv/Scripts/activate` (Git Bash)
- API Key 存放在 `.env`：`ZHIPU_API_KEY=your_api_key`

## Running

```bash
python main.py                                   # book CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务（前端对接）
```

## ⚠️ Gotchas

### 模块导入：必须从项目根目录运行

```python
# ✅ 根目录脚本用绝对导入
from configs.llm import configure_llm
from core.rag.data_loader import load_and_process_data

# ✅ 子模块内用相对导入
from .rag.pdf_parser import BookPDFParser

# ❌ 不要直接运行子目录脚本
# python core/rag/data_loader.py  → ModuleNotFoundError
```

### 分层：core 不依赖 api

依赖方向单向 `api/`(Web) → `core/`(领域) → `configs/`。守卫：`python scripts/check_layering.py`。

### 工具在组装层创建，注入 Agent

`api/main.py`（Web）与根 `main.py`（CLI）在各自组装层用
`core.tools.book_tools.create_book_search_tool / create_list_books_tool`
创建工具，注入 `core.agent.agent.BookAgent`。新增 workflow 工具见 `core/workflow/README.md`。

### 法条遗留

早期法律条文 RAG 代码已冻结于 `legacy/`（不保证可运行，见 `legacy/README.md`）。

## Code Style

- 所有 I/O 操作用 `async/await`
- 函数签名加类型注解
- 中文注释可接受
