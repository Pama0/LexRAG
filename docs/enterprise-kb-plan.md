# 书籍知识库智能助手 — 项目方案

> **定位**：以技术书籍为核心的私有知识库问答系统，帮助开发者快速学习、定位知识、基于书籍内容进行工作。  
> **简历核心叙事**：PDF 解析 → 多书跨库检索 → 来源溯源 → 知识工作（总结/对比/辅助写作）→ Ragas 量化评测。

---

## 1. 项目概述

### 解决什么问题

技术书籍（如《深入理解MySQL核心技术》《高性能MySQL》）内容密集，难以快速定位知识点。本系统支持将书籍 PDF 和文档入库，通过自然语言问答与知识工作工具，帮助用户：

- 快速定位某个知识点在哪本书的哪一章
- 跨多本书对比同一主题的不同讲法
- 生成某章节的总结 / 思维导图大纲
- 基于书中知识辅助写作技术文档或方案

### 支持的文档类型（按优先级）

| 优先级 | 格式 | 典型场景 |
|--------|------|---------|
| ★★★ | PDF | 技术书籍（中英文） |
| ★★☆ | Markdown / TXT | 笔记、技术文档、README |
| ★★☆ | DOCX | 方案文档、规范 |
| ★☆☆ | HTML / 网页 | 框架官方文档（爬取） |

---

## 2. 核心功能与技术亮点

### 2.1 功能概览

| 功能 | 说明 | 优先级 |
|------|------|--------|
| **书籍问答** | 自然语言提问，回答附带来源（书名 + 章节 + 页码） | P1 |
| **多书跨库检索** | 同时检索多本书，融合结果，识别最相关来源 | P1 |
| **来源溯源** | 每段回答可追溯至原书章节，前端展示引用卡片 | P1 |
| **章节总结** | 对指定书的指定章节生成结构化摘要 | P2 |
| **跨书对比** | 同一主题在不同书中的讲法对比（如两书对索引的解释） | P2 |
| **辅助写作** | 基于书中知识生成技术文档、代码示例、方案建议 | P2 |
| **文档管理** | 上传 / 删除 / 查看已入库书籍和文档 | P1 |
| **Ragas 评测** | 量化评测检索质量，对比不同策略效果 | P3 |
| **框架文档爬取** | 抓取 Vue3 / LlamaIndex 等官方文档入库 | P3 |

### 2.2 技术亮点（面试叙事）

| 亮点 | 技术体现 |
|------|---------|
| **PDF 结构化解析** | 章节检测、代码块提取、表格保留，解决中文技术书排版噪声问题 |
| **多书跨库检索** | 并行查询多个向量库 Collection，结果融合重排序 |
| **来源溯源** | 节点元数据携带书名 / 章节 / 页码，前端引用卡片可跳转 |
| **知识工作 Agent** | 总结、对比、辅助写作通过 ReActAgent + 工具编排实现 |
| **Ragas 量化评测** | 忠实度、相关性、上下文精度多维度可视化对比 |

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Vue3 前端                                    │
│  ChatWindow │ SourceCard（引用卡片）│ WorkPanel（知识工作）│ EvalDash  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ HTTP / SSE（流式输出）
┌───────────────────────────▼──────────────────────────────────────────┐
│                        FastAPI 后端                                   │
│  /chat  │  /work（总结/对比/写作）  │  /docs（文档管理）  │  /eval    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                  RAG Engine（LlamaIndex）                             │
│                                                                      │
│   MultiSourceRetriever                                               │
│      ├── book_mysql_collection    → ChromaDB                         │
│      ├── book_redis_collection    → ChromaDB                         │
│      └── docs_vue3_collection     → ChromaDB                         │
│                   │                                                  │
│         结果融合 + 重排序 + 来源标注                                    │
│                   │                                                  │
│   ReActAgent（知识工作编排）                                           │
│      ├── summarize_tool           # 章节总结                          │
│      ├── compare_tool             # 跨书对比                          │
│      └── multi_source_rag_tool    # 跨库检索                          │
│                   │                                                  │
│          ZhipuAI GLM（生成 + 推理）                                   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                       文档入库层                                      │
│  PDF 解析（PyMuPDF）  │  MD/TXT/DOCX  │  HTML 爬虫（低优先级）         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. 目录结构

```
llmaLearn/
├── api/                              # FastAPI 后端
│   ├── main.py
│   ├── routers/
│   │   ├── chat.py                   # 问答接口（SSE 流式）
│   │   ├── work.py                   # 知识工作接口（总结/对比/写作）
│   │   ├── documents.py              # 文档上传 / 入库 / 管理
│   │   └── eval.py                   # 评测接口
│   └── schemas.py
│
├── core/
│   ├── rag/
│   │   ├── data_loader.py            # 扩展：支持 PDF/MD/TXT/DOCX
│   │   ├── indexer.py                # 扩展：多 Collection 命名管理
│   │   ├── pdf_parser.py             # 新增：PDF 结构化解析
│   │   └── multi_source_retriever.py # 新增：跨库并行检索
│   ├── agent/
│   │   └── agent.py                  # 扩展：注入知识工作工具
│   ├── tools/
│   │   ├── tools.py                  # 保留
│   │   ├── summarize_tool.py         # 新增：章节总结工具
│   │   └── compare_tool.py           # 新增：跨书对比工具
│   └── workflow/
│       └── multi_strategy_rag.py     # 保留
│
├── crawler/                          # 低优先级，P3 实现
│   ├── base_crawler.py
│   ├── vue_crawler.py
│   └── llamaindex_crawler.py
│
├── frontend/                         # Vue3 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.vue
│   │   │   ├── SourceCard.vue        # 引用卡片：书名 + 章节 + 页码
│   │   │   ├── WorkPanel.vue         # 知识工作面板（总结/对比/写作）
│   │   │   ├── DocManager.vue        # 书籍/文档管理
│   │   │   └── EvalDashboard.vue
│   │   └── App.vue
│   └── package.json
│
├── evals/                            # 保留并升级
├── configs/                          # 保留
├── data/
│   ├── books/                        # PDF 书籍存放
│   └── docs/                         # 其他格式文档
└── docs/                             # 方案文档
```

---

## 5. 关键技术设计

### 5.1 PDF 结构化解析

中文技术书解析的核心挑战是章节识别和代码块提取：

```python
# core/rag/pdf_parser.py
class BookPDFParser:
    """解析技术书籍 PDF，保留章节结构和代码块"""

    def parse(self, pdf_path: str, book_title: str) -> list[Document]:
        # 1. 用 PyMuPDF 提取文本 + 页码
        # 2. 基于字体大小 / 样式检测章节标题
        # 3. 识别代码块（等宽字体区域）
        # 4. 按章节分块（chunk_size 适配章节长度）
        # 5. 注入元数据
        metadata = {
            "book_title": book_title,    # 《深入理解MySQL核心技术》
            "chapter": "第3章 InnoDB存储引擎",
            "page": 87,
            "content_type": "text"       # text / code / table
        }
```

### 5.2 来源引用卡片

前端 `SourceCard.vue` 展示：

```
┌─────────────────────────────────────────────┐
│ 📚 《深入理解MySQL核心技术》                   │
│    第3章 · InnoDB存储引擎 · 第87页            │
│ "InnoDB 通过 MVCC 实现了..."                 │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│ 📚 《高性能MySQL》                            │
│    第5章 · 索引优化 · 第143页                 │
│ "覆盖索引可以避免回表查询..."                 │
└─────────────────────────────────────────────┘
```

### 5.3 知识工作 Agent 工具

```python
# 跨书对比示例
compare_tool = FunctionTool.from_defaults(
    fn=compare_across_books,
    name="compare_books",
    description="对比多本书对同一主题的不同讲解，输入主题和书目列表"
)

# 章节总结示例
summarize_tool = FunctionTool.from_defaults(
    fn=summarize_chapter,
    name="summarize_chapter",
    description="对指定书籍的指定章节生成结构化摘要，可选思维导图格式"
)
```

---

## 6. 开发路线图

### Phase 1 — MVP（3-5 天）
- [ ] PDF 解析：PyMuPDF 提取文本 + 页码元数据，入库单本书（先用《深入理解MySQL》测试）
- [ ] 基础问答：单书检索 + 来源引用（书名 + 章节 + 页码）
- [ ] FastAPI 基础路由 + Vue3 对话界面 + SourceCard 组件

### Phase 2 — 跨库检索 + 知识工作（4-6 天）
- [ ] 多书入库，MultiSourceRetriever 并行跨库检索
- [ ] 章节总结工具（summarize_tool）
- [ ] 跨书对比工具（compare_tool）
- [ ] 辅助写作：基于书籍知识生成技术内容
- [ ] 前端 WorkPanel（总结/对比/写作入口）
- [ ] 支持 Markdown / TXT / DOCX 格式上传

### Phase 3 — 评测 + 工程完善（3-4 天）
- [ ] Ragas 评测升级：多书场景测试集
- [ ] 前端 EvalDashboard：策略对比图表
- [ ] 文档管理页（已入库书籍列表 / 删除 / 重新入库）
- [ ] 框架文档爬虫（Vue3 / LlamaIndex，低优先级）

---

## 7. 现有代码复用策略

| 现有模块 | 处理方式 |
|---------|---------|
| `configs/llm.py` | 直接复用 |
| `configs/embedding.py` | 直接复用 |
| `core/rag/data_loader.py` | 扩展：新增 PDF 解析分支 |
| `core/rag/indexer.py` | 扩展：多 Collection 按书籍命名 |
| `core/agent/agent.py` | 扩展：注入 summarize / compare 工具 |
| `core/workflow/multi_strategy_rag.py` | 保留 |
| `evals/` | 升级：多书场景测试集 |

---

## 8. 技术栈总览

| 层 | 技术 |
|----|------|
| LLM | ZhipuAI GLM-4-flash |
| Embedding | BAAI/bge-small-zh-v1.5 |
| RAG 框架 | LlamaIndex |
| 向量库 | ChromaDB |
| PDF 解析 | PyMuPDF（fitz） |
| 后端 | FastAPI + uvicorn |
| 前端 | Vue3 + Vite + TypeScript |
| 评测 | Ragas |
| 爬虫（P3） | httpx + BeautifulSoup4 |

---

## 9. 简历描述参考

```
书籍知识库智能助手（个人项目）| Python · LlamaIndex · FastAPI · Vue3

- 实现 PDF 技术书籍结构化解析，保留章节 / 页码元数据，每条回答可溯源至原书具体页码
- 实现多书跨库并行检索与结果融合，支持同一主题在不同书中的讲法对比
- 基于 ReActAgent 编排知识工作工具，支持章节总结、跨书对比、辅助写技术文档
- 集成 Ragas 评测框架，量化对比不同检索策略的忠实度、相关性等多维指标
- 后端 FastAPI + 前端 Vue3 全栈实现，支持 PDF / Markdown / TXT / DOCX 多格式入库
```
