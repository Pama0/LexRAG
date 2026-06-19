# LibraryRAG — 技术书籍知识库助手
> 上传技术书籍 PDF，做可溯源的 RAG 问答；并配套一套量化评测体系，用数据驱动系统迭代。

## LibraryRAG

LibraryRAG 是一个技术书籍 / 文档的 AI 知识库助手：把书籍 PDF 解析入库，基于检索增强生成（RAG）回答书里的具体问题。当前已实现文档问答（QA），并按"从受限检索到开放推理"的能力光谱规划了学习计划、人生/决策支持等后续能力。

## 为什么使用 LibraryRAG 
LibraryRAG，通过为各种倾向的问题预设深度优化的工作流，使得能稳定地输出答案，并降低LLM的概率性。

对比其他Agentic RAG系统，LibraryRAG做了严格的防幻觉并将大模型知识与rag数据源隔离开，令LibraryRAG可以忠于用户提供的，甚至是与大模型知识，外部流转的相悖的技术文档，知识来回答问题，而无需担忧多版本等干扰因素

## 架构概览

顶层是一个编排器：**IntentRouter**（净化 query：规范化 + 指代消解 → 意图分类）→ 分发到对应**能力（capability）**。QA 能力内部再做：探测检索（probe）→ 难度分类 → 按类别走不同分支（单轮检索 / 拆解汇总 / 归纳维度 / 反问澄清 / 高难度 agent）。

核心原则：**按任务可预测性给每个能力配 workflow 或 agent，二者是可组合的积木，不是二选一**——高可预测（步骤已知）用 workflow 求确定性与可观测；低可预测（路径需模型自决）用 agent。

```
Layer 0  检索 & 记忆服务（横切，注入各层）：Chroma 向量库 + 可插拔 Retriever/Reranker
Layer 1  IntentRouter：净化 + 意图分类 + 确定性分发
Layer 2  能力层：QA(workflow) / StudyPlan(workflow，规划中) / LifePlan(agent，规划中)
```

> 完整产品愿景、分层依据与落地路径见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 质量保障

LibraryRAG 配套一套量化评测体系：用 **ragas 5 指标 + 自定义分类准确率**对"决策路由 RAG"做 ablation 对比，能从评测数据反推系统缺陷、定位根因、修复并再验证（一个真实例子：评测发现库外问题被误判，由此引入 `out_of_scope` 分类）。

- 评测**驱动迭代**的过程（ablation 对比表 + case study + 诚实标注）见 [docs/EVAL_ITERATION.md](docs/EVAL_ITERATION.md)。
- 评测**体系全景**（被测什么、指标字段、数据集生成、脚本地图、怎么跑）见 [docs/EVAL_OVERVIEW.md](docs/EVAL_OVERVIEW.md)。

## 快速开始

环境：Python 3.12+，虚拟环境 `.venv`。**所有命令从项目根目录运行**（模块导入要求）。

```bash
# 激活虚拟环境（PowerShell）
.venv\Scripts\activate
pip install -r requirements.txt
# .env 配置 DEEPSEEK_API_KEY（主系统与评测 judge 共用）

python main.py                                   # CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务（前端对接）
```

> 跑评测的命令见 [docs/EVAL_ITERATION.md](docs/EVAL_ITERATION.md#怎么复跑)。

## 技术栈

- **LlamaIndex** — workflow 编排与 RAG 基础设施
- **Chroma** — 向量数据库
- **DeepSeek** — 主 LLM（`OpenAILike` 接入，已关 thinking）
- **ragas** — 评测指标
- **rank-bm25 + jieba** — hybrid 检索的稀疏侧（中文分词、RRF 融合）
- **bge-reranker** — 交叉编码器重排（可插拔）
- **FastAPI** — Web 服务

## 路线图

能力从"受限检索"到"开放推理"是一条光谱：

1. **RAG 问答** —— 基于已入库书籍回答具体问题（✅ 已实现）
2. **学习计划** —— 按一本书的结构生成结构化学习计划（workflow，规划中）
3. **人生 / 决策支持** —— 融合书的观点 + 用户长期记忆做长期规划，**建议显式锚定到书的依据、可溯源**（agent，规划中）
4. **进度 / 复盘** —— 读取已生成的计划产物与记忆，更新回顾进度（规划中）

## 项目结构

```
api/        Web 服务（FastAPI）
core/       领域逻辑：agent / workflow / retrieval / rag
configs/    LLM / embedding 等配置
eval/       评测体系：harness（runner/指标/SUT）+ datagen + dataset + results
legacy/     冻结的早期法律条文 RAG（不保证可运行）
```

依赖方向单向：`api/` → `core/` → `configs/`，由 `python scripts/check_layering.py` 守卫。
