# LibraryRAG — 技术书籍知识库助手

> 上传技术书籍 PDF，做**可溯源、防幻觉**的 RAG 问答；并配套一套量化评测体系，用数据驱动系统迭代。

LibraryRAG 是一个面向技术书籍 / 文档的 AI 知识库助手：把书籍 PDF 解析入库，基于检索增强生成（RAG）回答书里的具体问题。当前已落地文档问答（QA），并按"从受限检索到开放推理"的能力光谱规划了学习计划、决策支持等后续能力。

---

## ✨ 特性

- **知识隔离、忠于原书** —— 严格防幻觉，把大模型自身知识与 RAG 数据源隔离开。答案只依据你上传的文档，即便文档与大模型记忆、外部流转的版本相悖，也忠于你提供的资料，不被多版本等因素干扰。
- **按问题"形状"分流的决策路由** —— 不同问题走不同工作流：单概念讲解、对比选型、多实体罗列、多跳推理各有分支，把高可预测的任务交给确定性 workflow，低可预测的交给有界 agent，稳定输出、降低概率性。
- **多子问题并发处理** —— 一句话里夹带多个意图会被拆成互不相同的子问题，逐个独立判定可答性与难度，再按序流式合成答案。
- **库外问题不硬答** —— 问到库里没有的主题时，明确告知"超出范围"而非编造，可答性闸（Admitter）按召回片段与问题主体的相关性判定。
- **评测驱动迭代** —— 配套 ragas 指标 + 成本度量的对比框架，能从评测数据反推系统缺陷、定位根因、修复并再验证。
- **流式 Web 体验** —— FastAPI + SSE 后端，Vue 前端，检索进度与答案增量实时推送。

---

## 🏗️ 架构概览

顶层是一个编排器 **`DocQueryWorkflow`**，自身不持检索/合成实质，只编排 + 薄委托。一次问答的主干：

```
start → 净化(clean) → 拆子问题(split) → 逐子问题路由(route) ─┬─ QA 分支    → finalize
                                                          ├─ 直答/婉拒  → finalize
                                                          └─ 学习计划   → finalize
```

**门口 `FrontDoor`（三步，各一次独立 LLM 调用、各自降级）**

1. `clean`：读会话记忆消指代 + 规范化 → `clean_query`；指代无法消解则直接反问澄清。
2. `split`：把 `clean_query` 拆成若干互不相同的子问题（不拆时即原句一项）。
3. `route`：给每个子问题判一个出口 —— 进 QA，还是闲聊/婉拒。

**QA 能力 `QaCapability`（决策路由的实质）**

每个 QA 子问题先**并行判定**，再**按序流式执行**：

| 阶段 | 组件 | 职责 |
|---|---|---|
| 探测 | probe | 用子问题在库里做一次探测检索，拿到召回证据 |
| 可答性闸 | `Admitter` | 判 `ok` / `missing_info`（反问）/ `out_of_scope`（库外告知），并算出该子问题的检索 scope |
| 难度分类 | `QueryClassifier` | 对 `ok` 子问题判四类**答案形状**：`explain`（讲透概念）/ `compare`（比较选型）/ `simple`（单轮命中）/ `complex`（多跳/发散，交有界 agent） |
| 执行 | `_execute_subq` | 按类别检索 + 流式合成；多子问题各自分节，单子问题退化为单路径 |

**核心原则**：按任务可预测性给每个能力配 workflow 或 agent —— 二者是可组合的积木，不是二选一。步骤已知用 workflow 求确定性与可观测，路径需模型自决用 agent。

**两层记忆纪律**：会话记忆只存「用户原话 + 最终答案」（供门口消指代）；本轮的改写 query、子问题、中间产物只走 workflow `Context`，绝不写进会话记忆，避免污染下一轮指代消解。

```
Layer 0  检索 & 记忆服务（横切，注入各层）：Chroma 向量库 + 可插拔 Retriever/Reranker
Layer 1  FrontDoor：净化 → 拆子问题 → 逐子问题路由
Layer 2  能力层：QA(workflow，已落地) / StudyPlan(workflow，规划中) / LifePlan(agent，规划中)
```

> 完整产品愿景与分层依据见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)；QA workflow 实现细节见 [core/workflow/README.md](core/workflow/README.md)。

---

## 📊 质量保障：评测驱动

LibraryRAG 配套一套量化评测体系，对**两条问答路线**做同台对比，用数据反推缺陷：

- **workflow** —— `DocQueryWorkflow`，显式决策路由（门口净化/拆分/路由 + 难度分类）。
- **agent** —— 有界 `AutoAgent`，绕过决策路由，让模型自主多轮规划检索。

对比维度：

- **答案质量** —— ragas 5 指标：`faithfulness` / `answer_relevancy` / `context_precision` / `context_recall` / `factual_correctness`（claim 级判定，与召回 chunk 条数无关）。
- **成本** —— 时延（s/条）与 token（条均，客户端计数、只数被测系统、排除评测 judge）。回答「显式决策路由 vs 让 agent 自己规划」到底谁强、谁更省。

被测系统用 DeepSeek，评测 judge 侧独立配置，两套 LLM 刻意解耦、互不污染。一个真实例子：评测曾发现库外问题（PostgreSQL/MongoDB 等）被误判为可答，由此引入独立的 `out_of_scope` 分类。

```powershell
# 冒烟（前 2 条，确认链路通）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 2

# 全量两路线对比，落盘
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --out docs/compare.md --detail docs/compare_detail.csv
```

> 评测体系全景（被测什么、指标字段、数据集生成、脚本地图）见 [docs/EVAL_OVERVIEW.md](docs/EVAL_OVERVIEW.md)；评测驱动迭代的案例见 [docs/EVAL_ITERATION.md](docs/EVAL_ITERATION.md)。

---

## 🚀 快速开始

环境：Python 3.12+，虚拟环境 `.venv`。**所有命令从项目根目录运行**（模块导入要求）。

```powershell
# 1. 后端
.venv\Scripts\activate          # PowerShell；Git Bash 用 source .venv/Scripts/activate
pip install -e ".[dev]"          # 运行依赖 + 测试/lint（pytest、ruff）；仅运行用 pip install -e .
# 在 .env 配置 DEEPSEEK_API_KEY（主系统与评测 judge 共用）

python main.py                                   # CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务（前端对接，SSE 流式）

# 2. 前端（可选）
cd frontend
npm install
npm run dev
```

---

## 🧰 技术栈

| 领域 | 选型 |
|---|---|
| 编排 & RAG 基础设施 | **LlamaIndex**（Workflow 编排） |
| 向量数据库 | **Chroma** |
| 主 LLM | **DeepSeek**（`OpenAILike` 接入，已关 thinking） |
| 检索（可插拔） | `vector` 向量 / `hybrid` dense+BM25（**rank-bm25 + jieba** 中文分词、RRF 融合） |
| 重排（可插拔） | **bge-reranker** 交叉编码器 |
| 评测指标 | **ragas** 5 指标 + 时延/token 成本度量 |
| Web | **FastAPI**（SSE 流式） + **Vue 3 / Vite** 前端 |

---

## 🗺️ 路线图

能力从"受限检索"到"开放推理"是一条光谱：

1. **RAG 问答** —— 基于已入库书籍回答具体问题。（✅ 已实现）
2. **学习计划** —— 按一本书的结构生成结构化学习计划。（workflow，规划中）
3. **决策 / 人生支持** —— 融合书的观点 + 用户长期记忆做长期规划，建议**显式锚定到书的依据、可溯源**。（agent，规划中）
4. **进度 / 复盘** —— 读取已生成的计划产物与记忆，更新回顾进度。（规划中）

---

## 📁 项目结构

```
api/        Web 服务（FastAPI，SSE 流式）
core/       领域逻辑：workflow（门口+QA能力）/ agent / retrieval / rag / persistence
configs/    LLM / embedding 等配置
eval/       评测体系：harness（compare/指标/SUT）+ datagen + dataset + results
frontend/   Vue 3 + Vite 前端
docs/       架构、评测与设计文档
```

依赖方向单向：`api/`(Web) → `core/`(领域) → `configs/`，由 `python scripts/check_layering.py` 守卫。
