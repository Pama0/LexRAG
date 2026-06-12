# 评测系统：决策对比（ablation）框架 Design

**日期:** 2026-06-12
**状态:** 设计已定，待实现
**关联:** [book-rag-eval-harness](2026-06-08-book-rag-eval-harness-design.md) · [probe-then-classify](2026-06-12-probe-then-classify-design.md) · [项目架构](../../ARCHITECTURE.md)

## 1. 目标

让"工作流优化/迭代"可量化证明：固定测试集，参数化被测系统，跑多个变体（决策前 vs 决策后），同指标打分，输出**对比表格 + delta**，证明"某个决策带来 X 提升"。

## 2. 现状诊断（eval/）

**复用（好）**：基于 ragas，5 指标覆盖诉求——`context_recall`(召回)、`factual_correctness`(准确 F1)、`faithfulness`+`answer_relevancy`(答案分)；评测侧 DeepSeek（`make_eval_llm`，与被测解耦）；ragas `experiment` runner + `aggregate`。

**两个硬缺口**：
1. **SUT 包死系统**：`BookRagWorkflowSystem` 包退役的 `BookRagWorkflow`（语法错、被 `DocQueryWorkflow` 取代）——评测跑的不是当前系统。
2. **无对比/ablation 框架**：`run_eval` 一次一配置一 JSON，无法做"决策前 vs 后"对比。

**测试集**：ragas persona 生成，字段 `{user_input, reference_contexts, reference, persona_*}`，**无 category 标注**、难度未分层。

## 3. 设计：ablation harness（三层）

### Layer 1 — 测试集（金标准小集打底）
- **决策（已定）**：先**手工金标准小集（30~50 条）**，每个 category（retrievable/pending_split/ambiguous/missing_info/other）覆盖 + 难度分层 + 含易误判 case（如 `给我讲明白openclaw`）；自动生成（TestsetGenerator）留作后续扩样。
- **格式扩展**：现有字段 + `"category"`（金标准应走的类）+ 可选 `"difficulty"`、`"scope"`（book_titles）。
- 理由：决策级指标（分类准确率）最依赖准标注，手工小集质量保证、快速可用、立即量化 probe 修复。

### Layer 2 — 可配置 SUT（核心）
- 新建 `DocQueryWorkflowSystem` 包当前 `DocQueryWorkflow`，实现 `RagSystem`。
- **决策开关**（DocQueryWorkflow 加 feature flags）：`probe_then_classify` / `split_enabled` / `assume_enabled` / `other_agent_enabled`（off → 该类降级单轮 `retrieve`）。baseline=全 off（全单轮）；variant=逐个打开。
- **SUT 暴露 category/outcome**：评测要算分类准确率 + 分支分布，故 `DocQueryWorkflow` 在 `finalize` 把 `category`/`intent` 附到结果（`Response.metadata`）；SUT 读出填进 `RagOutput`（新增 `category` 字段）。

### Layer 3 — 对比 runner + delta 表格
- runner 接收**变体列表**，对同一测试集各跑一遍，输出 Markdown 对比表：每行一个变体，列含 ragas 5 指标 + **分类准确率** + 分支分布，并标 delta。

| 配置 | context_recall | factual_correctness | faithfulness | 分类准确率 | other误判率 |
|---|---|---|---|---|---|
| baseline(全单轮) | 0.62 | 0.55 | 0.71 | — | — |
| +probe-then-classify | 0.78 (+0.16) | 0.61 (+0.06) | 0.74 | 0.91 (+0.30) | 5% (−25pt) |

## 4. 关键新增指标（ragas 没有，最对诉求）

ragas 5 指标是端到端答案质量；你的优化在分类/分支层，故补**决策级指标**（纯代码、确定性、不耗评测 LLM）：
- **分类准确率**：SUT 实际 `category` vs 测试集标注 `category`。直接量化 probe-then-classify / 判定调优。
- **分支分布迁移**：各 outcome/category 触发比例（如 other 误判率 30%→5%）——openclaw bug 修复的量化证据。

确定性指标优先用于讲故事（不受评测 LLM 随机性影响）。

## 5. 方法学（结论可信）

- **配对对比**：同一 query 在 A/B 的 delta（比独立均值更敏感、可归因）。
- **分层报告**：按 category 分组（"pending_split 类开 split 后 recall +0.20"比全局均值有说服力）。
- **控随机性**：评测 LLM 固定低温/多跑取均值；确定性指标不受影响。

## 6. 关键技术点 / 风险

- **category 暴露**：`DocQueryWorkflow.finalize` 把 `category`/`intent` 放进 `Response.metadata`，SUT 读出——小改动，需保证不影响 api/前端（metadata 是附加字段）。
- **决策 flag 的"降级"语义**：flag off 时分支走单轮 `retrieve`（复用现有降级路径），保证 baseline 可跑、对比公平。
- **成本**：每变体跑全测试集 × ragas LLM 打分，N 变体 = N×成本。金标准小集（~30条）控成本；大集按需。
- **测试集标注主观性**：金标准 category 由人工标，边界 case（retrievable vs pending_split）需约定标注准则，写进测试集 README。

## 7. 落地分阶段

- **Phase 1（本期）**：换 SUT 接当前系统 + 决策 flag + category 暴露 + 分类准确率指标 + 金标准种子集 + 对比 runner（delta 表格）→ 能跑出"baseline vs +probe"对比表。
- **Phase 2（后续）**：TestsetGenerator 自动扩样（带 category 校验）、更多变体矩阵、按 category 分层报告、CI 化回归。

## 8. 不做（YAGNI）

- 不重写 ragas 指标（5 个够用，直接复用）。
- 不做自动测试集大规模生成（Phase 2）。
- 不动评测侧模型解耦（沿用现有 make_eval_llm）。
