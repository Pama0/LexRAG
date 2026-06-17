# 评测测试集说明

## 文件

- `golden.seed.jsonl` — 金标准**种子**（示范格式；核心只需 `user_input`/`category`/`scope`，reference 占位可留空）。
- `golden.jsonl` — **完整金标准集**（30~50 条，需按本说明人工补全/校验后生成，用于决策对比评测）。
- `testset.draft.jsonl` / `testset.jsonl` — ragas TestsetGenerator 自动生成（Phase 2 扩样用）。

## 字段

| 字段 | 必填? | 含义 |
|------|:---:|------|
| `user_input` | ✅ | 用户问题（原话，可含口语/措辞变体） |
| `category` | ✅ | **金标准应走的类**（分类准确率以此为准，量化决策的核心） |
| `scope` | ✅ | book_titles 列表；`null` = 全库 |
| `reference` | 可选 | 参考答案，**仅** context_precision/recall + factual_correctness 用。ragas 做**语义/事实比对**、要点对即可、不必完美。可复用 testset.draft 自动生成的。 |
| ~~`reference_contexts`~~ | ❌ 不填 | 指标用 SUT **实际检索的** `retrieved_contexts`，**不读**此字段。 |

> **最小可用集**：每行只填 `user_input` / `category` / `scope` 三个键，就能跑出**分类准确率 + faithfulness + answer_relevancy** 的对比表——量化「决策提升」的核心诉求**不需要标准答案**。reference 是想补 context_recall/factual_correctness 时的可选增强。

## category 标注准则（边界约定）

| 类 | 判据 |
|----|------|
| `retrievable` | 单一概念、单轮检索可集中命中。含「X是什么 / 讲明白X」**当 X 是单一概念**（即便 X 是冷门专名）。 |
| `pending_split` | X 是大主题、答案需罗列并列子项 / 横跨多章节（如「讲讲MySQL」「A和B的区别」）。 |
| `ambiguous` | 主题具体、能命中，但缺评判维度/角度（如「Redis做缓存好吗」）。 |
| `missing_info` | 召回到相关主题、但**缺关键限定/指代不明**，补充后才能精确检索（如「这个索引的应用场景」）。→ 反问澄清。 |
| `out_of_scope` | 问题清晰，但**探测召回片段与主题明显不相关**（库里没有该主题，如 PostgreSQL/MongoDB）。→ 如实告知「库里没有」，不反问、不硬答。 |
| `other` | 召回到内容、但需跨主题综合 / 多步推理 / 开放权衡。 |

## 必须覆盖

- **每类 ≥3 条**，难度分层（简单/中/难）。
- **「库里有但易误判」case**（如 `给我讲明白openclaw` 应判 retrievable，曾被误判 other）——这是量化 probe-then-classify 提升的关键样本。
- 措辞变体（口语、错别字、长短句）覆盖 router 净化能力。

## 标注流程

1. **最小档**：写 `user_input` + 标 `category`（按上表准则）+ `scope`。直接手写，或从 `testset.draft.jsonl` 挑条加 category。
2. **可选增强**：要 context_recall/factual_correctness 才填 `reference`——复用 testset.draft **自动生成的**，或基于真实检索片段让强 LLM 生成 + 人工抽检。`reference_contexts` 不填。
3. 边界 case 按上表准则裁定（核心看「召回散落与否」，非句式），存疑的记进本文件备注。
4. 汇总成 `golden.jsonl`，跑 `python -m eval.compare --testset eval/dataset/golden.jsonl ...`。
