# 检索层评测设计（Recall@k / nDCG@k，vector vs hybrid）

**日期**：2026-06-23
**状态**：设计已确认，待写实现计划

## 背景与动机

现有 `eval/`（`compare.py`）是端到端评测：用 ragas + LLM 裁判给最终答案打分（faithfulness、context_recall 等）。它度量的是整条链路，**慢、贵、有方差**，且无法隔离检索质量——改 `core/retrieval/retrieve.py` 的 RRF 参数、停用词表、`top_k` 时，端到端分数的波动分不清是检索变了还是 LLM 抽风。

混合检索（`HybridRetriever` = dense + BM25 + RRF）目前是**凭直觉合入的**，没有数据证明它比 `VectorRetriever` 基线好多少，也没有调参的尺子。

**本设计补一个检索层评测**：确定性、零生成/裁判 LLM、可秒级反复跑，用来

1. 做 ablation：vector vs hybrid 的 Recall@k / nDCG@k 对照；
2. 当调参尺子：验证停用词表、`top_k`、未来的加权 RRF 是涨点还是掉点。

## 关键决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 相关 chunk 标注来源 | **Pooling + LLM 离线判定** | golden 现无 chunk 级标注；ragas `reference_contexts` 因 query 已重新策展/改写而对不齐。TREC pooling 法：候选池 + LLM 逐个判，冻结成文件。一次性成本，之后复用。 |
| 相关性粒度 | **二元 相关/不相关** | LLM 二元判定稳、人工抽检快；够算 Recall/Precision/MRR/nDCG（二元增益）。分级 0/1/2 方差大、阈值难定，学习项目不划算。 |
| 评测入口 | **绕过 front_door** | 直接喂 golden `user_input` 给 retriever，不走 clean/split。隔离变量——测的就是 `retrieve.py`，不被 LLM 改写的随机性污染，保证评测确定可复现。 |
| 标注/评测分离 | **两阶段** | 标注（慢、调 LLM、一次性）产出冻结文件；评测（快、零 LLM）反复读它。改 `retrieve.py` 后只重跑评测。 |

## 架构

```
eval/datagen/label_retrieval.py     # 阶段一:离线 pooling + LLM 判定 → 产标注(调 LLM,一次性)
eval/dataset/golden.retrieval.jsonl # 冻结标注:query → 相关 chunk_id 集合
eval/harness/retrieval_metrics.py   # 纯函数:recall@k / precision@k / mrr / ndcg@k
eval/harness/retrieval_eval.py      # 阶段二:跑 retriever → 算指标 → 出表(零 LLM,反复跑)
tests/test_retrieval_metrics.py     # 指标纯函数单测
```

依赖方向沿用 `eval/` 现状：评测侧脚本从项目根以 `python -m eval.harness.xxx` 运行，复用 `core.retrieval.make_retriever`、`eval.config`、`RAGIndexManager`。

## 组件

### 1. `label_retrieval.py`（阶段一，调 LLM，一次性）

**职责**：把 golden query 标注成「相关 chunk_id 集合」，冻结到 `golden.retrieval.jsonl`。

- **入口**：读 `eval/dataset/golden.jsonl`，取每条 `user_input`，**不走 front_door**。
- **只标可答类**：`retrievable / pending_split / other / ambiguous` 才标；`missing_info / out_of_scope`（`reference=""`）跳过——它们本无相关 chunk，进 Recall 分母无意义。
- **候选池**：每个 query 取 `vector top-30 ∪ bm25 top-30`，按 `chunk_id` 去重。两路都走 `make_retriever`，保证候选池与被测系统同源。pool 深度 `POOL_N = 30`（常量，便于调）。
- **LLM 判定**：候选 chunk 正文（截断到合理长度，如 600 字）+ query 喂 DeepSeek，沿用 `fill_reference.py` 的 plain `AsyncOpenAI` client + `extra_body={"thinking": {"type": "disabled"}}`，`response_format` JSON。**分批**判，每批约 10 个 chunk，输出 `{chunk_id: 0|1}`。批大小 `JUDGE_BATCH = 10`（常量）。
- **产物** `golden.retrieval.jsonl`，每行：
  ```json
  {"user_input": "...", "category": "retrievable", "relevant_chunk_ids": ["id1","id2"], "skipped": false}
  ```
  - 跳过类 / 相关集为空的 query 也落盘，`skipped: true`，**不计入指标**（评测阶段过滤）。
- **收尾**：打印每条 pooling 命中数与零命中 query 列表，提醒人工抽检（pooling 法标配——LLM 判定可能漏标，需人工校）。
- **安全**：写到 `golden.retrieval.jsonl`，人工抽检后即用；与 `fill_reference.py` 的「先写 .with_ref 再人工替换」风格一致，这里直接产最终文件名但显式提示抽检。

### 2. `retrieval_metrics.py`（纯函数，零依赖）

**职责**：给定一条 query 的检索结果与标注，算各 k 上的指标。纯函数，对齐 `metrics.py` 「映射可离线单测」的现有风格。

- 输入：`retrieved_ids: list[str]`（按检索序）、`relevant_ids: set[str]`、`k`。
- `recall_at_k(retrieved, relevant, k)` = `|命中∩relevant| / |relevant|`
- `precision_at_k(retrieved, relevant, k)` = `|命中∩relevant| / k`
- `mrr(retrieved, relevant)` = 第一个相关命中的 `1/rank`（无命中→0）
- `ndcg_at_k(retrieved, relevant, k)` = 二元增益 DCG / IDCG
- `K_VALUES = (1, 3, 5, 10)`（常量）。
- 边界：`relevant` 为空时该 query 不应进来（评测层已过滤）；防御性返回时约定明确（如 recall 分母为 0 视为跳过，不返回 NaN 污染均值）。

### 3. `retrieval_eval.py`（阶段二，零 LLM，反复跑）

**职责**：跑被测 retriever，算并聚合指标，出对比表。

- 加载 `golden.retrieval.jsonl`，**过滤掉 `skipped: true`**。
- 对每个被测 retriever（默认 `["vector", "hybrid"]`，CLI `--retrievers` 可加）：逐 query 调 `make_retriever(name).retrieve(query, index_manager=..., book_titles=None, top_k=max(K_VALUES))`，**绕过 front_door**，取有序 `chunk_id` 列表（`node.node_id`）。
- 每条算各 k 指标 → **聚合均值**；同时**按 `category` 分组**出明细（看 hybrid 在哪类问题上赢/输）。
- 输出：console 对比表 + `eval/results/retrieval_eval.csv`（对齐 `compare.py`/`report.py` 的 CSV 风格）。
- 命令：`python -m eval.harness.retrieval_eval --retrievers vector hybrid`

### 4. `test_retrieval_metrics.py`

指标纯函数单测：手造 `retrieved/relevant` 小例子，断言 recall/precision/mrr/ndcg 已知值（含全命中、零命中、部分命中、k 截断、相关集大于 k 等边界）。

## 数据流

```
[阶段一·一次性·调LLM]
golden.jsonl ──可答类query──▶ make_retriever(vector/bm25).retrieve
                                   │ top-30 ∪ top-30 去重
                                   ▼
                              候选 chunk 池 ──分批──▶ DeepSeek 判 0/1
                                                          │
                                                          ▼
                                          golden.retrieval.jsonl(冻结)

[阶段二·反复跑·零LLM]
golden.retrieval.jsonl ──过滤skipped──▶ for retriever in [vector, hybrid]:
                                            retrieve top-10(绕过front_door)
                                            │ 有序 chunk_id
                                            ▼
                                       retrieval_metrics(纯函数)
                                            │
                                            ▼
                                   均值 + 按category明细 ─▶ console 表 + results/retrieval_eval.csv
```

## 错误处理

- **标注阶段**：单 query 判定失败（LLM 异常 / JSON 解析失败）→ 记 warning，该 query 标 `skipped: true` 落盘，不中断整体（对齐 `compare.py` 单条异常记 error 不中断的风格）。
- **评测阶段**：单 retriever 单 query 检索异常 → 记 warning，该条指标记缺失，不计入该 retriever 均值，不中断。
- **空标注文件 / 全 skipped** → 明确报错提示先跑 `label_retrieval.py`。

## 测试策略

- `retrieval_metrics.py` 纯函数 → `test_retrieval_metrics.py` 离线单测（无需 LLM / chroma）。
- `label_retrieval.py` / `retrieval_eval.py` 涉及真实 chroma + LLM，属集成 smoke，依赖 `.env` 与已入库的 `chroma_db`，不进 CI 单测，靠人工运行 + 抽检。

## 非目标（YAGNI）

- 不做分级相关性 / graded nDCG。
- 不做加权 RRF（本评测是为它**铺尺子**，调参是后续工作）。
- 不把检索层评测并进 `compare.py`（两者入口、是否调 LLM、运行频率都不同，强行合并反而耦合）。
- 不做检索前的关键词提取 / 同义词扩写（前序讨论已结论：IDF 已覆盖大半，且应在有此尺子后才评估）。

## 后续（本设计之外）

有了这把尺子后可依次：① 给 `bm25_tokenize` 加停用词过滤，用本评测验证涨点；② 扫 `top_k`；③ 评估加权 RRF。
