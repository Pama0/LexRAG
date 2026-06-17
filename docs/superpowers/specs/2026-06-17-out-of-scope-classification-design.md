# 新增 `out_of_scope` 分类：库外问题与信息不足解耦

**日期**：2026-06-17
**分支**：`feat/out-of-scope-classification`

## 背景与问题

评测中发现 golden 第 20 条「PostgreSQL 的 MVCC 是怎么实现的？」被标为 `missing_info`，但该问题表述清晰完整、不缺任何限定——它被归入 `missing_info` 的唯一理由是「知识库里没有 PostgreSQL 内容」。

根因是分类体系把两种正交的性质揉成了一类。`query_preprocess.py` 的 `missing_info` 判据是一个 AND 复合条件：

> `missing_info`（信息不足）：缺了检索必需的关键限定，根本无法检索；**且探测召回为空、或片段明显与问题无关**（知识库里确实没有相关内容）。

而 `missing_info` 的行为是**反问澄清**（`doc_workflow.py` → `ClarifyEvent` → `clarify_branch`）。对一个库外问题反问「请补充什么？」是答非所问。

### 为什么「召回为空」是死信号

probe 走 `VectorRetriever.retrieve`（`retrieve.py:78`）：`as_retriever(similarity_top_k=5).aretrieve(query)`，纯向量 ANN 检索、**无 similarity_cutoff**。向量 ANN 只返回最近邻、没有相关性下限——只要库里有 ≥5 个 chunk，就返回 5 个，哪怕全不相关。hybrid（dense+BM25 经 RRF）同理。

因此 `_format_probe` 里 `if not nodes: ...`（`qa_capability.py:104`）基本是死代码。库外问题（如 PostgreSQL MVCC）的真实表现**不是召回为空，而是召回了 5 段最近邻、但语义全不相关**（大概率是 MySQL 锁/索引）。judge 看到「共命中 5 段」就判 `retrievable` 硬答——这正是已知的「库外问题误判 retrievable」bug 的机制根因。

## 目标

新增 `out_of_scope` 分类，把「库外问题」从 `missing_info` 独立出来：

- `out_of_scope`：信息可充分或不足，但**探测召回片段与问题主题明显不相关（库里没有该主题）** → 如实告知「知识库里暂无相关内容」，**不检索、不合成、不反问**。
- `missing_info` 收窄回本义：**召回到了相关主题、但缺关键限定/指代不明，补充后可精确检索** → 反问澄清。

判据从「召回为空」改为「**召回了但片段与问题主题明显不相关**」——由 judge LLM 基于 probe 召回片段做语义相关性判断（方案 A），对 retriever 零侵入。

## 边界裁决：out_of_scope 优先

两个维度正交，存在「既信息不足又主题库外」的格子（如用户只问一句「mongodb」）：

| | 库里有 | 库里没有 |
|---|---|---|
| **信息充分** | retrievable / 其他 | out_of_scope |
| **信息不足** | missing_info（反问） | **out_of_scope** |

**裁决：out_of_scope 优先。** 论据：对库外主题反问是无效的——「mongodb」→反问「想问什么？」→用户答「分片」→库里照样没有。白白多一轮。

因此判据锚定在「召回相关性」单一轴上，信息完整度退为次要：

- 召回**不相关**（库里没这主题）→ `out_of_scope`（含「mongodb」这种又短又库外的）
- 召回**相关** + 缺限定/指代不明 → `missing_info`（反问，因为补充后确实能命中）
- 召回相关 + 完整 → retrievable / 其他

这恰好让 `missing_info` 收窄成它本来的样子：**库里有这主题、只是问法缺限定**——与 golden 第 16–19 行（指代历史对话里的库内对象）完全吻合。

## 改动范围

### 1. 分类 schema + prompt（`core/workflow/query_preprocess.py`）

- `QueryJudgment.category` 的 `Literal` 增加 `"out_of_scope"`。
- prompt：
  - 把 `missing_info` 判据里「且探测召回为空/片段无关（库里确实没有）」那半句**剥离**，`missing_info` 回归纯「召回到相关主题、但缺关键限定/指代不明，补充后可精确检索 → 反问」。
  - 新增 `out_of_scope` 定义：「探测召回片段与问题主题明显不相关（库里没有该主题）→ out_of_scope，**无论问题是否完整**；因为反问也补不出库里没有的内容。返回 `{"category":"out_of_scope","rewritten_query":"...","reason":"库外，如 PostgreSQL/MongoDB 不在本库主题范围"}`。」
  - 更新分类优先级段：**先判召回相不相关（out_of_scope）**，再在「召回相关」里判是否缺限定（missing_info）、角度不定（ambiguous）、需拆分（pending_split），否则 retrievable/other。
  - probe 关闭时无召回信号 → 沿用现有兜底降级 `retrievable`（ablation 里这恰好体现 probe 的贡献）。
- 解析失败仍降级 `retrievable`（不变）。

### 2. workflow 分支（`core/workflow/doc_workflow.py`）

- `preprocess` 的 `match` 增加 `case "out_of_scope": return OutOfScopeEvent(...)`。
- 新增 `OutOfScopeEvent` 与 `out_of_scope_branch`：直接返回固定话术 `FinalizeEvent(answer="知识库里暂无与该问题相关的内容。", source_nodes=[])`，**不检索、不合成、不反问**。结构类比现有 `chitchat_branch`。
- 不加 out_of_scope 专属 ablation flag——作为基础能力常开（识别它依赖 probe，baseline 关 probe 时这类落 retrievable，由 ablation 自然体现）。

### 3. 数据集

- `eval/dataset/golden.jsonl` 第 20–23 共 4 条（PostgreSQL MVCC / MongoDB 分片 / Oracle RAC / Cassandra 一致性级别）：`missing_info` → `out_of_scope`。第 16–19 行（指代不明）保持 `missing_info`。
- `eval/dataset/README.md` 第 28 行 category 准则表：拆成 `missing_info`（信息不足→反问）+ `out_of_scope`（库外→如实告知）两行。

### 4. 评测侧

无需改代码——`run_eval.aggregate` 用 `category`/`expected_category` 字符串比对，新枚举值自动纳入分类准确率统计与 category 分布。

## 话术

`out_of_scope` 命中固定回：**「知识库里暂无与该问题相关的内容。」** 不动态列库内书名（书名是会变的元数据，且有中文编码/名称清理问题）。

## 测试（TDD）

- `tests/test_query_preprocess`（或现有等价文件）：mock LLM 返回 `out_of_scope` → `PreprocessResult.category == "out_of_scope"`；`QueryJudgment` schema 校验接受新枚举值。
- `tests/test_doc_workflow.py`：`preprocess` 产 `out_of_scope` → 走 `out_of_scope_branch` → 答案为固定话术、`source_nodes` 为空、**不调用** `qa.retrieve`。
- 现有 `missing_info` 测试不破。

## 验证

- 跑单测全绿。
- `python -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit <N>` smoke。
- 人工 smoke（probe 开）：「PostgreSQL 的 MVCC 是怎么实现的？」应判 `out_of_scope` 并回固定话术，不反问、不硬答。

## 不做（YAGNI）

- 不加相似度阈值 cutoff（方案 B）：chroma 距离语义难标定，hybrid 的 RRF 分数无绝对意义、跨检索器不可比，脆且不通用。
- 不加 `out_of_scope` 专属 ablation flag。
- 话术不动态列库内书名。
