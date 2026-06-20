# 答案意图轴 + explain 精修工作流（设计 note）

> 状态：**架构方向已定，explain 路径待实现**。记问题、诊断、定下的形状、边界红线与已知缺口。
> 起因：2026-06-20「MySQL基础知识」被 split 拆成"字符集查看 / engine_cost / EXPLAIN / 小册前言"——一堆召回噪声里抠出的零散片段（见日志）。顺着追，暴露出"讲清楚"这条核心竞争力没有专属路径，且现有难度分类偷塞了意图。
> 背景轴诊断接 [routing-architecture-note.md](routing-architecture-note.md) 第 2 节（正交轴被压扁）。

## 1. 问题：难度分类偷塞了"答案意图"

当前 QA 预处理的六分类（`retrievable / pending_split / ambiguous / missing_info / other / out_of_scope`）名义上判"检索结构"，**实则夹带了答案意图**：

| 现有类 | 定义里写着 | 其实是哪条轴 |
|---|---|---|
| `pending_split` | "**对比/区别**…各自检索再综合" | compare **意图** |
| `other` | "**开放设计/权衡**比较" | design **意图** |
| `ambiguous` | "想要的维度/立场未给" | 意图维度缺失 |

"讲清楚一个概念"这条诉求更是**横跨两类**：

| 问题 | 现归 | 但诉求都是"讲清楚" |
|---|---|---|
| 「什么是聚簇索引」 | retrievable（窄、一次检索） | explain |
| 「讲懂MySQL基础知识」 | pending_split（宽、需扇出） | explain |

→ **检索结构轴**（一条够不够 / 扇出 / 多跳）和**答案意图轴**（查事实 / 讲清楚 / 对比 / 设计）被焊死在一个 enum 里。这是 routing-note 第 2 节"正交轴压扁"在 QA 内部的同源复发。

「MySQL基础知识」拆烂的直接机理：未单选书 → `_book_chapters` 因 `len != 1` 返回 `[]` → 结构骨架消失 → 拆分退化成纯内容主导 → vague query 的噪声召回被铁律"只准用素材实体"逼着抠成零散子查询。**垃圾进、垃圾出。**

## 2. 决策：不加第七类，加一条正交意图轴

"讲清楚"是 LibraryRAG 的主打竞争力，但它**不该是 enum 里的第七个平级类**——那是 routing-note 戳破的穷举陷阱（"讲懂MySQL基础知识"到底算 concept_explain 还是 pending_split？两个都对 = 正交轴又被压一维）。

正确形状是**两条正交轴**：

```
检索结构轴（不变）：retrievable / pending_split / other / ...   → 决定怎么取
答案意图轴（新增）：explain（讲透彻） / 非explain               → 决定怎么写
```

- 意图轴 v1 **实质是二元闸**：`explain` / 非 `explain`。
- `lookup / compare / design` 等"其他意图各自精修工作流"**暂定，以后再长**；现在它们全归"非 explain"，**默认滑进难度分类**走现有那套（一行不改）。

## 3. 定下的数据流

```
front_door 净化（已有） → clean_query
        ↓
[降噪 + 意图]  ← Call A（意图融进现有降噪，不加往返）→ clean_query' + intent
   ├ intent == explain → explain 工作流（新，独立）：
   │      列骨架(尺寸自适应,下限1节) → 广度从骨架涌现(单检索/扇出)
   │      → 教学体透彻合成（grounding 不松）
   │      ※ explain 跳过难度分类
   │
   └ intent != explain → 难度分类（六分类，原方向不动）← Call B（独立阶段）
          → 现有分支(retrievable/pending_split/ambiguous/missing_info/other/out_of_scope)
```

**成本（已拍板）**：拆成两次 call。非 explain 路径 = Call A（降噪+意图）+ Call B（难度），比今天多 1 次往返；explain 路径省掉难度那次。接受。

## 4. explain 工作流的几个关键设计点

| 点 | 决定 | 为什么 |
|---|---|---|
| **永远列骨架，不分窄宽** | 小概念也列（「LSN是什么」= 是什么/为什么/怎么算/关联/例子） | "讲清楚"本身就是骨架，只是节点多少不同 |
| **骨架尺寸自适应，下限 1 节** | 原子概念 → 1~2 节（退化成一段结构化回答） | 透彻 ≠ 注水；逼原子概念分五节会被撑出废话 |
| **广度从骨架涌现，不预分类** | 子查询指向同片区 → 单检索；互不重叠 → 扇出 | "窄/宽"不是一个前置分类口，是骨架的涌现属性 |
| **教学体合成** | 开场全景 → 逐节展开 → 收束 | 这是差异化的落点 |
| **grounding 不松** | 骨架/语气可由模型组织，**事实只来自检索 chunk** | explain 容易诱发模型用世界知识脑补；防幻觉红线不许破 |

**容错性**：意图判错是**优雅降级**不是崩——该 explain 被判非 explain，顶多答得不够透彻（仍正常答）；简单题被判 explain，顶多话多。没有"判错就卡死"的硬伤 → explain 闸可以判得激进点。

## 5. 边界 / 红线

- **意图在 content query 上判，不上提门口**：意图属 QA 内部（要干净 content query），front_door 保持精瘦只管对话准入。落点＝降噪之后。
- **难度分类原样保留**：非 explain 默认滑入，逻辑一行不改；只是它从"与降噪同一 call"变成独立 Call B。
- **explain 合成的事实源唯一性**：骨架优先 + 教学体绝不等于放权给模型答内容，事实仍只来自 chunk（沿用现有结构性防幻觉）。

## 6. 已知缺口（标记，不在 v1）

- **explain 里的多跳依赖**：如「讲清楚 MySQL 默认隔离级别会有哪些并发问题」——得先检索出默认级别是 RR、才知道讲哪些并发问题。骨架优先 + 扇出对付不了"下一步查什么取决于上一步查到什么"。v1 **先不接**；将来要么 explain 内加一个轻量"多跳？"探测转 agent，要么落回现有 `other`。
- **骨架 prompt 细化**：怎么让尺寸自适应、怎么避免原子概念注水——待写 spec 时定。
- **"广度涌现"的具体机制**：子查询去重 / 命中区域重叠度判定放哪、怎么轻量做。
- **意图 taxonomy 封口**：v1 二元够用；将来长 lookup/compare/design 时，要按**答案形状** MECE 切、有默认（lookup），别重蹈"出一个问题加一个意图"。
- **评测**：golden 需要 explain 类样例 + "是否真讲透彻"的度量（不止分类准确率）。

## 7. 与现状的迁移映射

| 现状 | 去哪 | 备注 |
|---|---|---|
| `_JUDGE_PROMPT`（降噪 + 难度分类，一个 call） | 拆：降噪并入 Call A（+意图）；难度分类独立成 Call B | 难度六分类内容不动，只是位置变 |
| split / decompose（章节树骨架） | explain 工作流的**骨架优先**取代它在宽 explain 上的活 | 章节树降级为"单本深挖时的可选精修"，不再是宽题的唯一拆法 |
| pending_split / other / ambiguous | 仍在难度分类下（非 explain 路径） | 待将来 compare/design 长出各自意图工作流时再迁 |
| `_book_chapters` 的 `len != 1` 死门槛 | explain 不再依赖单本章节树 → 该缺陷在 explain 路径自然绕开 | 非 explain 的 split 仍受其限，留作另一刀 |
