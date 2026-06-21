# ConversationScoper：全库多轮的隐式作用域收窄 —— 设计

**日期**：2026-06-21
**状态**：设计已评审（含一次实测根因修正），待写实现计划
**关联**：`core/workflow/front_door.py`、`core/workflow/doc_workflow.py`、`core/workflow/qa_capability.py`、`core/retrieval/retrieve.py`

---

## 1. 问题

多轮对话里，用户先问「讲讲 openclaw」，下一轮问「讲一下 gateway」。在**全库模式**（用户没在前端选定书籍，`book_titles` 为空）下，"gateway" 可能在多本书里都出现，检索会被其它书的同名概念污染。

现状两道防线，且**实测发现第一道对本场景根本不触发**：

- **门口指代消解（只消显式指代词，不接隐式话题续接）**：`front_door` 的净化规则只把**"它/这个/那个/前面提到的/这本书"**等显式指代词补全（prompt 明写「无指代则不动」）。而「讲一下 gateway」是个**裸概念名、无指代词**，门口按设计原样放过——**不会**补成「openclaw 的 gateway」。
  - 实测确证（2026-06-21 15:28 日志）：第二轮 `front_door: action=dispatch_qa clean_query='讲一下gateway'`，主体 openclaw **未被折进 query**。`query_preprocess.py` 也明说指代消解只在门口做、别处不补，故全链路没有任何地方做这个补全。
- **scope 元数据硬过滤（硬隔离，仅选书时生效）**：`retrieve.build_book_filters()` 在 `book_titles` 非空时下推 `book_title IN [...]`，dense + BM25 两路都过滤。**但全库模式下 `book_titles` 为空，这道防线不触发。**

**缺口**：全库模式下，裸概念续问（「gateway」）既没被门口补全主体、又没有 scope 硬过滤，probe / 检索拿到的就是各书同名概念混在一起的污染分布。

**对设计的连锁影响**：ConversationScoper **不能**假设 `clean_query` 已带主体（实测它不带）。因此 scoper 的锚定不依赖 clean_query，而是**自己从会话历史重新锚定**（见 §2.1、§4.1）。

**目标**：把全库模式下的"软约束"补成"在能确定主体落点时的硬隔离"，且：

- 覆盖**两种粒度**：openclaw 是书名、openclaw 是某书里/跨书的概念。
- 约束强度 = **硬过滤 + 透明 + 可纠偏**（推断偏了用户能一句话扩回全库）。
- 推断不出主体落点时**不收窄**（保持全库，本就该全库），绝不阻塞。

---

## 2. 为什么是「front_door 之后的独立节点」而非「放大 front_door」

`front_door` 跑在**最前面**，那时还没有任何内容 probe 结果——它手里只有书名目录（`list_books_text` 那种静态元数据查询），**没有正文语义命中**。

- 若 openclaw 是**书名**：front_door 理论上能用书名目录匹配出 scope。
- 若 openclaw 是**概念**："openclaw" 不是书名，目录里匹配不到，**必须跑内容 probe**才能定位它落哪本书。

而把内容 probe 塞进 front_door 有三个代价：① 与下游 `classify` 的 probe 重复；② 破 front_door 的红线（prompt 明写「tool 只能 list_books，绝不可 book_search，内容问题一律下沉」）；③ front_door 从"纯决策 + 目录查"变成"决策 + 内容检索"，边界糊掉。

**时序硬约束**：概念定位最早能知道的地方，是第一次内容 probe 跑完之后。因此把 scope 推断做成一个**独立节点**，放在 `front_door` 之后、`gate`/`classify`/`explain` 路由之前，让它自己跑一次轻量 probe。这个位置一处即罩住 explain 与非 explain 两条路。

### 2.1 为什么锚定收进 scoper，而不是让门口补全续问主体

实测根因暴露后，"把主体补进续问"有两条修法，本设计选后者（Fork B）：

- **Fork A（不选）·门口补全隐式话题续接**：扩门口净化规则，把「讲一下 gateway」补成「openclaw 的 gateway」。代价：① **话题切换误补**——用户若问的是与 openclaw 无关的通用 gateway，门口会硬焊 openclaw 上去；"延续 vs 切换"是门口每轮的判断题，会错；② 门口从保守变激进，污染 clean_query 给所有下游消费者。原设计刻意「无指代则不动」就是为避免这个。
- **Fork B（选）·scoper 自带会话锚定**：门口**保持保守不动**，锚定职责收进 scoper——它本就是负责 scope 的单元。scoper 用会话历史自己锚定（§4.1 的上下文拼接 probe），并以 probe 的集中度天然兜底话题切换：续问的概念若在上文主体的书里命中弱、在别书命中强 → 判定真切换 → 跟新集中度走，不硬焊。门口零风险，会话作用域逻辑集中一处。

---

## 3. 架构

```
front_door（已有，保持保守不动）
  └─ dispatch_qa → clean_query（裸概念续问不带主体，如「讲一下gateway」）
        │
        ▼
ConversationScoper（新增独立节点）            ← 仅当 user_book_titles 为空时动作
  ├─ 拼 probe 文本 = 最近 N 轮用户问 ⊕ clean_query（会话锚定，不依赖 clean_query 自带主体）
  ├─ 对拼接文本跑一次轻量 vector probe
  ├─ 统计命中片段的 book_title 分布 → 判主导书
  ├─ 有主导 → effective_book_titles = [主导书...] + note
  └─ 无主导/失败/probe 空 → None（保持全库）+ note=""
        │  写 ctx：book_titles ← effective or 原值；scope_note
        ▼
gate / classify / explain / retrieve / split / assume
  └─ scope 过滤侧【零改动】：统一读 ctx 的 book_titles → build_book_filters 硬过滤
  └─ 透明声明侧【小改】：把 scope_note 当答案前缀输出（preamble，见 §4.3）
```

**核心不变量**：`book_titles` 在 ctx 里始终表示"本轮检索的有效作用域"。

- 用户手选过书 → scoper no-op，值不变（手选硬约束永远赢）。
- 全库 + 推断出主导书 → 值被覆盖成 `[X]`，下游天然硬过滤，**下游分支零改动**。
- 全库 + 无主导 → 值仍为空（全库）。

---

## 4. 组件

### 4.1 `ConversationScoper`（新增，`core/workflow/conversation_scoper.py`）

注入式协作单元，仿 `QueryGate`/`Admitter`：注入依赖、对外只暴露一个 `run`、失败降级、便于单测。

```python
@dataclass
class ScopeDecision:
    effective_book_titles: Optional[list[str]]  # None = 不收窄（保持全库）
    note: str = ""                              # 透明声明（收窄时填，否则空）

class ConversationScoper:
    def __init__(
        self,
        index_manager,
        probe_retriever: Retriever,        # 独立 probe（vector 即可），不复用 classify 的
        probe_k: int = 8,
        n_history_turns: int = 2,          # 拼进 probe 的最近用户问轮数
        dominant_share: float = 0.60,      # 头部书占比阈值（可调）
        dominant_ratio: float = 2.0,       # 头部 ≥ 第二名几倍（可调）
        cover_share: float = 0.80,         # 多本封顶累计覆盖阈值（可调）
        max_books: int = 2,                # 最多锁几本
        min_count: int = 2,                # 单本计入主导的最小命中数（防噪声）
    ): ...

    async def run(
        self,
        clean_query: str,
        user_book_titles: Optional[list[str]],
        memory: Optional[ChatMemoryBuffer],   # 会话锚定来源
    ) -> ScopeDecision: ...
```

**`run` 逻辑**：

1. `user_book_titles` 非空 → 直接返回 `ScopeDecision(user_book_titles, note="")`（no-op）。
2. **拼接 probe 文本**：从 `memory` 取最近 `n_history_turns` 条**用户**消息（role==USER；空 memory → 仅用 clean_query），与 `clean_query` 换行拼成 probe 文本。裸概念续问靠这步把上文主体（openclaw）带进 probe；自包含强 query 则自己立得住。
3. 对**拼接文本**跑 probe（`probe_retriever.retrieve(..., top_k=probe_k)`）。
4. 统计命中片段的 `book_title` 计数，按计数降序。
5. **主导书判据（v1，可锁多本）**：
   - **单一主导**：`top1_share ≥ dominant_share` 且 `top1_count ≥ dominant_ratio × top2_count` → 锁 `[top1]`。
   - **少数主导**：非单一主导时，取按计数降序的最小前缀，使其累计占比 `≥ cover_share`，且前缀长度 `≤ max_books`、前缀内每本 `count ≥ min_count`、且被排除的尾部每本 `count < min_count` → 锁该前缀。
   - 否则（分散）→ `None`。
6. 命中为空 / probe 抛错 → `None`（保持全库）。
7. 收窄时产 `note`，如：`（我按《openclaw》里的内容回答；想看全部书可以说"在所有书里讲"。）\n`。

**降级铁律**：任何异常（probe 失败、空命中、统计异常）→ `ScopeDecision(None, "")`，全库照走，绝不阻塞。

### 4.2 `doc_workflow.route` 接线（改）

`front_door` 判 `dispatch_qa` 之后、返回 `PreprocessEvent` 之前：

```python
# dispatch_qa（含降级）—— memory/book_titles 在 route 顶部已取
await ctx.store.set("clean_query", decision.clean_query)
disable_scope = await ctx.store.get("disable_scope")  # D4：front_door 置位则跳过
if not disable_scope:
    scope = await self.scoper.run(decision.clean_query, book_titles, memory)
    await ctx.store.set("book_titles", scope.effective_book_titles or book_titles)
    await ctx.store.set("scope_note", scope.note)
return PreprocessEvent()
```

`ConversationScoper` 在 `DocQueryWorkflow.__init__` 与 `front_door`/`qa` 同处构造：`self.scoper = ConversationScoper(index_manager, probe_retriever=make_retriever(probe_retriever))`（沿用 §4.1 默认阈值；probe_retriever 名字复用 workflow 已有的 `probe_retriever` 入参，None→vector）。

### 4.3 透明声明输出（改，D3）

复用现有 `preamble`/`AnswerDeltaEvent` 机制（与 `assume` 的角度声明、missing_info 假设声明同源，前端零改）：

- `retrieve` 已有 `preamble` 参数 → 各 RetrieveAgentEvent 分支把 `scope_note` 经 preamble 传入。
- `split`、`explain`、`assume` 需补上 preamble 支持（assume 已有自己的 preamble，需与 scope_note 拼接，scope_note 在前）。
- 收尾时 `scope_note` 已拼进答案文本，落 DB 的答案含声明。

> 实现细节：各分支 step 从 ctx 取 `scope_note`，传入对应 capability 方法。空 note 不输出任何前缀。

### 4.4 纠偏（D4，v1 最小版）

`front_door` prompt 增一条：若本轮用户明确表示"在所有书里 / 不要限定范围 / 全库"，在 `dispatch_qa` 决策上附 `disable_scope=true`。`route` 读到则跳过 scoper（本轮全库）。`scope_note` 文案里告诉用户这句话怎么说。

> `disable_scope` 是**逐轮**信号，不持久化；每轮重新判定。

---

## 5. 数据流（全库下两轮示例）

关键点：front_door 对裸概念续问**不补主体**（实测，§1），clean_query 保持裸；锚定靠 scoper 的拼接 probe 文本。

| 轮次 | 用户输入 | clean_query（门口产出，裸） | scoper probe 文本（拼接） | probe 书分布 | effective scope | 输出前缀 |
|---|---|---|---|---|---|---|
| 1 | 讲讲 openclaw | 讲讲 openclaw | 讲讲 openclaw | 《openclaw》×7，其它×1 | [openclaw] | 我按《openclaw》回答… |
| 2 | 讲一下 gateway | 讲一下 gateway | 讲讲 openclaw / 讲一下 gateway | 《openclaw》×6，《X》×2 | [openclaw] | 我按《openclaw》回答… |
| 3 | 在所有书里讲讲 gateway | 讲讲 gateway（disable_scope） | —（跳过 scoper） | — | None（全库） | 无 |

---

## 6. 错误处理

| 情况 | 行为 |
|---|---|
| 用户手选了书 | scoper no-op，effective = 手选值 |
| probe 抛错 / 空命中 | `None`，保持全库，无声明 |
| 命中分散无主导 | `None`，保持全库，无声明 |
| 空 memory（首轮无历史） | 拼接文本退化为仅 clean_query；首轮本就无上文可串，行为等同单 query probe |
| front_door 降级（用原 query 当 clean_query） | scoper 照常拼接历史跑 probe；定位不到则不收窄 |
| `disable_scope` 置位 | 跳过 scoper，本轮全库 |

全链路任意失败都倒向"不收窄 + 全库"，与全项目降级铁律一致。

---

## 7. 测试

**ConversationScoper 单测**（mock `probe_retriever` 返回构造的 book 分布）：

- 单一主导（占比/倍数过线）→ 锁单本。
- 少数主导（两本覆盖 ≥ cover_share，尾部为噪声）→ 锁两本。
- 分散（无主导）→ `None`。
- 命中为空 / probe 抛错 → `None`，不抛。
- `user_book_titles` 非空 → no-op，原样返回。
- 阈值边界（恰好 = / 略低于 dominant_share、dominant_ratio）。

**doc_workflow 装配测**：

- 全库 + probe 主导某书 → `ctx.book_titles` 被锁；`scope_note` 非空且进入答案前缀。
- 全库 + 分散 → `ctx.book_titles` 仍空；无前缀。
- 用户手选 → scoper no-op，`book_titles` 不变。
- `disable_scope` → scoper 不被调用。
- explain 路与非 explain 路都继承 `ctx.book_titles`（一处接线罩两路）。

---

## 8. 范围与非目标

**v1 范围**：ConversationScoper 单元 + route 接线 + scope_note 透明声明 + disable_scope 最小纠偏 + 测试。

**非目标（留后续）**：

- probe 复用（scoper 与 classify 共享一次 probe）—— v1 接受重复，隔离优先。
- 阈值自动调参 / scope 推断准确率的离线评测（可后续接 eval/）。
- 持久化的会话主体追踪（v1 每轮从 probe 重新推断，不维护跨轮主体状态）。
