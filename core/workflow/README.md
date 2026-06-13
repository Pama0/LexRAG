# core/workflow/ —— 文档问答 workflow 包

当前 book 问答走 **`DocQueryWorkflow`**（顶层编排）+ **`QaCapability`**（QA 检索/合成实质），
由 **`DocQueryService`** 在装配层组装，`api/main.py` 与根 `main.py` 各自注入。

> 历史：早期 `book_rag.py`（`BookRagWorkflow`）+ `core/tools/book_tools.py` 已退役并删除，
> 逻辑被本包的 `doc_workflow` / `qa_capability` 取代。

## 结构

```
DocQueryService (doc_query_service.py)   装配 + 每请求新建 workflow
   └─ DocQueryWorkflow (doc_workflow.py) 顶层编排：门口 Router → QA 分支 → finalize
        ├─ IntentRouter (intent_router.py)      Layer1：净化 + 意图分类
        └─ QaCapability (qa_capability.py)       Layer2：probe→判 category→检索/拆解/归纳
             ├─ QueryPreprocessor (query_preprocess.py)  降噪 + 难度分类
             ├─ QueryDecomposer  (query_decompose.py)    pending_split 拆解
             └─ DimensionExtractor (query_dimension.py)  ambiguous 归纳维度
```

## 新增一个分支/能力

1. 在 `qa_capability.py` 加分支方法（参考现有 `retrieve` / `split` / `assume`），
   通过 `ctx.write_event_to_stream` 推流式事件。
2. 在 `query_preprocess.py` 的 category 体系里挂上对应类别（如需新类别）。
3. 在 `doc_workflow.py` 的路由处把新 category dispatch 到该分支。
4. 决策开关（供评测 ablation）走 `DocQueryWorkflow` 的 flag 参数。

分层约束：`api/`(Web) → `core/`(领域) → `configs/`，core 不依赖 api。
