# legacy/ —— 法条业务冻结归档

本目录是项目早期"法律条文 RAG"相关代码的冻结快照，2026-05-26 整合时从主干移出。
**不保证可直接运行**：部分共享逻辑（如 `RAGIndexManager.add_documents` 法条增量索引）
已从 `core/rag/data_loader.py` 删除，仅存于该提交之前的 git 历史。

将来重启法条业务时：
1. 从 git 历史恢复 `data_loader.add_documents` / `_update_citation_graph` / `_fetch_all_nodes`；
2. 校验本目录内 import（已尽力 rewire 为 `legacy.*`）；
3. 重建 chroma `documents` 集合。
