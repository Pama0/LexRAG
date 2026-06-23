from types import SimpleNamespace

from eval.retrieval.label import merge_pool, parse_judgement


def _nws(node_id: str):
    return SimpleNamespace(node=SimpleNamespace(node_id=node_id))


def test_merge_pool_dedup_preserves_order():
    dense = [_nws("a"), _nws("b")]
    sparse = [_nws("b"), _nws("c")]
    ids, id2node = merge_pool([dense, sparse])
    assert ids == ["a", "b", "c"]              # 保序、去重
    assert set(id2node) == {"a", "b", "c"}
    assert id2node["a"].node_id == "a"


def test_parse_judgement_maps_local_index_to_id():
    idx_to_id = {0: "a", 1: "b", 2: "c"}
    assert parse_judgement('{"0": 1, "1": 0, "2": 1}', idx_to_id) == {"a", "c"}


def test_parse_judgement_tolerates_fence_and_bad_keys():
    idx_to_id = {0: "a", 1: "b"}
    text = '```json\n{"0": 1, "9": 1, "x": 1, "1": "1"}\n```'
    # 9/x 超范围或非法忽略；"1":"1" 视为相关
    assert parse_judgement(text, idx_to_id) == {"a", "b"}
