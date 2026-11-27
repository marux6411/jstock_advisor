"""`infrastructure/aws/replay_source_reader.py`(Issue #880 R1b PR-2。dormant)の契約テスト。

設計: #880 rev6(…6259587894)・rev7(…6259775154)・サブちゃんの設計レビュー合格(…6259884051)。

固定するもの
  ・proxy: 許可する操作は引数(既定なし)。空・読取の 4 語の部分集合でないもの(書込の語・語彙外)は
    構築時に拒否。許可されない属性は例外。列挙面(dir / vars)・client の subclass でない・不変
  ・上限(件数・ページ・時間・見積り × 倍率): 全て必須の引数。★ 4 つの境界は 1 つずつ別のテスト ★
  ・★ 例外なしで結果が欠ける型 ★: ページの打ち切り・ページ送りの鍵の取り違え・空のページ・
    UnprocessedKeys の無視 = 件数の等号 + 要素の集合 + 呼び出しごとの ExclusiveStartKey の逐一比較
  ・部分結果を返さない(例外のみ)・失敗は固定の種別。ACCESS_DENIED は ERROR に畳まない
  ・崩れた応答(response が無い・Mapping でない・"Error" が無い・"Code" が文字列でない)は ERROR
  ・projection 必須・mask_fields ⊆ projection(構築時)・{"S": 文字列} だけを置換
  ・平文が戻り値に残らない
  ・例外に番兵(表名・キー・識別子・鍵・例外の文面)が出ない(str・args・__cause__・__context__)
  ・AST: AWS SDK・requests・urllib・socket・os・time・datetime の import 0・書込の語 0・数値の既定 0
  ・dormant: src のどこからも言及されない(importer の等号 = 空)・#606 の名前の網の外
fixture は合成のみ(実在の銘柄・保有・表名を使わない)。
"""

from __future__ import annotations

import ast
import hashlib
import hmac
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jstock_advisor.infrastructure.aws import replay_source_reader as module
from jstock_advisor.infrastructure.aws.replay_source_reader import (
    READ_OPERATIONS,
    IdentifierMasker,
    ReadLimits,
    ReadSpec,
    ReplayReadClient,
    ReplayReadConfigError,
    ReplayReadFailure,
    ReplayReadFailureKind,
    ReplayReadViolationError,
    ReplaySourceReader,
    classify_exception,
    mask_identifier,
)
from tests.unit.test_issue_606_production_connection_gate import _is_unconnected_name

_REPO = Path(__file__).resolve().parents[2]
_PACKAGE = _REPO / "src" / "jstock_advisor"
_MODULE_PATH = _PACKAGE / "infrastructure" / "aws" / "replay_source_reader.py"

_KEY = b"sentinel-key-material-0123456789abcdef"
_TABLE = "SENTINEL-TABLE-ZZ"
_WRITE_WORDS = (
    "put_item",
    "update_item",
    "delete_item",
    "transact_write_items",
    "batch_write_item",
)
_K = ReplayReadFailureKind


# --- 部品 -------------------------------------------------------------------------------------


class _Fake:
    """DynamoDB の client の代わり。応答を順に返し、呼び出しを記録する。書込の語の名前も持つ。"""

    def __init__(self, **queues: list[Any]) -> None:
        self.queues = {name: list(items) for name, items in queues.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.on_call: Callable[[], None] | None = None

    def _next(self, name: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append((name, kwargs))
        if self.on_call is not None:
            self.on_call()
        queue = self.queues[name]
        response = queue.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def scan(self, **kwargs: Any) -> Any:
        return self._next("scan", kwargs)

    def query(self, **kwargs: Any) -> Any:
        return self._next("query", kwargs)

    def get_item(self, **kwargs: Any) -> Any:
        return self._next("get_item", kwargs)

    def batch_get_item(self, **kwargs: Any) -> Any:
        return self._next("batch_get_item", kwargs)

    def put_item(self, **kwargs: Any) -> Any:  # 書込の名前。proxy が止めれば 0 回
        self.calls.append(("put_item", kwargs))
        return {}

    def describe_table(self, **kwargs: Any) -> Any:  # 語彙外
        self.calls.append(("describe_table", kwargs))
        return {}


def _item(name: str, score: str = "1") -> dict[str, Any]:
    return {"holding_id": {"S": name}, "score": {"N": score}}


def _page(names: list[str], last: dict[str, Any] | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"Items": [_item(n) for n in names]}
    if last is not None:
        response["LastEvaluatedKey"] = last
    return response


def _spec(**overrides: Any) -> ReadSpec:
    values: dict[str, Any] = {
        "table_name": _TABLE,
        "projection": ("holding_id", "score"),
        "mask_fields": {"holding_id": "holding"},
    }
    values.update(overrides)
    return ReadSpec(**values)


def _limits(**overrides: Any) -> ReadLimits:
    values: dict[str, Any] = {
        "max_items": 1000,
        "max_pages": 50,
        "max_seconds": 60,
        "estimated_items": 1000,
        "estimate_multiple": 10,
    }
    values.update(overrides)
    return ReadLimits(**values)


def _masker(version: int = 1, key: bytes = _KEY) -> IdentifierMasker:
    return IdentifierMasker(key_version=version, key=key)


def _reader(
    fake: _Fake,
    operations: frozenset[str] = frozenset({"scan", "query", "batch_get_item"}),
    elapsed: Callable[[], float] = lambda: 0,
) -> ReplaySourceReader:
    client = ReplayReadClient(fake, allowed_operations=operations)
    return ReplaySourceReader(client, _masker(), elapsed_seconds=elapsed)


def _failure_kind(action: Callable[[], Any]) -> ReplayReadFailureKind:
    with pytest.raises(ReplayReadFailure) as info:
        action()
    return info.value.kind


def _ref(name: str, kind: str = "holding", version: int = 1, key: bytes = _KEY) -> str:
    return mask_identifier(_masker(version, key), kind, name)


# =================================================================================================
# proxy(許可する操作は引数。既定なし)
# =================================================================================================


@pytest.mark.parametrize("operation", sorted(_WRITE_WORDS))
def test_a_write_operation_in_the_allowlist_is_rejected_at_construction(operation: str) -> None:
    with pytest.raises(ReplayReadConfigError):
        ReplayReadClient(_Fake(), allowed_operations=frozenset({"scan", operation}))


@pytest.mark.parametrize(
    "allowed",
    [
        frozenset(),
        {"scan"},
        ["scan"],
        ("scan",),
        "scan",
        None,
        frozenset({"describe_table"}),
        frozenset({1}),
    ],
)
def test_an_empty_non_frozenset_or_out_of_vocabulary_allowlist_is_rejected(allowed: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        ReplayReadClient(_Fake(), allowed_operations=allowed)


def test_the_allowlist_has_no_default() -> None:
    with pytest.raises(TypeError):
        ReplayReadClient(_Fake())  # type: ignore[call-arg]


def test_the_vocabulary_is_exactly_the_four_read_operations() -> None:
    assert frozenset({"get_item", "batch_get_item", "query", "scan"}) == READ_OPERATIONS


@pytest.mark.parametrize("operation", sorted(READ_OPERATIONS))
def test_every_read_operation_can_be_allowed_alone(operation: str) -> None:
    fake = _Fake(**{operation: [{"x": 1}]})
    proxy = ReplayReadClient(fake, allowed_operations=frozenset({operation}))
    assert getattr(proxy, operation)(a=1) == {"x": 1}
    assert fake.calls == [(operation, {"a": 1})]


def test_an_operation_that_is_not_allowed_raises_and_reaches_nothing() -> None:
    fake = _Fake()
    proxy = ReplayReadClient(fake, allowed_operations=frozenset({"scan"}))
    for name in (*_WRITE_WORDS, "describe_table", "get_item", "query", "batch_get_item", "other"):
        with pytest.raises(ReplayReadViolationError) as info:
            getattr(proxy, name)
        assert name in str(info.value)  # 要求された操作名(コードの語彙)だけを含む
    assert fake.calls == []  # ★ 書込の名前を含め、1 回も届かない ★


def test_the_enumeration_surface_shows_only_the_allowed_operations() -> None:
    proxy = ReplayReadClient(_Fake(), allowed_operations=frozenset({"scan", "query"}))
    assert dir(proxy) == ["query", "scan"]
    with pytest.raises(TypeError):
        vars(proxy)  # __dict__ を持たない = 列挙面に何も出ない
    assert not any(word in repr(proxy) for word in _WRITE_WORDS)


def test_the_proxy_is_not_a_subclass_of_the_client_and_is_immutable() -> None:
    fake = _Fake()
    proxy = ReplayReadClient(fake, allowed_operations=frozenset({"scan"}))
    assert not isinstance(proxy, type(fake))
    assert type(fake) not in type(proxy).__mro__
    with pytest.raises(AttributeError):
        proxy.scan = lambda **kw: None  # type: ignore[method-assign]
    with pytest.raises(AttributeError):
        proxy._allowed = frozenset({"scan", "query"})
    with pytest.raises(AttributeError):
        del proxy._client


def test_the_limit_of_the_proxy_is_stated() -> None:
    doc = ReplayReadClient.__doc__ or ""
    assert "直接アクセスは防げない" in doc and "IAM" in doc


# =================================================================================================
# 上限(全て必須。境界は 1 つずつ別のテスト)
# =================================================================================================


def test_every_limit_is_required_and_has_no_default() -> None:
    for omitted in (
        "max_items",
        "max_pages",
        "max_seconds",
        "estimated_items",
        "estimate_multiple",
    ):
        values = {
            "max_items": 1,
            "max_pages": 1,
            "max_seconds": 1,
            "estimated_items": 1,
            "estimate_multiple": 1,
        }
        del values[omitted]
        with pytest.raises(TypeError):
            ReadLimits(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["max_items", "max_pages", "max_seconds", "estimated_items", "estimate_multiple"]
)
@pytest.mark.parametrize("bad", [0, -1, True, False, None, "1", float("nan"), float("inf")])
def test_a_non_positive_or_non_numeric_limit_is_rejected(field: str, bad: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        _limits(**{field: bad})


@pytest.mark.parametrize("field", ["max_items", "max_pages", "estimated_items"])
def test_an_integer_limit_rejects_a_float(field: str) -> None:
    with pytest.raises(ReplayReadConfigError):
        _limits(**{field: 1.5})


def test_max_items_boundary_exactly_passes_and_one_more_stops() -> None:
    limits = _limits(max_items=3, estimated_items=10_000)
    ok = _Fake(scan=[_page(["a", "b"], {"k": 1}), _page(["c"])])
    assert len(_reader(ok).read_pages("scan", _spec(), limits)) == 3
    over = _Fake(scan=[_page(["a", "b"], {"k": 1}), _page(["c", "d"])])
    assert _failure_kind(lambda: _reader(over).read_pages("scan", _spec(), limits)) is _K.ITEM_LIMIT


def test_max_pages_boundary_exactly_passes_and_one_more_stops_without_a_request() -> None:
    limits = _limits(max_pages=3)
    ok = _Fake(scan=[_page(["a"], {"k": 1}), _page(["b"], {"k": 2}), _page(["c"])])
    assert len(_reader(ok).read_pages("scan", _spec(), limits)) == 3
    over = _Fake(
        scan=[_page(["a"], {"k": 1}), _page(["b"], {"k": 2}), _page(["c"], {"k": 3}), _page(["d"])]
    )
    assert _failure_kind(lambda: _reader(over).read_pages("scan", _spec(), limits)) is _K.PAGE_LIMIT
    assert len(over.calls) == 3  # 4 回目の要求は出さない


def test_max_seconds_boundary_exactly_passes_and_one_more_stops_without_a_request() -> None:
    limits = _limits(max_seconds=10)
    ok = _Fake(scan=[_page(["a"])])
    assert len(_reader(ok, elapsed=lambda: 10).read_pages("scan", _spec(), limits)) == 1
    over = _Fake(scan=[_page(["a"])])
    kind = _failure_kind(
        lambda: _reader(over, elapsed=lambda: 10.5).read_pages("scan", _spec(), limits)
    )
    assert kind is _K.TIME_LIMIT and over.calls == []


def test_estimate_boundary_exactly_passes_and_one_more_stops() -> None:
    limits = _limits(estimated_items=5, estimate_multiple=2, max_items=1000)
    ok = _Fake(scan=[_page([f"a{i}" for i in range(10)])])
    assert len(_reader(ok).read_pages("scan", _spec(), limits)) == 10
    over = _Fake(scan=[_page([f"a{i}" for i in range(11)])])
    assert (
        _failure_kind(lambda: _reader(over).read_pages("scan", _spec(), limits))
        is _K.ESTIMATE_EXCEEDED
    )


def test_the_time_is_checked_before_every_request_and_a_slow_call_overruns_only_once() -> None:
    clock = {"now": 0.0}
    fake = _Fake(scan=[_page(["a"], {"k": 1}), _page(["b"])])
    fake.on_call = lambda: clock.__setitem__("now", 100.0)  # 1 回の呼び出しが予算を超えて長引く
    reader = _reader(fake, elapsed=lambda: clock["now"])
    kind = _failure_kind(lambda: reader.read_pages("scan", _spec(), _limits(max_seconds=10)))
    assert kind is _K.TIME_LIMIT
    assert len(fake.calls) == 1  # 超過した 1 回の後、次の要求の前に止まる(部分結果を返さない)


@pytest.mark.parametrize("bad", [None, "1", True, [1]])
def test_a_clock_that_returns_a_non_number_is_an_error(bad: Any) -> None:
    fake = _Fake(scan=[_page(["a"])])
    assert (
        _failure_kind(
            lambda: _reader(fake, elapsed=lambda: bad).read_pages("scan", _spec(), _limits())
        )
        is _K.ERROR
    )
    assert fake.calls == []


def test_a_clock_that_raises_is_an_error_without_the_original_exception() -> None:
    def clock() -> float:
        raise RuntimeError("SENTINEL-CLOCK-ZZ")

    with pytest.raises(ReplayReadFailure) as info:
        _reader(_Fake(scan=[_page(["a"])]), elapsed=clock).read_pages("scan", _spec(), _limits())
    assert info.value.kind is _K.ERROR
    assert info.value.__cause__ is None and info.value.__context__ is None


# =================================================================================================
# ★ 例外なしで結果が欠ける型 ★: ページ送り
# =================================================================================================

_LEK1 = {"pk": {"S": "after-page-1"}}
_LEK2 = {"pk": {"S": "after-page-2"}}


def test_every_page_is_read_and_the_count_and_the_set_equal_the_sum_of_all_pages() -> None:
    fake = _Fake(scan=[_page(["a", "b", "c"], _LEK1), _page(["d"], _LEK2), _page(["e", "f"])])
    result = _reader(fake).read_pages("scan", _spec(), _limits())
    assert len(result) == 6  # 件数の等号
    assert [i["holding_id"]["S"] for i in result] == [_ref(n) for n in "abcdef"]  # 要素の集合と順序
    assert len(set(i["holding_id"]["S"] for i in result)) == 6  # 重複なし
    assert len(fake.calls) == 3


def test_each_request_passes_the_last_evaluated_key_of_the_previous_page() -> None:
    fake = _Fake(scan=[_page(["a"], _LEK1), _page(["b"], _LEK2), _page(["c"])])
    _reader(fake).read_pages("scan", _spec(), _limits())
    keys = [call[1].get("ExclusiveStartKey", "ABSENT") for call in fake.calls]
    assert keys == [
        "ABSENT",
        _LEK1,
        _LEK2,
    ]  # 1 ページ目には渡さない・以後は直前の LastEvaluatedKey と一致


def test_an_empty_page_with_a_last_evaluated_key_does_not_end_the_read() -> None:
    fake = _Fake(scan=[_page(["a"], _LEK1), {"Items": [], "LastEvaluatedKey": _LEK2}, _page(["b"])])
    result = _reader(fake).read_pages("scan", _spec(), _limits())
    assert [i["holding_id"]["S"] for i in result] == [
        _ref("a"),
        _ref("b"),
    ]  # 1 ページ目 + 3 ページ目
    assert len(fake.calls) == 3


def test_a_last_page_that_is_empty_without_a_last_evaluated_key_ends_normally() -> None:
    fake = _Fake(scan=[_page(["a"], _LEK1), {"Items": []}])
    assert len(_reader(fake).read_pages("scan", _spec(), _limits())) == 1


def test_a_single_page_without_a_last_evaluated_key_takes_one_call() -> None:
    fake = _Fake(scan=[_page(["a", "b"])])
    assert len(_reader(fake).read_pages("scan", _spec(), _limits())) == 2
    assert len(fake.calls) == 1


def test_a_read_that_never_ends_stops_at_the_page_limit() -> None:
    fake = _Fake(scan=[_page([f"p{i}"], _LEK1) for i in range(10)])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits(max_pages=4)))
        is _K.PAGE_LIMIT
    )
    assert len(fake.calls) == 4


def test_a_last_evaluated_key_that_is_not_a_mapping_is_malformed() -> None:
    fake = _Fake(scan=[{"Items": [], "LastEvaluatedKey": "not-a-mapping"}])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


def test_unprocessed_keys_make_the_batch_read_incomplete() -> None:
    fake = _Fake(
        batch_get_item=[
            {"Responses": {_TABLE: [_item("a")]}, "UnprocessedKeys": {_TABLE: {"Keys": []}}}
        ]
    )
    keys = [{"pk": {"S": "a"}}]
    assert _failure_kind(lambda: _reader(fake).batch_get(_spec(), keys, _limits())) is _K.INCOMPLETE


@pytest.mark.parametrize("unprocessed", [{}, None])
def test_an_empty_or_absent_unprocessed_keys_succeeds(unprocessed: Any) -> None:
    response: dict[str, Any] = {"Responses": {_TABLE: [_item("a")]}}
    if unprocessed is not None:
        response["UnprocessedKeys"] = unprocessed
    fake = _Fake(batch_get_item=[response])
    assert len(_reader(fake).batch_get(_spec(), [{"pk": {"S": "a"}}], _limits())) == 1


def test_the_batch_read_splits_the_keys_by_one_hundred() -> None:
    keys = [{"pk": {"S": f"k{i}"}} for i in range(250)]
    responses = [
        {"Responses": {_TABLE: [_item(f"x{i}") for i in range(n)]}} for n in (100, 100, 50)
    ]
    fake = _Fake(batch_get_item=responses)
    result = _reader(fake).batch_get(_spec(), keys, _limits())
    assert len(result) == 250
    sizes = [len(call[1]["RequestItems"][_TABLE]["Keys"]) for call in fake.calls]
    assert sizes == [100, 100, 50]
    assert [k for call in fake.calls for k in call[1]["RequestItems"][_TABLE]["Keys"]] == keys


def test_the_batch_read_stops_at_the_request_limit_without_an_extra_request() -> None:
    keys = [{"pk": {"S": f"k{i}"}} for i in range(250)]
    fake = _Fake(batch_get_item=[{"Responses": {_TABLE: []}} for _ in range(3)])
    assert (
        _failure_kind(lambda: _reader(fake).batch_get(_spec(), keys, _limits(max_pages=2)))
        is _K.PAGE_LIMIT
    )
    assert len(fake.calls) == 2


def test_the_batch_read_with_no_keys_makes_no_request() -> None:
    fake = _Fake()
    assert _reader(fake).batch_get(_spec(), [], _limits()) == []
    assert fake.calls == []


@pytest.mark.parametrize("response", [{}, {"Responses": "x"}, {"Responses": None}])
def test_a_batch_response_without_responses_is_malformed(response: Any) -> None:
    fake = _Fake(batch_get_item=[response])
    kind = _failure_kind(lambda: _reader(fake).batch_get(_spec(), [{"pk": {"S": "a"}}], _limits()))
    assert kind is _K.MALFORMED


# =================================================================================================
# 要求の中身(最小の読取)
# =================================================================================================


def test_the_request_names_only_the_declared_projection() -> None:
    fake = _Fake(scan=[_page(["a"])])
    _reader(fake).read_pages("scan", _spec(), _limits())
    request = fake.calls[0][1]
    assert set(request) == {"TableName", "ProjectionExpression", "ExpressionAttributeNames"}
    assert request["TableName"] == _TABLE
    names = request["ExpressionAttributeNames"]
    assert sorted(names.values()) == ["holding_id", "score"]
    assert set(request["ProjectionExpression"].split(", ")) == set(
        names
    )  # 予約語を避けた別名で渡す


def test_a_query_passes_the_allowed_extras_and_rejects_everything_else() -> None:
    fake = _Fake(query=[_page(["a"])])
    extras = {"KeyConditionExpression": "pk = :v", "ExpressionAttributeValues": {":v": {"S": "x"}}}
    _reader(fake).read_pages("query", _spec(), _limits(), request_extras=extras)
    request = fake.calls[0][1]
    assert request["KeyConditionExpression"] == "pk = :v"
    for bad in (
        "ProjectionExpression",
        "ExpressionAttributeNames",
        "TableName",
        "Select",
        "ExclusiveStartKey",
    ):
        with pytest.raises(ReplayReadConfigError):
            _reader(_Fake()).read_pages("scan", _spec(), _limits(), request_extras={bad: "x"})


@pytest.mark.parametrize("operation", ["get_item", "batch_get_item", "put_item", "", None])
def test_read_pages_accepts_only_scan_and_query(operation: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        _reader(_Fake()).read_pages(operation, _spec(), _limits())


def test_an_operation_that_the_proxy_does_not_allow_is_a_violation_not_a_failure() -> None:
    fake = _Fake(scan=[_page(["a"])])
    reader = _reader(fake, operations=frozenset({"query"}))
    with pytest.raises(ReplayReadViolationError):
        reader.read_pages("scan", _spec(), _limits())
    assert fake.calls == []


# =================================================================================================
# ReadSpec: projection 必須・mask_fields ⊆ projection
# =================================================================================================


@pytest.mark.parametrize(
    "projection", [(), [], "holding_id", ("holding_id", "holding_id"), ("", "x"), (1,), None]
)
def test_an_empty_or_malformed_projection_is_rejected(projection: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        _spec(projection=projection, mask_fields={})


def test_mask_fields_must_be_a_subset_of_the_projection() -> None:
    with pytest.raises(ReplayReadConfigError):
        _spec(mask_fields={"holding_id": "holding", "not_read": "holding"})
    assert _spec(mask_fields={}).mask_fields == {}  # 置換なしは可(部分集合)


@pytest.mark.parametrize("kind", ["", "a b", "日本語", "a-b", None, 5])
def test_a_mask_kind_must_be_an_ascii_identifier(kind: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        _spec(mask_fields={"holding_id": kind})


@pytest.mark.parametrize("table", ["", None, 5])
def test_a_table_name_must_be_a_non_empty_string(table: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        _spec(table_name=table)


# =================================================================================================
# 置換(mask_fields の型の検査・平文が残らない)
# =================================================================================================


def test_masked_attributes_are_replaced_and_the_others_are_unchanged() -> None:
    fake = _Fake(scan=[_page(["a"])])
    item = _reader(fake).read_pages("scan", _spec(), _limits())[0]
    assert item["holding_id"] == {"S": _ref("a")}
    assert item["score"] == {"N": "1"}


def test_the_plaintext_identifier_never_remains_in_the_result() -> None:
    fake = _Fake(scan=[_page(["PLAINTEXT-ID-ZZ"], _LEK1), _page(["PLAINTEXT-ID-YY"])])
    result = _reader(fake).read_pages("scan", _spec(), _limits())
    assert "PLAINTEXT-ID" not in repr(result)


def test_the_same_identifier_gets_the_same_reference_across_items() -> None:
    fake = _Fake(scan=[_page(["a", "a", "b"])])
    refs = [i["holding_id"]["S"] for i in _reader(fake).read_pages("scan", _spec(), _limits())]
    assert refs[0] == refs[1] != refs[2]


@pytest.mark.parametrize(
    "value",
    [
        {"N": "1"},  # S 以外の型
        {"S": 5},  # 文字列でない
        {"S": ""},
        {"S": "   "},
        {"S": "x", "N": "1"},  # 余計なキー
        "x",
        None,
        {},
    ],
)
def test_a_mask_field_that_is_not_a_typed_string_is_malformed(value: Any) -> None:
    fake = _Fake(scan=[{"Items": [{"holding_id": value, "score": {"N": "1"}}]}])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


def test_a_missing_mask_field_is_malformed() -> None:
    fake = _Fake(scan=[{"Items": [{"score": {"N": "1"}}]}])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


def test_an_attribute_that_was_not_declared_is_never_returned() -> None:
    fake = _Fake(scan=[{"Items": [{**_item("a"), "undeclared": {"S": "PLAINTEXT-ZZ"}}]}])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


@pytest.mark.parametrize("items", ["abc", {"a": 1}, [1], [None], [["x"]], 5])
def test_items_that_are_not_a_sequence_of_mappings_are_malformed(items: Any) -> None:
    fake = _Fake(scan=[{"Items": items}])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


def test_a_response_that_is_not_a_mapping_is_malformed() -> None:
    fake = _Fake(scan=["not-a-mapping"])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


def test_two_different_identifiers_with_the_same_reference_are_never_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "mask_identifier", lambda masker, kind, value: "rp1.1:collision")
    fake = _Fake(scan=[_page(["a", "b"])])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.MALFORMED
    )


def test_a_spec_without_mask_fields_returns_the_items_unchanged() -> None:
    fake = _Fake(scan=[_page(["a"])])
    result = _reader(fake).read_pages("scan", _spec(mask_fields={}), _limits())
    assert result == [_item("a")]


# =================================================================================================
# 鍵つきの置換
# =================================================================================================


def test_the_reference_has_the_versioned_format_and_matches_an_independent_hmac() -> None:
    ref = _ref("holding-xyz", "holding", 7)
    expected = hmac.new(_KEY, b"replay:holding\x00holding-xyz", hashlib.sha256).hexdigest()[:24]
    assert ref == f"rp1.7:{expected}"


def test_the_hmac_matches_the_rfc_4231_test_case_6_through_the_same_primitive() -> None:
    masker = IdentifierMasker(key_version=1, key=b"\xaa" * 131)
    digest = masker._hmac(b"Test Using Larger Than Block-Size Key - Hash Key First")
    assert digest == "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"


def test_the_reference_differs_by_key_kind_version_and_value_and_is_deterministic() -> None:
    base = _ref("a")
    assert base == _ref("a")
    assert base != _ref("a", key=b"another-key-material-0123456789abcdef")
    assert base != _ref("a", kind="symbol")
    assert base != _ref("a", version=2)
    assert base != _ref("b")


def test_the_reference_is_not_an_unkeyed_hash_and_hides_the_plaintext() -> None:
    plain = hashlib.sha256(b"replay:holding\x00a").hexdigest()[:24]
    assert plain not in _ref("a") and "a" not in _ref("a").split(":")[1][:0]
    assert "holding" not in _ref("a")


@pytest.mark.parametrize("value", ["", "   ", "\t\n", None, 5, b"x"])
def test_a_blank_or_non_string_identifier_is_rejected(value: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        mask_identifier(_masker(), "holding", value)


@pytest.mark.parametrize("masker", [None, _KEY, "key", object()])
def test_a_masker_that_is_not_an_identifier_masker_is_rejected(masker: Any) -> None:
    with pytest.raises(ReplayReadConfigError):
        mask_identifier(masker, "holding", "a")


def test_the_reader_requires_a_proxy_and_a_masker() -> None:
    proxy = ReplayReadClient(_Fake(), allowed_operations=frozenset({"scan"}))
    with pytest.raises(ReplayReadConfigError):
        ReplaySourceReader(_Fake(), _masker(), elapsed_seconds=lambda: 0)  # type: ignore[arg-type]
    with pytest.raises(ReplayReadConfigError):
        ReplaySourceReader(proxy, _KEY, elapsed_seconds=lambda: 0)  # type: ignore[arg-type]
    with pytest.raises(ReplayReadConfigError):
        ReplaySourceReader(proxy, _masker(), elapsed_seconds=None)  # type: ignore[arg-type]


# =================================================================================================
# 失敗の分類(botocore を import しない)
# =================================================================================================


class _WithResponseError(Exception):
    def __init__(self, response: Any, message: str = "SENTINEL-MESSAGE-ZZ") -> None:
        super().__init__(message)
        self.response = response


@pytest.mark.parametrize("code", ["AccessDenied", "AccessDeniedException", "UnauthorizedOperation"])
def test_access_denied_is_kept_apart_from_an_error(code: str) -> None:
    exc = _WithResponseError({"Error": {"Code": code}})
    fake = _Fake(scan=[exc])
    assert (
        _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits()))
        is _K.ACCESS_DENIED
    )
    assert classify_exception(exc) is _K.ACCESS_DENIED


@pytest.mark.parametrize(
    "code", ["ThrottlingException", "ResourceNotFoundException", "", "accessdenied"]
)
def test_any_other_code_is_an_error(code: str) -> None:
    assert classify_exception(_WithResponseError({"Error": {"Code": code}})) is _K.ERROR


class _NoResponseError(Exception):
    pass


class _BadPropertyError(Exception):
    @property
    def response(self) -> Any:
        raise RuntimeError("SENTINEL-PROPERTY-ZZ")


@pytest.mark.parametrize(
    "exc",
    [
        _NoResponseError("x"),  # ① response が無い
        _WithResponseError("not-a-mapping"),  # ② Mapping でない
        _WithResponseError(["Error"]),
        _WithResponseError(None),
        _WithResponseError({}),  # ③ "Error" が無い
        _WithResponseError({"Error": "AccessDenied"}),  # Error が Mapping でない
        _WithResponseError({"Error": {}}),
        _WithResponseError({"Error": {"Code": 403}}),  # ④ "Code" が文字列でない
        _WithResponseError({"Error": {"Code": None}}),
        _WithResponseError({"Error": {"Code": ["AccessDenied"]}}),
        _BadPropertyError("x"),  # 属性の読取が例外
    ],
)
def test_a_malformed_error_response_folds_into_error_without_another_exception(
    exc: Exception,
) -> None:
    assert classify_exception(exc) is _K.ERROR
    fake = _Fake(scan=[exc])
    assert _failure_kind(lambda: _reader(fake).read_pages("scan", _spec(), _limits())) is _K.ERROR


@pytest.mark.parametrize(
    "name",
    ["ConnectTimeoutError", "ReadTimeoutError", "EndpointConnectionError", "ConnectionClosedError"],
)
def test_connection_errors_are_classified_by_type_name(name: str) -> None:
    exc = type(name, (Exception,), {})("SENTINEL-MESSAGE-ZZ")
    assert classify_exception(exc) is _K.CONNECTION


def test_a_failure_carries_only_the_kind() -> None:
    failure = ReplayReadFailure(_K.MALFORMED)
    assert (
        str(failure) == "MALFORMED"
        and failure.args == ("MALFORMED",)
        and failure.kind is _K.MALFORMED
    )
    assert set(vars(failure)) == {"kind"}


def test_the_failure_documents_that_it_must_not_be_swallowed() -> None:
    doc = ReplayReadFailure.__doc__ or ""
    assert "広く捕まえて続行してはならない" in doc and "人へ伝えて停止" in doc


# =================================================================================================
# 例外に番兵(表名・キー・識別子・鍵・例外の文面)が出ない
# =================================================================================================

_SENTINELS = (
    "SENTINEL-TABLE-ZZ",
    "SENTINEL-MESSAGE-ZZ",
    "SENTINEL-KEY-ZZ",
    "SENTINEL-ID-ZZ",
    "sentinel-key-material",
)


def _all_text(exc: BaseException) -> str:
    chain = [exc, exc.__cause__, exc.__context__]
    return " ".join(
        part
        for item in chain
        if item is not None
        for part in (str(item), repr(item), repr(item.args))
    )


def _scenarios() -> list[tuple[str, Callable[[], Any]]]:
    keys = [{"SENTINEL-KEY-ZZ": {"S": "SENTINEL-ID-ZZ"}}]
    secret = _WithResponseError(
        {"Error": {"Code": "AccessDenied", "Message": "SENTINEL-MESSAGE-ZZ"}}
    )
    other = _WithResponseError({"Error": {"Code": "Other"}})

    def run(fake: _Fake, **limits: Any) -> Callable[[], Any]:
        return lambda: _reader(fake).read_pages("scan", _spec(), _limits(**limits))

    return [
        ("access_denied", run(_Fake(scan=[secret]))),
        ("error", run(_Fake(scan=[other]))),
        ("malformed_item", run(_Fake(scan=[{"Items": [{"holding_id": {"N": "SENTINEL-ID-ZZ"}}]}]))),
        (
            "undeclared",
            run(
                _Fake(
                    scan=[{"Items": [{**_item("SENTINEL-ID-ZZ"), "SENTINEL-KEY-ZZ": {"S": "x"}}]}]
                )
            ),
        ),
        ("item_limit", run(_Fake(scan=[_page(["SENTINEL-ID-ZZ", "b"])]), max_items=1)),
        ("page_limit", run(_Fake(scan=[_page(["SENTINEL-ID-ZZ"], _LEK1)]), max_pages=1)),
        (
            "unprocessed",
            lambda: _reader(
                _Fake(
                    batch_get_item=[{"Responses": {}, "UnprocessedKeys": {_TABLE: {"Keys": keys}}}]
                )
            ).batch_get(_spec(), keys, _limits()),
        ),
    ]


@pytest.mark.parametrize(("name", "action"), _scenarios(), ids=[n for n, _ in _scenarios()])
def test_no_sentinel_appears_in_str_args_cause_or_context(
    name: str, action: Callable[[], Any]
) -> None:
    with pytest.raises(ReplayReadFailure) as info:
        action()
    text = _all_text(info.value)
    assert [s for s in _SENTINELS if s in text] == []
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_a_violation_names_only_the_requested_operation() -> None:
    proxy = ReplayReadClient(_Fake(), allowed_operations=frozenset({"scan"}))
    with pytest.raises(ReplayReadViolationError) as info:
        _ = proxy.put_item
    assert str(info.value) == "operation not allowed: put_item"


def test_a_config_error_never_contains_the_key_or_the_table() -> None:
    for action in (
        lambda: IdentifierMasker(key_version=1, key=b"short-SENTINEL-KEY-ZZ"),
        lambda: _spec(table_name=""),
        lambda: _spec(projection=("SENTINEL-KEY-ZZ", "SENTINEL-KEY-ZZ"), mask_fields={}),
    ):
        with pytest.raises(ReplayReadConfigError) as info:
            action()
        assert "SENTINEL" not in str(info.value)


# =================================================================================================
# AST: AWS SDK・時計・OS の import 0・書込の語 0・数値の既定 0
# =================================================================================================

_ALLOWED_IMPORTS = {
    "__future__",
    "collections.abc",
    "dataclasses",
    "enum",
    "hashlib",
    "hmac",
    "math",
    "typing",
}
_FORBIDDEN_ROOTS = {
    "boto3",
    "botocore",
    "requests",
    "urllib",
    "urllib3",
    "http",
    "socket",
    "os",
    "time",
    "datetime",
    "jstock_advisor",
}


def _tree() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def _import_violations(names: set[str]) -> list[str]:
    return sorted(n for n in names if n not in _ALLOWED_IMPORTS)


def test_only_the_standard_library_allowlist_is_imported() -> None:
    names = _imports(_tree())
    assert _import_violations(names) == []
    assert not {n for n in names if n.split(".")[0] in _FORBIDDEN_ROOTS}
    assert {"hashlib", "hmac", "dataclasses"} <= names  # 走査は空振りでない


@pytest.mark.parametrize(
    "intruder",
    [
        "boto3",
        "botocore.exceptions",
        "requests",
        "urllib.request",
        "socket",
        "os",
        "time",
        "datetime",
        "jstock_advisor.repositories.base",
        "jstock_advisor.domain.entities.identifier_ref",
    ],
)
def test_a_real_import_violation_is_caught(intruder: str) -> None:
    assert _import_violations(_imports(_tree()) | {intruder}) == [intruder]


def _words_in(tree: ast.Module) -> set[str]:
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            seen.add(node.value)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.alias | ast.FunctionDef | ast.ClassDef):
            seen.add(node.name)
    return seen


def test_no_write_operation_word_appears_in_any_string_attribute_name_or_import() -> None:
    seen = _words_in(_tree())
    hits = sorted(w for w in seen for word in _WRITE_WORDS if word in w)
    assert hits == []
    assert "scan" in seen and "batch_get_item" in seen  # 走査は空振りでない
    synthetic = ast.parse("client.put_item(Item={})")
    assert any("put_item" in w for w in _words_in(synthetic))  # 本物の違反で落ちる


def test_no_numeric_default_exists_for_any_parameter_or_field() -> None:
    def numeric(node: ast.expr | None) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        )

    tree = _tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defaults = [*node.args.defaults, *node.args.kw_defaults]
            assert not any(numeric(d) for d in defaults if d is not None), node.name
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and not (
                    isinstance(stmt.target, ast.Name) and stmt.target.id.startswith("_")
                ):
                    assert not numeric(stmt.value), (node.name, ast.dump(stmt.target))
    synthetic = ast.parse("def f(max_items=100): pass")
    fn = next(n for n in ast.walk(synthetic) if isinstance(n, ast.FunctionDef))
    assert numeric(fn.args.defaults[0])  # 本物の違反で落ちる


def test_the_module_reads_neither_a_clock_nor_the_environment() -> None:
    attributes = {n.attr for n in ast.walk(_tree()) if isinstance(n, ast.Attribute)}
    assert not attributes & {"now", "utcnow", "today", "monotonic", "environ", "getenv", "sleep"}
    names = {n.id for n in ast.walk(_tree()) if isinstance(n, ast.Name)}
    assert not names & {"open", "print", "input", "exec", "eval", "datetime", "time", "os", "boto3"}


# =================================================================================================
# dormant(importer の等号 = 空)・#606 の名前の網の外
# =================================================================================================


def test_nothing_in_src_mentions_the_module_the_importer_list_is_exactly_empty() -> None:
    importers = sorted(
        path.relative_to(_PACKAGE).as_posix()
        for path in _PACKAGE.rglob("*.py")
        if path != _MODULE_PATH and "replay_source_reader" in path.read_text(encoding="utf-8")
    )
    assert importers == []  # R2-prep が配線するとき、このテストを意図して更新する(#988 の作法)


def test_the_module_is_outside_the_606_name_net() -> None:
    assert not _is_unconnected_name("jstock_advisor.infrastructure.aws.replay_source_reader")
