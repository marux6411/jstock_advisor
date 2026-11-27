"""Issue #989: 資金配分 Shadow の CLI の表示で、10 進数の欄が読めない値を素通しせず『不明』へ畳み、
その件数を `unparsed` に数える(案 B。USER 判断 = #122 …6280435404)。

背景: 金額・RAER 値の欄は、従来『読めない文字列』をそのまま画面へ出し(NaN / Infinity も同様)、
unparsed にも数えなかった。記録の中身が画面へ素通しになり、『数えない日』の判定(R1 の営業日の数え)
からも漏れていた。

固定するもの:
  ① 判定は `classify_decimal_value` の 1 本。表示が『不明』⇔ 記述が 1 件(標本の表で両方向)
  ② 5 つの欄(買付余力・評価額・配分額・不足額・RAER 値)で、読めない値は『不明』+ 場所つき 1 件
  ③ `unparsed - len(unreadable)` = 理由コード + 確信度 + owner 単位の不読(内訳の式)
  ④ 値そのものは、表・JSON・repr のどこにも出ない(場所・種別・長さだけ)
  ⑤ census: 表示面で `DecimalTally` を呼ぶのは 5 か所。classify の本体の呼び出しは 2 か所だけ。
     detail 側(3 か所)は、形の検査(`_DEC`)を通ったものだけを format へ渡し、その形は classify で
     必ず『読める』(同値)
  ⑥ 読める値の出力は従来と同じ(桁区切り・丸めない・数値は repr)
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pytest

from jstock_advisor.domain.notification import allocation_shadow_cli_view as view
from jstock_advisor.domain.notification.allocation_shadow_cli_renderer import (
    render_json,
    render_table,
)
from jstock_advisor.domain.notification.allocation_shadow_cli_view import (
    UNKNOWN_TEXT,
    DayInput,
    DecimalTally,
    OwnerStatus,
    RecordInput,
    RecordState,
    UnreadableKind,
    build_screen,
    classify_decimal_value,
    format_decimal,
    format_yen,
)
from tests.support import allocation_synthetic_days as syn

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "jstock_advisor"
_VIEW_PATH = _SRC / "domain" / "notification" / "allocation_shadow_cli_view.py"
_DAY = "2026-10-14"
_SENTINEL = "SENTINEL-989-raw-value"
MISSING = RecordInput(RecordState.NOT_FOUND)

K = UnreadableKind

# --- ① 標本の表(値, 表示, 種別 or None, 長さ)----------------------------------------------------
# 種別 None = 読める(件数に数えない)。
_READABLE: list[tuple[Any, str]] = [
    ("1000", "1,000"),
    ("-3.25", "-3.25"),
    ("007", "7"),
    ("1" * 30, "-"),  # 桁数の多い値も読める(表示の中身は見ない)
    ("1e5", "100,000"),  # 様式外だが Decimal() では読める
    (" 12 ", "12"),  # 空白つきも Decimal() では読める
    (12, "12"),
    (12.5, "12.5"),
]
_UNREADABLE: list[tuple[Any, UnreadableKind, int | None]] = [
    ("abc", K.NOT_DECIMAL_STRING, 3),
    ("", K.NOT_DECIMAL_STRING, 0),
    ("1,000", K.NOT_DECIMAL_STRING, 5),
    ("1 000", K.NOT_DECIMAL_STRING, 5),
    (_SENTINEL, K.NOT_DECIMAL_STRING, len(_SENTINEL)),
    ("NaN", K.NON_FINITE_STRING, 3),
    ("Infinity", K.NON_FINITE_STRING, 8),
    ("-Infinity", K.NON_FINITE_STRING, 9),
    ("sNaN", K.NON_FINITE_STRING, 4),
    (math.nan, K.NON_FINITE_NUMBER, None),
    (math.inf, K.NON_FINITE_NUMBER, None),
    (-math.inf, K.NON_FINITE_NUMBER, None),
    ({"a": 1}, K.OPAQUE_STRUCTURE, None),
    ([1], K.OPAQUE_STRUCTURE, None),
    (True, K.OPAQUE_STRUCTURE, None),
    (False, K.OPAQUE_STRUCTURE, None),
]


def _ids(samples: list[Any]) -> list[str]:
    return [repr(s[0])[:24] for s in samples]


@pytest.mark.parametrize("sample", _READABLE, ids=_ids(_READABLE))
def test_a_readable_value_is_shown_and_not_counted(sample: tuple[Any, str]) -> None:
    value, _ = sample
    text, bad = classify_decimal_value(value)

    assert bad is None
    assert text is not None
    assert text != UNKNOWN_TEXT


@pytest.mark.parametrize("sample", _UNREADABLE, ids=_ids(_UNREADABLE))
def test_an_unreadable_value_is_folded_to_unknown_and_counted_once_with_its_kind(
    sample: tuple[Any, UnreadableKind, int | None],
) -> None:
    value, kind, length = sample
    text, bad = classify_decimal_value(value)

    assert text == UNKNOWN_TEXT
    assert bad == (kind, length)


def test_none_is_an_absent_field_not_shown_and_not_counted() -> None:
    assert classify_decimal_value(None) == (None, None)


def test_display_is_unknown_if_and_only_if_an_unreadable_is_reported() -> None:
    """★ 表示と件数は同じ判断(両方向)★ 標本の全体で『不明 ⇔ 記述が 1 件』。"""
    samples = [v for v, _ in _READABLE] + [v for v, _, _ in _UNREADABLE] + [None]
    for value in samples:
        text, bad = classify_decimal_value(value)
        assert (text == UNKNOWN_TEXT) == (bad is not None), repr(value)
        tally = DecimalTally()
        shown = tally.decimal("x", value)
        assert shown == text
        assert len(tally.items) == (0 if bad is None else 1)
        yen = DecimalTally().yen("y", value)
        assert yen == (None if text is None else text if bad else f"{text}円")


def test_the_readable_formats_are_unchanged_from_before() -> None:
    assert format_decimal("1234567.89") == "1,234,567.89"
    assert format_decimal("-1234.5000") == "-1,234.5000"  # 丸めない
    assert format_decimal("0") == "0"
    assert format_decimal(12.5) == "12.5"
    assert format_decimal(7) == "7"
    assert format_yen("5000000") == "5,000,000円"
    assert format_yen(None) is None


def test_the_unit_is_not_attached_to_the_unknown_marker() -> None:
    assert format_yen("abc") == UNKNOWN_TEXT
    assert format_yen({"a": 1}) == UNKNOWN_TEXT
    assert DecimalTally().yen("x", "NaN") == UNKNOWN_TEXT


# --- ② 5 つの欄 --------------------------------------------------------------------------------


def _day(record: dict[str, Any]) -> DayInput:
    owner = "owner-a"
    return DayInput(
        _DAY,
        RecordInput(RecordState.FOUND, syn.index_record(_DAY, [owner])),
        MISSING,
        {syn.ref(owner): RecordInput(RecordState.FOUND, record)},
    )


def _record(**kw: Any) -> dict[str, Any]:
    return syn.owner_record(_DAY, "owner-a", **kw)


def _set(record: dict[str, Any], location: str, value: Any) -> None:
    first = record["candidates"][0]
    if location == "cash.available_cash":
        record["cash"]["available_cash"] = value
    elif location == "portfolio.portfolio_value":
        record["portfolio"]["portfolio_value"] = value
    elif location == "candidates[0].allocation.amount":
        first["allocation"]["amount"] = value
    elif location == "candidates[0].allocation.cash_shortfall":
        first["allocation"]["cash_shortfall"] = value
    elif location == "candidates[0].raer_out.value":
        first["raer_out"]["value"] = value
    else:  # pragma: no cover
        raise AssertionError(location)


_LOCATIONS = [
    "cash.available_cash",
    "portfolio.portfolio_value",
    "candidates[0].allocation.amount",
    "candidates[0].allocation.cash_shortfall",
    "candidates[0].raer_out.value",
]


def _shown(owner: view.OwnerSection, location: str) -> str | None:
    row = owner.rows[0]
    return {
        "cash.available_cash": owner.available_cash,
        "portfolio.portfolio_value": owner.portfolio_value,
        "candidates[0].allocation.amount": row.required_funds,
        "candidates[0].allocation.cash_shortfall": row.shortfall,
        "candidates[0].raer_out.value": row.raer_value,
    }[location]


def _owner(record: dict[str, Any]) -> view.OwnerSection:
    screen = build_screen(_day(record))
    assert screen.owners[0].status is OwnerStatus.OK
    return screen.owners[0]


def test_the_baseline_has_nothing_unreadable() -> None:
    owner = _owner(_record())

    assert owner.unreadable == ()
    assert owner.unparsed == 0


@pytest.mark.parametrize("location", _LOCATIONS)
@pytest.mark.parametrize("sample", _UNREADABLE, ids=_ids(_UNREADABLE))
def test_an_unreadable_value_in_any_of_the_five_fields_is_unknown_and_counted_with_its_place(
    location: str, sample: tuple[Any, UnreadableKind, int | None]
) -> None:
    value, kind, length = sample
    record = _record()
    _set(record, location, value)

    owner = _owner(record)

    assert [(u.location, u.kind, u.length) for u in owner.unreadable] == [(location, kind, length)]
    assert owner.unparsed == 1
    shown = _shown(owner, location)
    # ★ 生の値は出さず『不明』を出す(不足額は『0 以下なら出さない』判断があるが、読めなければ出す)★
    assert shown == UNKNOWN_TEXT


@pytest.mark.parametrize("location", _LOCATIONS)
@pytest.mark.parametrize("sample", _READABLE, ids=_ids(_READABLE))
def test_a_readable_value_in_any_of_the_five_fields_is_not_counted(
    location: str, sample: tuple[Any, str]
) -> None:
    value, _ = sample
    record = _record()
    _set(record, location, value)

    owner = _owner(record)

    assert owner.unreadable == ()
    assert owner.unparsed == 0


def test_display_unknown_iff_counted_at_the_screen_level_for_every_sample_and_field() -> None:
    """① の同値を、画面モデルまで通した形で固定する(判断が 2 つに割れていないこと)。"""
    for location in _LOCATIONS:
        for value in [v for v, _ in _READABLE] + [v for v, _, _ in _UNREADABLE]:
            record = _record()
            _set(record, location, value)
            owner = _owner(record)
            shown = _shown(owner, location)
            assert (shown == UNKNOWN_TEXT) == (len(owner.unreadable) == 1), (location, repr(value))


def test_an_unreadable_amount_on_a_not_allocated_candidate_is_counted_but_not_shown() -> None:
    """配分されなかった候補の金額は表示しない(従来どおり)が、読めない値は数える(従来も数えていた)。"""
    record = _record(
        candidates=[
            syn.candidate(1, decision="NOT_ALLOCATED", reason="INSUFFICIENT_CASH_REMAINING")
        ]
    )
    record["candidates"][0]["allocation"]["amount"] = "abc"

    owner = _owner(record)

    assert [u.location for u in owner.unreadable] == ["candidates[0].allocation.amount"]
    assert owner.unparsed == 1
    assert owner.rows[0].required_funds != "abc円"


def test_several_unreadable_fields_are_all_listed_in_record_order() -> None:
    record = _record(candidates=[syn.candidate(1), syn.candidate(2)])
    record["cash"]["available_cash"] = "abc"
    record["candidates"][1]["raer_out"]["value"] = "NaN"
    record["candidates"][0]["allocation"]["amount"] = [1]

    owner = _owner(record)

    assert [(u.location, u.kind) for u in owner.unreadable] == [
        ("cash.available_cash", K.NOT_DECIMAL_STRING),
        ("candidates[0].allocation.amount", K.OPAQUE_STRUCTURE),
        ("candidates[1].raer_out.value", K.NON_FINITE_STRING),
    ]
    assert owner.unparsed == 3


def test_an_unreadable_value_makes_the_day_incomplete() -> None:
    record = _record()
    record["portfolio"]["portfolio_value"] = "NaN"

    screen = build_screen(_day(record))

    assert screen.complete is False
    assert screen.unparsed == 1


# --- ③ unparsed の内訳の式 -----------------------------------------------------------------------


def test_unparsed_minus_unreadable_is_the_other_kinds_of_unparsed() -> None:
    """`unparsed − len(unreadable)` = 理由コード + 確信度 + owner 単位の不読。"""
    record = _record()
    record["cash"]["available_cash"] = "abc"  # 10 進数の欄 1
    record["candidates"][0]["allocation"]["amount"] = "NaN"  # 10 進数の欄 2
    record["run"]["outcome"] = "SOMETHING-NEW"  # 理由コード 1
    record["candidates"][0]["raer_out"]["confidence"] = "SOMETHING-NEW"  # 確信度 1

    owner = _owner(record)

    assert len(owner.unreadable) == 2
    assert owner.unparsed - len(owner.unreadable) == 2


def test_an_owner_that_could_not_be_read_has_no_decimal_breakdown() -> None:
    refused = view._owner("ref-x", None, {})
    assert refused.unparsed == 1
    assert refused.unreadable == ()


# --- ④ 値は出ない ---------------------------------------------------------------------------------


@pytest.mark.parametrize("location", _LOCATIONS)
def test_the_raw_value_never_reaches_the_table_the_json_or_the_repr(location: str) -> None:
    record = _record()
    _set(record, location, _SENTINEL)
    screen = build_screen(_day(record))

    for text in (render_table(screen), render_json(screen), repr(screen)):
        assert _SENTINEL not in text


def test_the_table_lists_place_kind_and_length_only() -> None:
    record = _record()
    record["cash"]["available_cash"] = _SENTINEL
    table = render_table(build_screen(_day(record)))

    assert "cash.available_cash" in table
    assert "10 進数として読めない文字列" in table
    assert f"{len(_SENTINEL)} 文字" in table


def test_the_table_caps_the_listing_and_points_to_the_json() -> None:
    cands = [syn.candidate(n) for n in range(1, 14)]
    record = _record(candidates=cands)
    for cand in record["candidates"]:
        cand["raer_out"]["value"] = "abc"

    owner = _owner(record)
    table = render_table(build_screen(_day(record)))
    payload = json.loads(render_json(build_screen(_day(record))))

    assert len(owner.unreadable) == 13
    assert "ほか 3 件" in table
    assert len(payload["owners"][0]["unreadable"]) == 13


def test_the_json_has_the_unreadable_key_only_when_something_is_unreadable() -> None:
    clean = json.loads(render_json(build_screen(_day(_record()))))
    assert "unreadable" not in clean["owners"][0]  # 無い日の出力は従来と同じ

    record = _record()
    record["cash"]["available_cash"] = "abc"
    dirty = json.loads(render_json(build_screen(_day(record))))
    assert dirty["owners"][0]["unreadable"] == [
        {"location": "cash.available_cash", "kind": "NOT_DECIMAL_STRING", "length": 3}
    ]


# --- ⑤ census ----------------------------------------------------------------------------------


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if callee == name:
                found.append(node)
    return found


def test_census_the_display_sites_that_count_are_exactly_the_five_fields() -> None:
    tree = ast.parse(_VIEW_PATH.read_text(encoding="utf-8"))
    sites = [
        c
        for c in _calls(tree, "yen") + _calls(tree, "decimal")
        if isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "tally"
    ]
    locations = sorted(
        ast.unparse(c.args[0])
        for c in sites
        if c.args  # 場所(固定の語彙。f-string を含む)
    )

    assert len(sites) == 5, locations
    assert any("cash.available_cash" in loc for loc in locations)
    assert any("portfolio.portfolio_value" in loc for loc in locations)
    assert any("allocation.amount" in loc for loc in locations)
    assert any("allocation.cash_shortfall" in loc for loc in locations)
    assert any("raer_out.value" in loc for loc in locations)


def test_census_classify_is_called_from_the_wrapper_and_the_tally_only() -> None:
    tree = ast.parse(_VIEW_PATH.read_text(encoding="utf-8"))
    owners: list[str] = []

    class Finder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _scope(self, node: ast.AST, name: str) -> None:
            self.stack.append(name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scope(node, node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._scope(node, node.name)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id == "classify_decimal_value":
                owners.append(".".join(self.stack))
            self.generic_visit(node)

    Finder().visit(tree)

    assert sorted(owners) == ["DecimalTally.decimal", "format_decimal"]


def test_census_the_direct_users_of_format_decimal_and_yen_are_pinned() -> None:
    """src のうち format_decimal / format_yen を直接呼ぶ場所を固定する。

    cli_view: コードが自分で組み立てた有限の値(残余・単価 × 株数)の整形
    detail_view: 形の検査(`_DEC`)を通ったものだけを渡す(その形が必ず『読める』ことは下で固定)
    """
    users: dict[str, int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if path == _VIEW_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        n = len(_calls(tree, "format_decimal")) + len(_calls(tree, "format_yen"))
        if n:
            users[path.relative_to(_SRC).as_posix()] = n

    assert users == {"domain/notification/allocation_shadow_detail_view.py": 3}


def test_every_string_the_detail_view_lets_through_is_readable_by_classify() -> None:
    from jstock_advisor.domain.notification.allocation_shadow_detail_view import _DEC

    rng = random.Random(989)
    samples = ["0", "-0", "1", "-1.5", "0.0", "9" * 30, "9" * 30 + "." + "9" * 30]
    for _ in range(500):
        whole = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 30)))
        frac = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 30)))
        sign = rng.choice(["", "-"])
        samples.append(f"{sign}{whole}" if rng.random() < 0.5 else f"{sign}{whole}.{frac}")
    for sample in samples:
        assert re.fullmatch(_DEC.pattern, sample), sample
        text, bad = classify_decimal_value(sample)
        assert bad is None, sample
        assert text is not None and text != UNKNOWN_TEXT


def test_the_old_opaque_counter_is_gone() -> None:
    assert not hasattr(view, "_opaque_values")


def test_census_cli_view_calls_format_yen_directly_only_for_values_the_code_built() -> None:
    """cli_view 自身の `format_decimal` / `format_yen` の直接呼び出しを固定する(審査の SHOULD-1)。

    ① 包みの本体(`format_yen` の中の `format_decimal`)を除いて、ちょうど 3 件
    ② 3 件の引数はいずれも `format(...)` の呼び出し = コードが組み立てた値(記録の取り出しではない)

    記録由来の欄を `format_yen` / `format_decimal` で出すと、『畳むのに数えない』状態が通る。
    新しい欄は `DecimalTally` を通すこと。
    """
    tree = ast.parse(_VIEW_PATH.read_text(encoding="utf-8"))
    found: list[tuple[str, ast.Call]] = []

    class Finder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("format_decimal", "format_yen"):
                found.append((self.stack[-1] if self.stack else "", node))
            self.generic_visit(node)

    Finder().visit(tree)
    outside_wrappers = [call for owner, call in found if owner not in ("format_yen",)]

    assert len(outside_wrappers) == 3, [ast.unparse(c) for c in outside_wrappers]
    for call in outside_wrappers:
        argument = call.args[0]
        assert isinstance(argument, ast.Call), ast.unparse(call)
        assert isinstance(argument.func, ast.Name) and argument.func.id == "format", ast.unparse(
            call
        )
