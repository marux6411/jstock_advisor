"""`services/replay_input_loader.py`(Issue #880 R1b)の契約テスト。

固定するもの: 数値は Decimal / int だけ(float を作らない)・必須キーの完全一致・未知の語の拒否・
範囲外の値と重複の拒否・事前登録の本体の名前の完全一致・★ 症例(held-out)は事前登録の指定から決め、
`Observation(` の構築点は 1 か所で `held_out=` に判定関数の戻り値を明示で渡す ★
(定数の直書きを持たない)。
fixture は合成のみ(`tests/support/replay_fixtures.py`)。
"""

from __future__ import annotations

import ast
import copy
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from jstock_advisor.domain.backtest.errors import ReplayInputError
from jstock_advisor.domain.backtest.outcome_window import OriginRule
from jstock_advisor.domain.backtest.replay_classification import ReplayAction
from jstock_advisor.domain.backtest.replay_events import Fire, PopulationExclusion
from jstock_advisor.services import replay_input_loader as loader
from tests.support import replay_fixtures as fx

_SRC = Path(__file__).resolve().parents[2] / "src" / "jstock_advisor"
_LOADER_PATH = _SRC / "services" / "replay_input_loader.py"
_REFS = frozenset({"h-case"})


def _prereg_text(**overrides: Any) -> str:
    return fx.to_json(fx.prereg_body(**overrides))


# =================================================================================================
# JSON の読み方(float を作らない)
# =================================================================================================


def test_numbers_are_decimal_or_int_never_float() -> None:
    value = loader.parse_json('{"a": 1, "b": 1.5, "c": 1e2, "d": [0.1, 2]}')
    assert isinstance(value, dict)
    assert isinstance(value["a"], int) and not isinstance(value["a"], bool)
    assert isinstance(value["b"], Decimal) and value["b"] == Decimal("1.5")
    assert isinstance(value["c"], Decimal)
    assert [type(x) for x in value["d"]] == [Decimal, int]


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", '{"x": NaN}', "[Infinity]"])
def test_non_finite_numbers_are_rejected(text: str) -> None:
    with pytest.raises(ReplayInputError):
        loader.parse_json(text)


@pytest.mark.parametrize("text", ["", "{", "{'a': 1}", "[1,]", "tru"])
def test_invalid_json_is_rejected_with_a_fixed_message(text: str) -> None:
    with pytest.raises(ReplayInputError, match="invalid JSON"):
        loader.parse_json(text)


def test_error_messages_never_contain_the_input_value() -> None:
    rows = fx.observation_rows()
    rows[0]["holding_ref"] = "SECRET-REF-ZZ"
    rows[0]["exclusion"] = "SECRET-WORD-ZZ"
    with pytest.raises(ReplayInputError) as info:
        loader.load_observations(fx.to_lines(rows), held_out_refs=_REFS, max_lines=100)
    assert "SECRET" not in str(info.value)


# =================================================================================================
# 事前登録の本体
# =================================================================================================


def test_the_registration_body_is_read_into_typed_values() -> None:
    body, params = loader.load_prereg(_prereg_text())
    assert body["registration_id"] == "REG-TEST-1"
    assert params.max_gap == 2 and params.min_cell == 2
    assert params.outcome_spec.windows == (1, 2)
    assert params.outcome_spec.origin is OriginRule.DECISION_DAY_CLOSE
    assert params.sale_cost_rate == Decimal("0.2")
    assert params.required_agreement == Fraction(1)
    assert params.reason_vocabulary == frozenset({"R_A", "R_B"})
    assert params.held_out_refs == frozenset({"h-case"})
    assert (params.max_input_lines, params.max_input_bytes, params.rate_digits) == (100, 100000, 2)


def test_the_required_parameter_names_are_exactly_the_registered_set() -> None:
    assert frozenset(fx.PARAMETERS) == loader.PARAMETER_NAMES  # 名前の完全一致


@pytest.mark.parametrize("name", sorted(fx.PARAMETERS))
def test_a_missing_registered_parameter_is_rejected_there_are_no_defaults(name: str) -> None:
    body = fx.prereg_body()
    body["parameters"] = [p for p in body["parameters"] if p["name"] != name]
    with pytest.raises(ReplayInputError):
        loader.load_prereg(fx.to_json(body))


def test_an_unknown_or_duplicate_parameter_is_rejected() -> None:
    body = fx.prereg_body()
    body["parameters"].append({"name": "extra_value", "value": 1})
    with pytest.raises(ReplayInputError):
        loader.load_prereg(fx.to_json(body))
    body = fx.prereg_body()
    body["parameters"].append({"name": "max_gap", "value": 3})
    with pytest.raises(ReplayInputError, match="duplicate"):
        loader.load_prereg(fx.to_json(body))


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_gap": 0},
        {"max_gap": True},
        {"max_gap": "2"},
        {"min_cell": 0},
        {"outcome_windows": []},
        {"outcome_windows": [2, 1]},
        {"outcome_windows": [1, 1]},
        {"outcome_windows": ["1"]},
        {"outcome_origin": "SOMEDAY"},
        {"sale_cost_rate": 0.2},  # 本体の小数は文字列(数値の小数は拒否)
        {"sale_cost_rate": "1"},
        {"sale_cost_rate": "-0.1"},
        {"sale_cost_rate": "abc"},
        {"required_agreement": "2"},
        {"required_agreement": 1},
        {"required_agreement": "x/y"},
        {"reason_vocabulary": "R_A"},
        {"reason_vocabulary": [""]},
        {"held_out_refs": [1]},
        {"max_input_lines": 0},
        {"rate_digits": -1},
        {"rate_digits": 13},
    ],
)
def test_out_of_range_or_wrongly_typed_registered_values_are_rejected(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ReplayInputError):
        loader.load_prereg(_prereg_text(**overrides))


def test_a_float_in_the_body_is_rejected_because_the_body_holds_json_types_only() -> None:
    with pytest.raises(ReplayInputError):
        loader.load_prereg(_prereg_text().replace('"0.2"', "0.2"))


def test_the_citation_needs_exactly_its_three_keys() -> None:
    body = fx.prereg_body()
    cite = fx.citation_for(body)
    assert loader.load_citation(fx.to_json(cite)).registration_id == "REG-TEST-1"
    for key in ("registration_id", "content_hash", "used"):
        broken = {k: v for k, v in cite.items() if k != key}
        with pytest.raises(ReplayInputError):
            loader.load_citation(fx.to_json(broken))
    with pytest.raises(ReplayInputError):
        loader.load_citation(fx.to_json({**cite, "extra": 1}))


# =================================================================================================
# 観測 — 症例(held-out)は登録された指定から決める
# =================================================================================================


def test_observations_are_read_and_the_held_out_case_comes_from_the_registered_refs() -> None:
    rows = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=_REFS, max_lines=100
    )
    assert len(rows) == 8
    held = [row.holding_ref for row in rows if row.held_out]
    assert held == ["h-case"]
    excluded = {row.holding_ref: row.exclusion for row in rows if row.exclusion is not None}
    assert excluded == {
        "h3": PopulationExclusion.LEGACY,
        "h4": PopulationExclusion.DATA_INSUFFICIENT,
    }


def test_with_no_registered_refs_nobody_is_a_case() -> None:
    rows = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=frozenset(), max_lines=100
    )
    assert not any(row.held_out for row in rows)


def test_a_registered_ref_becomes_a_case_even_when_the_file_does_not_say_so() -> None:
    """症例かどうかはファイルの値ではなく登録の指定で決まる(ファイルに held_out のキーは無い)。"""
    assert "held_out" not in loader._OBSERVATION_KEYS
    rows = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=frozenset({"h1"}), max_lines=100
    )
    assert [r.holding_ref for r in rows if r.held_out] == ["h1", "h1", "h1"]


def test_is_held_out_is_membership_in_the_registered_refs() -> None:
    assert loader.is_held_out("a", frozenset({"a"})) is True
    assert loader.is_held_out("b", frozenset({"a"})) is False


def test_the_observation_construction_site_is_unique_and_passes_the_held_out_decision() -> None:
    """`Observation(` の構築点の census: 1 か所・`held_out=` 明示・値は判定関数の戻り値。"""
    tree = ast.parse(_LOADER_PATH.read_text(encoding="utf-8"))
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Observation"
    ]
    assert len(sites) == 1  # 空振りでない + 構築点は 1 か所
    keywords = {kw.arg: kw.value for kw in sites[0].keywords}
    assert "held_out" in keywords  # 省略に落ちない
    value = keywords["held_out"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name) and value.func.id == "is_held_out"
    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, bool) for n in [value])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r[0].pop("phase_kind"),
        lambda r: r[0].update(extra=1),
        lambda r: r[0].update(held_out=True),  # 症例の指定をファイルで持てない
        lambda r: r[0].update(exclusion="NOT_A_REASON"),
        lambda r: r[0].update(day_index=-1),
        lambda r: r[0].update(day_index=True),
        lambda r: r[0].update(day_index="10"),
        lambda r: r[0].update(day_index=1.5),
        lambda r: r[0].update(holding_ref=""),
        lambda r: r[0].update(phase_key=""),
        lambda r: r[0].update(is_first_evaluation="yes"),
        lambda r: r.append(copy.deepcopy(r[0])),  # 重複
    ],
)
def test_malformed_observation_rows_are_rejected(mutate: Any) -> None:
    rows = fx.observation_rows()
    mutate(rows)
    with pytest.raises(ReplayInputError):
        loader.load_observations(fx.to_lines(rows), held_out_refs=_REFS, max_lines=100)


def test_the_line_limit_is_a_parameter_and_is_enforced() -> None:
    text = fx.to_lines(fx.observation_rows())
    assert len(loader.load_observations(text, held_out_refs=_REFS, max_lines=8)) == 8
    with pytest.raises(ReplayInputError, match="too many"):
        loader.load_observations(text, held_out_refs=_REFS, max_lines=7)


def test_blank_lines_are_ignored_and_a_non_object_line_is_rejected() -> None:
    text = "\n" + fx.to_lines(fx.observation_rows()) + "\n\n"
    assert len(loader.load_observations(text, held_out_refs=_REFS, max_lines=100)) == 8
    with pytest.raises(ReplayInputError):
        loader.load_observations("[1, 2]\n", held_out_refs=_REFS, max_lines=10)


# =================================================================================================
# 対(OLD と NEW の判断。入力のファイルで受ける)
# =================================================================================================


def test_pairs_are_read_into_judgements_and_decimals() -> None:
    pairs = loader.load_pairs(fx.to_lines(fx.pair_rows()), max_lines=100)
    assert len(pairs) == 8
    second = pairs[1]
    assert second.old.action is ReplayAction.FULL and second.new.action is ReplayAction.PARTIAL
    assert second.new_sold_ratio == Decimal("0.5") and second.old_sold_ratio == Decimal(1)
    assert second.reason_code == "R_A"
    assert second.fires == (Fire.FIRED, Fire.NOT_FIRED)
    assert second.prior_high == Decimal("100.5")
    assert pairs[3].new.undetermined is True and pairs[3].new.action is ReplayAction.HOLD
    assert all(isinstance(p.prior_high, Decimal) for p in pairs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r[0].pop("prior_high"),
        lambda r: r[0].update(extra=1),
        lambda r: r[0]["old"].pop("sold_ratio"),
        lambda r: r[0]["new"].update(extra=1),
        lambda r: r[0]["old"].update(action="SELL"),
        lambda r: r[0]["new"].update(action="BUY"),
        lambda r: r[0]["old"].update(sold_ratio="0.5"),  # HOLD は何も売らない
        lambda r: r[1]["old"].update(sold_ratio="0.5"),  # FULL は全部
        lambda r: r[1]["new"].update(sold_ratio="1"),  # PARTIAL は一部(0 < x < 1)
        lambda r: r[1]["new"].update(sold_ratio="0"),
        lambda r: r[1]["new"].update(sold_ratio=True),
        lambda r: r[3]["new"].update(action="FULL", sold_ratio="1"),  # 判定不能は HOLD
        lambda r: r[0].update(prior_high="0"),
        lambda r: r[0].update(prior_high="-1"),
        lambda r: r[0].update(prior_high="x"),
        lambda r: r[0].update(fire_first="MAYBE"),
        lambda r: r[0].update(fire_second=None),
        lambda r: r[0].update(recorded=""),
        lambda r: r[0].update(recomputed=1),
        lambda r: r[0]["new"].update(reason_code=""),
        lambda r: r[0].update(old="x"),
        lambda r: r.append(copy.deepcopy(r[0])),
    ],
)
def test_malformed_pair_rows_are_rejected(mutate: Any) -> None:
    rows = fx.pair_rows()
    mutate(rows)
    with pytest.raises(ReplayInputError):
        loader.load_pairs(fx.to_lines(rows), max_lines=100)


def test_a_pair_for_an_unknown_observation_is_rejected() -> None:
    observations = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=_REFS, max_lines=100
    )
    rows = fx.pair_rows()
    loader.check_references(
        observations, loader.load_pairs(fx.to_lines(rows), max_lines=100)
    )  # 正常
    rows[0]["day_index"] = 999
    with pytest.raises(ReplayInputError, match="unknown observation"):
        loader.check_references(observations, loader.load_pairs(fx.to_lines(rows), max_lines=100))


# =================================================================================================
# 価格
# =================================================================================================


def test_price_series_are_read_as_decimal_bars() -> None:
    series = loader.load_prices(fx.to_lines(fx.price_rows()), max_lines=100)
    assert set(series) == {"h1", "h2", "h3", "h-case", "h4"}
    assert series["h1"].first_day_index == 10 and len(series["h1"].bars) == 14
    assert series["h1"].bars[0].open == Decimal(100)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r[0].pop("bars"),
        lambda r: r[0].update(bars=[]),
        lambda r: r[0].update(bars="x"),
        lambda r: r[0]["bars"][0].pop("low"),
        lambda r: r[0]["bars"][0].update(extra=1),
        lambda r: r[0]["bars"][0].update(low="200"),  # low > high
        lambda r: r[0]["bars"][0].update(open="0"),
        lambda r: r[0].update(first_day_index=-1),
        lambda r: r.append(copy.deepcopy(r[0])),
    ],
)
def test_malformed_price_rows_are_rejected(mutate: Any) -> None:
    rows = fx.price_rows()
    mutate(rows)
    with pytest.raises(ReplayInputError):
        loader.load_prices(fx.to_lines(rows), max_lines=100)


# =================================================================================================
# 静的な契約(float を作らない・丸めない)
# =================================================================================================


def _tree() -> ast.Module:
    return ast.parse(_LOADER_PATH.read_text(encoding="utf-8"))


def test_the_loader_never_builds_a_float_or_rounds() -> None:
    tree = _tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            assert not (isinstance(func, ast.Name) and func.id in {"float", "round", "int"}), (
                "loader builds a float / rounds / coerces"
            )
            assert not (isinstance(func, ast.Attribute) and func.attr in {"quantize", "__round__"})
        if isinstance(node, ast.Constant):
            assert not isinstance(node.value, float), node.value


def test_json_is_read_with_parse_float_decimal_and_parse_int_explicit() -> None:
    calls = [
        n
        for n in ast.walk(_tree())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "loads"
    ]
    assert len(calls) == 1  # 読む入口は 1 か所
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords["parse_float"], ast.Name) and keywords["parse_float"].id == "Decimal"
    assert isinstance(keywords["parse_int"], ast.Name) and keywords["parse_int"].id == "int"
    assert "parse_constant" in keywords  # NaN / Infinity を拒否


def test_the_loader_has_no_numeric_default_for_any_parameter() -> None:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef):
            defaults = [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]
            for default in defaults:
                assert not (
                    isinstance(default, ast.Constant)
                    and isinstance(default.value, int | float)
                    and not isinstance(default.value, bool)
                ), node.name


def test_the_decimal_conversion_accepts_only_numbers_and_decimal_strings() -> None:
    assert loader.to_decimal(3) == Decimal(3)
    assert loader.to_decimal("1.25") == Decimal("1.25")
    assert loader.to_decimal(Decimal("2.5")) == Decimal("2.5")
    for bad in (True, None, [], "abc", "NaN", "Infinity", Decimal("NaN"), 1.5):
        with pytest.raises(ReplayInputError):
            loader.to_decimal(bad)


def _error_messages(path: Path) -> list[ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ReplayInputError"
        and node.args
    ]


@pytest.mark.parametrize(
    "relative",
    ["services/replay_input_loader.py", "services/replay_report.py", "cli/replay.py"],
)
def test_every_error_message_in_the_replay_layer_is_a_fixed_string(relative: str) -> None:
    """入力の値を例外の文言へ入れない: 文言は定数の文字列だけ(f-string・連結・変数を使わない)。"""
    messages = _error_messages(_SRC / relative)
    assert [
        m for m in messages if not (isinstance(m, ast.Constant) and isinstance(m.value, str))
    ] == []


def test_the_error_message_scan_is_not_vacuous() -> None:
    assert len(_error_messages(_LOADER_PATH)) >= 20
    synthetic = ast.parse('raise ReplayInputError(f"bad {value}")')
    call = next(n for n in ast.walk(synthetic) if isinstance(n, ast.Call))
    assert not isinstance(call.args[0], ast.Constant)  # 値を埋めた文言は、固定文字列の検査で落ちる


def test_the_domain_messages_interpolate_only_names_and_limits() -> None:
    """R1a の検査関数の f-string は、引数の『名前』と定数の下限だけを埋める(入力の値は入れない)。"""
    backtest = _SRC / "domain" / "backtest"
    interpolated: set[str] = set()
    for path in sorted(backtest.glob("*.py")):
        for message in _error_messages(path):
            if isinstance(message, ast.JoinedStr):
                for part in message.values:
                    if isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
                        interpolated.add(part.value.id)
    assert interpolated <= {"name", "minimum"}
