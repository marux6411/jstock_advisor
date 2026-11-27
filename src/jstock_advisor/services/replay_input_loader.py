"""replay の入力(ローカルのファイル)を、R1a の型へ読み込む(Issue #880 R1b。F-33)。

`jstock replay` が読む 4 種類のファイルを、`domain/backtest/` の型へ変える純粋な変換:

  事前登録の本体    JSON。`{"registration_id": str, "parameters": [{"name", "value"}, ...]}`
  観測              JSON Lines。1 行 = (保有, 評価日)の 1 行
  対                JSON Lines。1 行 = OLD と NEW の判断の 1 対 + 転帰・同時発火・再現性の入力
  価格              JSON Lines。1 行 = 1 つの保有の価格の列

## 守ること(契約テストで固定)

* import は標準ライブラリと `jstock_advisor.domain.backtest` だけ(AST の allowlist)。本番の型・
  boto3 などは持ち込まない(#606 の CLI 側の網を動かさない)
* ★ 数値は Decimal / int だけ ★。JSON は `parse_float=Decimal` と `parse_int=int` を明示して読み、
  float を作らない。NaN / Infinity は拒否。事前登録の本体の小数は文字列で持つ
  (本体の型は JSON の型だけ)
* ★ 値の既定を持たない ★。必須キーの欠落・未知のキー・未知の語・範囲外の値・重複は
  ReplayInputError(固定の文言。値・識別子を含めない)
* ★ 症例(held-out)は、事前登録の本体の指定から決める ★。`Observation(` の構築点は 1 か所で、
  `held_out=` に `is_held_out(...)` の戻り値を明示で渡す(定数の直書きを持たない)
* OLD / NEW の判断は入力のファイルで受け取る。判断を作り直さない・本番の判定の module を
  import しない
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from jstock_advisor.domain.backtest.errors import ReplayInputError
from jstock_advisor.domain.backtest.outcome_window import Bar, OriginRule, OutcomeSpec
from jstock_advisor.domain.backtest.replay_classification import Judgement, ReplayAction
from jstock_advisor.domain.backtest.replay_events import Fire, Observation, PopulationExclusion

#: 事前登録の本体の名前の完全なリスト(これより多くても少なくても拒否)
PARAMETER_NAMES: frozenset[str] = frozenset(
    {
        "max_gap",
        "min_cell",
        "outcome_windows",
        "outcome_origin",
        "sale_cost_rate",
        "required_agreement",
        "reason_vocabulary",
        "held_out_refs",
        "max_input_lines",
        "max_input_bytes",
        "rate_digits",
    }
)

#: 事前登録の本体の大きさの上限(本体を読む前の固定の安全弁。解析の値ではない)
PREREG_MAX_BYTES = 262144

_OBSERVATION_KEYS = frozenset(
    {
        "holding_ref",
        "day_index",
        "phase_kind",
        "phase_key",
        "exclusion",
        "is_first_evaluation",
    }
)
_OLD_KEYS = frozenset({"action", "exit_class", "sold_ratio"})
_NEW_KEYS = frozenset({"action", "exit_class", "undetermined", "sold_ratio", "reason_code"})
_PAIR_KEYS = frozenset(
    {
        "holding_ref",
        "day_index",
        "old",
        "new",
        "fire_first",
        "fire_second",
        "recorded",
        "recomputed",
        "prior_high",
    }
)
_SERIES_KEYS = frozenset({"holding_ref", "first_day_index", "bars"})
_BAR_KEYS = frozenset({"open", "high", "low", "close"})
_CITE_KEYS = frozenset({"registration_id", "content_hash", "used"})

_ZERO = Decimal(0)
_ONE = Decimal(1)


# --- JSON の読み方(float を作らない)----------------------------------------------------------


def _reject_constant(_: str) -> object:
    raise ReplayInputError("non-finite number is not allowed")


def parse_json(text: str) -> object:
    """JSON を読む。小数は Decimal・整数は int(どちらも明示)。NaN / Infinity は拒否。"""
    try:
        return json.loads(text, parse_float=Decimal, parse_int=int, parse_constant=_reject_constant)
    except ReplayInputError:
        raise
    except (ValueError, RecursionError):
        raise ReplayInputError("invalid JSON") from None


def _json_lines(text: str, *, max_lines: int) -> list[Mapping[str, object]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > max_lines:
        raise ReplayInputError("too many lines")
    rows: list[Mapping[str, object]] = []
    for line in lines:
        value = parse_json(line)
        if not isinstance(value, dict):
            raise ReplayInputError("each line must be a JSON object")
        rows.append(value)
    return rows


def _exact_keys(row: Mapping[str, object], expected: frozenset[str]) -> None:
    keys = set(row)
    if keys != expected:
        raise ReplayInputError("missing or unknown keys")


# --- 値の変換(1 か所ずつ)---------------------------------------------------------------------


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayInputError("expected a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _int(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayInputError("expected an integer")
    if value < minimum:
        raise ReplayInputError("integer out of range")
    return value


def _bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ReplayInputError("expected a boolean")
    return value


def to_decimal(value: object) -> Decimal:
    """Decimal へ。JSON の数値(Decimal / int)か、十進の文字列だけ。bool・非有限は拒否。"""
    if isinstance(value, bool):
        raise ReplayInputError("expected a number")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except InvalidOperation:
            raise ReplayInputError("invalid decimal") from None
    else:
        raise ReplayInputError("expected a number")
    if not result.is_finite():
        raise ReplayInputError("non-finite number is not allowed")
    return result


def to_fraction(value: object) -> Fraction:
    """Fraction へ。事前登録の本体の文字列(`"19/20"` や `"0.95"`)だけ。"""
    if not isinstance(value, str) or not value:
        raise ReplayInputError("expected a fraction string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ReplayInputError("invalid fraction") from None


def _enum[E](kind: Callable[[str], E], value: object) -> E:
    if not isinstance(value, str):
        raise ReplayInputError("expected a vocabulary word")
    try:
        return kind(value)
    except ValueError:
        raise ReplayInputError("unknown vocabulary word") from None


# --- 事前登録の本体 ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayParameters:
    """事前登録の本体から読んだ値(すべて登録された値。コードに既定を置かない)。"""

    max_gap: int
    min_cell: int
    outcome_spec: OutcomeSpec
    sale_cost_rate: Decimal
    required_agreement: Fraction
    reason_vocabulary: frozenset[str]
    held_out_refs: frozenset[str]
    max_input_lines: int
    max_input_bytes: int
    rate_digits: int


def registered_mapping(body: Mapping[str, Any]) -> dict[str, Any]:
    """本体の `parameters`(名前 → 値)。名前の集合が PARAMETER_NAMES と完全一致でなければ拒否。"""
    parameters = body.get("parameters")
    if not isinstance(parameters, list):
        raise ReplayInputError("parameters must be a list")
    values: dict[str, Any] = {}
    for entry in parameters:
        if not isinstance(entry, dict) or set(entry) != {"name", "value"}:
            raise ReplayInputError("each parameter needs exactly a name and a value")
        name = _text(entry["name"])
        if name in values:
            raise ReplayInputError("duplicate parameter name")
        values[name] = entry["value"]
    if set(values) != PARAMETER_NAMES:
        raise ReplayInputError("the registered parameter names are not the expected set")
    return values


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ReplayInputError("expected a list")
    return [_text(item) for item in value]


def load_prereg(text: str) -> tuple[dict[str, Any], ReplayParameters]:
    """事前登録の本体(JSON の text)→ (本体, 値)。"""
    body = parse_json(text)
    if not isinstance(body, dict):
        raise ReplayInputError("the registration body must be a JSON object")
    values = registered_mapping(body)
    windows = values["outcome_windows"]
    if not isinstance(windows, list):
        raise ReplayInputError("outcome_windows must be a list")
    if not isinstance(values["sale_cost_rate"], str):
        raise ReplayInputError("registered decimals are strings in the registration body")
    sale_cost_rate = to_decimal(values["sale_cost_rate"])
    if not _ZERO <= sale_cost_rate < _ONE:
        raise ReplayInputError("sale_cost_rate out of range")
    required = to_fraction(values["required_agreement"])
    if not 0 <= required <= 1:
        raise ReplayInputError("required_agreement out of range")
    spec = OutcomeSpec(
        windows=tuple(_int(w, minimum=1) for w in windows),
        origin=_enum(OriginRule, values["outcome_origin"]),
    )
    parameters = ReplayParameters(
        max_gap=_int(values["max_gap"], minimum=1),
        min_cell=_int(values["min_cell"], minimum=1),
        outcome_spec=spec,
        sale_cost_rate=sale_cost_rate,
        required_agreement=required,
        reason_vocabulary=frozenset(_string_list(values["reason_vocabulary"])),
        held_out_refs=frozenset(_string_list(values["held_out_refs"])),
        max_input_lines=_int(values["max_input_lines"], minimum=1),
        max_input_bytes=_int(values["max_input_bytes"], minimum=1),
        rate_digits=_int(values["rate_digits"], minimum=0),
    )
    if parameters.rate_digits > 12:
        raise ReplayInputError("rate_digits out of range")
    return body, parameters


@dataclass(frozen=True)
class Citation:
    """報告が引用する登録 ID・ハッシュ・使った値(照合の対象。報告書の自己申告)。"""

    registration_id: str
    content_hash: str
    used: dict[str, Any]


def load_citation(text: str) -> Citation:
    value = parse_json(text)
    if not isinstance(value, dict):
        raise ReplayInputError("the citation must be a JSON object")
    _exact_keys(value, _CITE_KEYS)
    used = value["used"]
    if not isinstance(used, dict):
        raise ReplayInputError("used must be an object")
    return Citation(
        registration_id=_text(value["registration_id"]),
        content_hash=_text(value["content_hash"]),
        used=used,
    )


# --- 観測 --------------------------------------------------------------------------------------


def is_held_out(holding_ref: str, registered_refs: frozenset[str]) -> bool:
    """症例か。事前登録の本体の指定(仮名化済みの保有の参照の一覧)に含まれるか。"""
    return holding_ref in registered_refs


def load_observations(
    text: str, *, held_out_refs: frozenset[str], max_lines: int
) -> tuple[Observation, ...]:
    rows = _json_lines(text, max_lines=max_lines)
    seen: set[tuple[str, int]] = set()
    observations: list[Observation] = []
    for row in rows:
        _exact_keys(row, _OBSERVATION_KEYS)
        holding_ref = _text(row["holding_ref"])
        day_index = _int(row["day_index"], minimum=0)
        if (holding_ref, day_index) in seen:
            raise ReplayInputError("duplicate observation")
        seen.add((holding_ref, day_index))
        raw_exclusion = row["exclusion"]
        observations.append(
            Observation(
                holding_ref=holding_ref,
                day_index=day_index,
                phase_kind=_text(row["phase_kind"]),
                phase_key=_optional_text(row["phase_key"]),
                exclusion=(
                    None if raw_exclusion is None else _enum(PopulationExclusion, raw_exclusion)
                ),
                is_first_evaluation=_bool(row["is_first_evaluation"]),
                held_out=is_held_out(holding_ref, held_out_refs),
            )
        )
    return tuple(observations)


# --- 対(OLD と NEW の判断)------------------------------------------------------------------------


@dataclass(frozen=True)
class PairRecord:
    """1 つの観測の OLD と NEW の判断(入力のファイルで受ける。ここでは作り直さない)。"""

    holding_ref: str
    day_index: int
    old: Judgement
    new: Judgement
    old_sold_ratio: Decimal
    new_sold_ratio: Decimal
    reason_code: str | None
    fires: tuple[Fire, Fire]
    recorded: str
    recomputed: str
    prior_high: Decimal


def _sold_ratio(value: object, action: ReplayAction) -> Decimal:
    ratio = to_decimal(value)
    if action is ReplayAction.HOLD and ratio != _ZERO:
        raise ReplayInputError("a HOLD judgement sells nothing")
    if action is ReplayAction.FULL and ratio != _ONE:
        raise ReplayInputError("a FULL judgement sells everything")
    if action is ReplayAction.PARTIAL and not _ZERO < ratio < _ONE:
        raise ReplayInputError("a PARTIAL judgement sells part")
    return ratio


def load_pairs(text: str, *, max_lines: int) -> tuple[PairRecord, ...]:
    rows = _json_lines(text, max_lines=max_lines)
    seen: set[tuple[str, int]] = set()
    pairs: list[PairRecord] = []
    for row in rows:
        _exact_keys(row, _PAIR_KEYS)
        holding_ref = _text(row["holding_ref"])
        day_index = _int(row["day_index"], minimum=0)
        if (holding_ref, day_index) in seen:
            raise ReplayInputError("duplicate pair")
        seen.add((holding_ref, day_index))
        old_row, new_row = row["old"], row["new"]
        if not isinstance(old_row, dict) or not isinstance(new_row, dict):
            raise ReplayInputError("old and new must be objects")
        _exact_keys(old_row, _OLD_KEYS)
        _exact_keys(new_row, _NEW_KEYS)
        old_action = _enum(ReplayAction, old_row["action"])
        new_action = _enum(ReplayAction, new_row["action"])
        prior_high = to_decimal(row["prior_high"])
        if prior_high <= _ZERO:
            raise ReplayInputError("prior_high must be positive")
        pairs.append(
            PairRecord(
                holding_ref=holding_ref,
                day_index=day_index,
                old=Judgement(old_action, _optional_text(old_row["exit_class"])),
                new=Judgement(
                    new_action,
                    _optional_text(new_row["exit_class"]),
                    _bool(new_row["undetermined"]),
                ),
                old_sold_ratio=_sold_ratio(old_row["sold_ratio"], old_action),
                new_sold_ratio=_sold_ratio(new_row["sold_ratio"], new_action),
                reason_code=_optional_text(new_row["reason_code"]),
                fires=(_enum(Fire, row["fire_first"]), _enum(Fire, row["fire_second"])),
                recorded=_text(row["recorded"]),
                recomputed=_text(row["recomputed"]),
                prior_high=prior_high,
            )
        )
    return tuple(pairs)


# --- 価格 --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceSeries:
    """1 つの保有の価格の列。bars[i] は営業日の索引 first_day_index + i の価格。"""

    first_day_index: int
    bars: tuple[Bar, ...]


def load_prices(text: str, *, max_lines: int) -> dict[str, PriceSeries]:
    rows = _json_lines(text, max_lines=max_lines)
    series: dict[str, PriceSeries] = {}
    for row in rows:
        _exact_keys(row, _SERIES_KEYS)
        holding_ref = _text(row["holding_ref"])
        if holding_ref in series:
            raise ReplayInputError("duplicate price series")
        raw_bars = row["bars"]
        if not isinstance(raw_bars, list) or not raw_bars:
            raise ReplayInputError("bars must be a non-empty list")
        bars: list[Bar] = []
        for raw in raw_bars:
            if not isinstance(raw, dict):
                raise ReplayInputError("each bar must be an object")
            _exact_keys(raw, _BAR_KEYS)
            bars.append(
                Bar(
                    open=to_decimal(raw["open"]),
                    high=to_decimal(raw["high"]),
                    low=to_decimal(raw["low"]),
                    close=to_decimal(raw["close"]),
                )
            )
        series[holding_ref] = PriceSeries(
            first_day_index=_int(row["first_day_index"], minimum=0), bars=tuple(bars)
        )
    return series


def check_references(observations: Iterable[Observation], pairs: Iterable[PairRecord]) -> None:
    """対は、観測に実在する (保有, 評価日) だけを指す(指さない対は入力の誤り)。"""
    known = {(o.holding_ref, o.day_index) for o in observations}
    for pair in pairs:
        if (pair.holding_ref, pair.day_index) not in known:
            raise ReplayInputError("a pair refers to an unknown observation")
