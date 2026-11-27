"""`jstock replay`(Issue #880 R1b)の契約テストが使う、合成の入力(実在の銘柄・保有・価格を使わない)。

全部の値は架空: 保有の参照は `h1` などの短い仮名、価格は 100 前後の等差、
判断は HOLD / PARTIAL / FULL の組。各テストが期待する件数は、下の表の読み方と対応する。

観測(8 行): 分析対象 5 / 除外 2(LEGACY 1・DATA_INSUFFICIENT 1)/
症例 1(`h-case`。事前登録の指定)
対(OLD と NEW。8 行): 分析対象の 5 対の分類 = MATCH 1 / NEW_ONLY 1 / STRENGTH_DIFF 1 /
CLASS_DIFF 1 / UNDETERMINED 1。再現性 = 5 対中 4 一致
"""

from __future__ import annotations

import json
from typing import Any

from jstock_advisor.domain.backtest.preregistration_check import content_hash, registered_values

PARAMETERS: dict[str, Any] = {
    "max_gap": 2,
    "min_cell": 2,
    "outcome_windows": [1, 2],
    "outcome_origin": "DECISION_DAY_CLOSE",
    "sale_cost_rate": "0.2",
    "required_agreement": "1",
    "reason_vocabulary": ["R_A", "R_B"],
    "held_out_refs": ["h-case"],
    "max_input_lines": 100,
    "max_input_bytes": 100000,
    "rate_digits": 2,
}


def prereg_body(**overrides: Any) -> dict[str, Any]:
    values = {**PARAMETERS, **overrides}
    return {
        "registration_id": "REG-TEST-1",
        "parameters": [{"name": name, "value": value} for name, value in values.items()],
    }


def citation_for(body: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    cite: dict[str, Any] = {
        "registration_id": body["registration_id"],
        "content_hash": content_hash(body),
        "used": registered_values(body),
    }
    cite.update(overrides)
    return cite


def to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def to_lines(rows: list[dict[str, Any]]) -> str:
    return "\n".join(to_json(row) for row in rows) + "\n"


def _observation(
    holding: str,
    day: int,
    key: str | None,
    *,
    exclusion: str | None = None,
    first: bool = False,
) -> dict[str, Any]:
    return {
        "holding_ref": holding,
        "day_index": day,
        "phase_kind": "L2",
        "phase_key": key,
        "exclusion": exclusion,
        "is_first_evaluation": first,
    }


def observation_rows() -> list[dict[str, Any]]:
    return [
        _observation("h1", 10, "A", first=True),
        _observation("h1", 11, "A"),
        _observation("h1", 20, "B"),
        _observation("h2", 10, "A"),
        _observation("h2", 11, None),
        _observation("h3", 10, "A", exclusion="LEGACY"),
        _observation("h-case", 10, "A"),
        _observation("h4", 10, "C", exclusion="DATA_INSUFFICIENT"),
    ]


def _pair(
    holding: str,
    day: int,
    old: tuple[str, str | None, str],
    new: tuple[str, str | None, bool, str],
    *,
    reason: str | None,
    fires: tuple[str, str],
    recorded: str,
    recomputed: str,
) -> dict[str, Any]:
    return {
        "holding_ref": holding,
        "day_index": day,
        "old": {"action": old[0], "exit_class": old[1], "sold_ratio": old[2]},
        "new": {
            "action": new[0],
            "exit_class": new[1],
            "undetermined": new[2],
            "sold_ratio": new[3],
            "reason_code": reason,
        },
        "fire_first": fires[0],
        "fire_second": fires[1],
        "recorded": recorded,
        "recomputed": recomputed,
        "prior_high": "100.5",
    }


def pair_rows() -> list[dict[str, Any]]:
    return [
        _pair(
            "h1",
            10,
            ("HOLD", None, "0"),
            ("HOLD", None, False, "0"),
            reason=None,
            fires=("FIRED", "FIRED"),
            recorded="HOLD",
            recomputed="HOLD",
        ),
        _pair(
            "h1",
            11,
            ("FULL", "C1", "1"),
            ("PARTIAL", "C1", False, "0.5"),
            reason="R_A",
            fires=("FIRED", "NOT_FIRED"),
            recorded="FULL",
            recomputed="FULL",
        ),
        _pair(
            "h1",
            20,
            ("HOLD", None, "0"),
            ("FULL", "C2", False, "1"),
            reason="R_B",
            fires=("NOT_FIRED", "NOT_FIRED"),
            recorded="HOLD",
            recomputed="HOLD",
        ),
        _pair(
            "h2",
            10,
            ("PARTIAL", "C1", "0.5"),
            ("HOLD", None, True, "0"),
            reason=None,
            fires=("UNDETERMINED", "FIRED"),
            recorded="PARTIAL",
            recomputed="HOLD",
        ),
        _pair(
            "h2",
            11,
            ("FULL", "C1", "1"),
            ("FULL", "C2", False, "1"),
            reason="R_UNKNOWN",
            fires=("NOT_FIRED", "FIRED"),
            recorded="FULL",
            recomputed="FULL",
        ),
        _pair(
            "h3",
            10,
            ("HOLD", None, "0"),
            ("HOLD", None, False, "0"),
            reason=None,
            fires=("NOT_FIRED", "NOT_FIRED"),
            recorded="HOLD",
            recomputed="HOLD",
        ),
        _pair(
            "h-case",
            10,
            ("FULL", "C1", "1"),
            ("HOLD", None, False, "0"),
            reason="R_A",
            fires=("FIRED", "FIRED"),
            recorded="FULL",
            recomputed="HOLD",
        ),
        _pair(
            "h4",
            10,
            ("HOLD", None, "0"),
            ("HOLD", None, False, "0"),
            reason=None,
            fires=("NOT_FIRED", "NOT_FIRED"),
            recorded="HOLD",
            recomputed="HOLD",
        ),
    ]


def _bars(count: int) -> list[dict[str, Any]]:
    return [
        {"open": str(100 + i), "high": str(101 + i), "low": str(99 + i), "close": str(100 + i)}
        for i in range(count)
    ]


def price_rows() -> list[dict[str, Any]]:
    return [
        {"holding_ref": "h1", "first_day_index": 10, "bars": _bars(14)},
        {"holding_ref": "h2", "first_day_index": 10, "bars": _bars(4)},
        {"holding_ref": "h3", "first_day_index": 10, "bars": _bars(4)},
        {"holding_ref": "h-case", "first_day_index": 10, "bars": _bars(4)},
        {"holding_ref": "h4", "first_day_index": 10, "bars": _bars(4)},
    ]
