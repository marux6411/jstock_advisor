"""`jstock allocation summary` の通しのリハーサル(Issue #934 BT-3。合成データのみ・AWS なし)。

日ごとに内容の違う営業日分の合成記録(欠落・日単位の skip・索引なし・記録 0 件・算出不能・
読取失敗・版の違い・解釈できない項目)を、CLI から JSON と表の両方で通しで流し、数え方
(どの日を数える日にするか・数えない理由・母数・終了コード・出力の秘密)が構成どおりかを固定する。

★ 期待値は『構成から手で導いた表』である ★(実装の `summarize` を再実行した値ではない)。ただし
設計の定義(`classify_day` の優先順)を別の場所に同じ順序で書き下した oracle であり、実装と
独立に導いたものではない(定義そのものの誤りは検出できない)。各行に『どの規則から導いたか』を
添える。値は件数の確認で、良し悪しの判定・閾値・評価の合否は付けない(#960 / USER の領域)。
★ このテストが通ることは『数え方が構成どおり』であることだけを意味し、結果の良し悪しの評価では
ない ★。

既存のテストとの違い: 読取元の休日を読まない確認・目標の境界・秘密の出力は既存の契約テストにも
ある。本テストはそれらを『日ごとに内容が変わる 10 営業日』で通しで重ね直す(休日が数えない日に
現れないことの確認と、記録に仕込んだ canary が出力に出ないことの強化を含む)。
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from typer.testing import CliRunner

from jstock_advisor.cli import allocation_summary as cli
from jstock_advisor.cli.main import app as main_app
from jstock_advisor.domain.business_calendar import BusinessCalendar
from jstock_advisor.domain.notification.allocation_shadow_cli_view import build_screen
from jstock_advisor.domain.notification.allocation_shadow_summary import (
    DayRecord,
    NotCounted,
    build_ten_day_report,
    summarize,
)
from jstock_advisor.infrastructure.aws.allocation_shadow_reader import (
    ReadFailure,
    ReadFailureKind,
    ReadStatus,
    RecordRead,
    read_day,
)
from tests.support import allocation_synthetic_days as syn

_runner = CliRunner()
_HOLIDAY = "2026-10-12"  # スポーツの日(営業日ではない)


class _Source:
    """audit_id → 記録の fake。読んだ audit_id を数える。fail_dates の日の読取は ACCESS_DENIED。"""

    def __init__(
        self, records: Mapping[str, dict[str, Any]], fail_dates: Sequence[str] = ()
    ) -> None:
        self.records = dict(records)
        self.fail_dates = tuple(fail_dates)
        self.requested: list[str] = []

    def get(self, audit_id: str) -> RecordRead:
        self.requested.append(audit_id)
        if any(day in audit_id for day in self.fail_dates):
            raise ReadFailure(ReadFailureKind.ACCESS_DENIED)
        record = self.records.get(audit_id)
        return RecordRead(ReadStatus.FOUND, record) if record else RecordRead(ReadStatus.NOT_FOUND)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(cli, "_today_jst", lambda: dt.date(2026, 10, 20))
    # 休日は jpholiday(国民の祝日)が決める。10/12 は祝日。固有の休業日は足さない
    monkeypatch.setattr(
        cli, "_default_calendar", lambda: BusinessCalendar(frozenset(), frozenset())
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    source: _Source,
    frm: str,
    to: str,
    *extra: str,
) -> Any:
    monkeypatch.setattr(cli, "_build_source", lambda selection: source)
    return _runner.invoke(
        main_app,
        ["allocation", "summary", "--backend", "local", "--from", frm, "--to", to, *extra],
    )


def _json(
    monkeypatch: pytest.MonkeyPatch,
    source: _Source,
    frm: str,
    to: str,
    *extra: str,
) -> tuple[int, dict[str, Any]]:
    result = _run(monkeypatch, source, frm, to, "--format", "json", *extra)
    return result.exit_code, json.loads(result.stdout)


# =================================================================================================
# 合成セット(構成)。期待値はテスト側の表に手で書く
# =================================================================================================


def _merge(*parts: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for part in parts:
        merged.update(part)
    return merged


def _full(day: str, *, model: str = syn.V1_MODEL, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """2 owner とも計算できた完全な日(候補は AVAILABLE で配分あり)。"""
    return syn.day_records(
        day,
        {
            "owner-a": syn.owner_record(day, "owner-a", model_version=model, **kwargs),
            "owner-b": syn.owner_record(day, "owner-b", model_version=model, **kwargs),
        },
    )


def _mixed_day(day: str) -> dict[str, dict[str, Any]]:
    """RAER の算出可・暫定が混ざり、配分と見送りがある日(canary を含む)。"""
    a_candidates = [
        syn.candidate(
            1, stock_code=syn.CANARY_STOCK_CODE, price=syn.CANARY_PRICE, raer="AVAILABLE"
        ),
        syn.candidate(
            2,
            raer="PROVISIONAL",
            decision="NOT_ALLOCATED",
            reason="INSUFFICIENT_CASH_REMAINING",
        ),
    ]
    b_candidates = [syn.candidate(1, raer="PROVISIONAL")]
    return syn.day_records(
        day,
        {
            "owner-a": syn.owner_record(
                day, "owner-a", candidates=a_candidates, available_cash=syn.CANARY_CASH
            ),
            "owner-b": syn.owner_record(day, "owner-b", candidates=b_candidates),
        },
    )


def _varied_reasons_day(day: str) -> dict[str, dict[str, Any]]:
    """見送り理由が多様な日(owner-a に 5 候補、owner-b に 1 候補)。"""
    a_candidates = [
        syn.candidate(1, raer="AVAILABLE"),
        syn.candidate(2, raer="PROVISIONAL", decision="NOT_ALLOCATED", reason="BELOW_HURDLE"),
        syn.candidate(3, raer="UNAVAILABLE", decision="NOT_ALLOCATED", reason="RAER_UNAVAILABLE"),
        syn.candidate(
            4, raer="AVAILABLE", decision="NOT_ALLOCATED", reason="BELOW_MINIMUM_TRADING_UNIT"
        ),
        syn.candidate(5, raer="PROVISIONAL", decision="NOT_ALLOCATED", reason="SECTOR_UNKNOWN"),
    ]
    return syn.day_records(
        day,
        {
            "owner-a": syn.owner_record(day, "owner-a", candidates=a_candidates),
            "owner-b": syn.owner_record(day, "owner-b"),
        },
    )


def _partial_day(day: str) -> dict[str, dict[str, Any]]:
    """索引: 試行 3・記録 2・未記録 1(WRITE_FAILED)。owner-a・owner-b の記録はある。"""
    owners = {
        "owner-a": syn.owner_record(day, "owner-a"),
        "owner-b": syn.owner_record(day, "owner-b"),
    }
    index = syn.index_record(
        day, ["owner-a", "owner-b"], attempted=3, unrecorded=1, reasons={"WRITE_FAILED": 1}
    )
    return syn.day_records(day, owners, index=index)


def _empty_day(day: str) -> dict[str, dict[str, Any]]:
    return syn.day_records(day, {}, index=syn.index_record(day, [], attempted=0))


def _no_computed_day(day: str) -> dict[str, dict[str, Any]]:
    """全 owner が計算していない(owner-a = SKIPPED・owner-b = FAILED)。"""
    return syn.day_records(
        day,
        {
            "owner-a": syn.owner_record(
                day, "owner-a", outcome="SKIPPED", skip_reason="TIME_BUDGET"
            ),
            "owner-b": syn.owner_record(
                day, "owner-b", outcome="FAILED", skip_reason="COMPUTATION_FAILED"
            ),
        },
    )


def set_a() -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """A 混在(10/05〜10/20 の 11 営業日。10/12 は祝日)。戻り値 = (記録, 読取失敗にする日)。"""
    records = _merge(
        _mixed_day("2026-10-05"),  # 1 完全
        _mixed_day("2026-10-06"),  # 2 完全
        _varied_reasons_day("2026-10-07"),  # 3 完全(見送り理由が多様)
        _partial_day("2026-10-08"),  # 4 PARTIAL
        syn.skip_only_records("2026-10-09"),  # 5 索引なし + 日単位の skip
        # 6 2026-10-13 は何も無い(索引なし)
        _empty_day("2026-10-14"),  # 7 索引あり・記録 0 件
        _no_computed_day("2026-10-15"),  # 8 全 owner が SKIPPED / FAILED
        # 9 2026-10-16 は読取失敗(fail_dates)
        _full("2026-10-19", model=syn.V2_MODEL, config_hash="cfg-synthetic-2"),  # 10 版 V2
        _full("2026-10-20", model=syn.V2_MODEL, config_hash="cfg-synthetic-2"),  # 11 版 V2
    )
    return records, ("2026-10-16",)


def set_b() -> dict[str, dict[str, Any]]:
    """B 全日が数える日(10/05〜10/19 の 10 営業日)。日ごとに算出不能の数・鮮度を変える。

    owner-a: 3 候補(算出不能が i % 3 件・残りは算出)・最終照合日は i % 4 日前 /
    owner-b: 1 候補(暫定)・当日照合。i = 0〜9(日付順)。
    """
    days = syn.business_days("2026-10-05", "2026-10-19", [_HOLIDAY])
    records: dict[str, dict[str, Any]] = {}
    for i, day in enumerate(days):
        unavailable = i % 3
        a_candidates = [
            syn.candidate(
                n + 1,
                raer="UNAVAILABLE" if n < unavailable else "AVAILABLE",
                decision="NOT_ALLOCATED" if n < unavailable else "ALLOCATED",
                reason="RAER_UNAVAILABLE" if n < unavailable else "ALLOCATED",
            )
            for n in range(3)
        ]
        reconciled = (dt.date.fromisoformat(day) - dt.timedelta(days=i % 4)).isoformat()
        records.update(
            syn.day_records(
                day,
                {
                    "owner-a": syn.owner_record(
                        day, "owner-a", candidates=a_candidates, reconciled_on=reconciled
                    ),
                    "owner-b": syn.owner_record(
                        day, "owner-b", candidates=[syn.candidate(1, raer="PROVISIONAL")]
                    ),
                },
            )
        )
    return records


def set_d1() -> dict[str, dict[str, Any]]:
    """D1 数えない理由 6 区分を 1 日ずつ(READ_FAILED を含まない。10/05〜10/12 の営業日 6 日)。"""
    unsupported = syn.index_record("2026-10-05", ["owner-a"], schema_version=99)
    unparsed_index = syn.index_record(
        "2026-10-09", ["owner-a"], attempted=2, unrecorded=1, reasons={"NOT_A_KNOWN_REASON": 1}
    )
    return _merge(
        syn.day_records("2026-10-05", {}, index=unsupported),  # UNSUPPORTED(対応しない版)
        syn.skip_only_records("2026-10-06"),  # NO_INDEX_DAY_SKIP
        # 2026-10-07 は何も無い = NO_INDEX
        _empty_day("2026-10-08"),  # EMPTY
        syn.day_records(  # UNPARSED(解釈できない理由コード + 計算できた owner あり)
            "2026-10-09",
            {"owner-a": syn.owner_record("2026-10-09", "owner-a")},
            index=unparsed_index,
        ),
        _no_computed_day("2026-10-13"),  # NO_COMPUTED_OWNER
    )


# =================================================================================================
# A 混在: 分類・母数・版・終了コード(手書きの oracle)
# =================================================================================================

# (日付, 数える, 理由, 状態, 計算/見送り/失敗/その他, 試行, 記録, 欠落, 読取失敗, 規則)
_A_ROWS: list[tuple[Any, ...]] = [
    # 規則: 計算できた owner がいれば数える日
    ("2026-10-05", True, None, "COMPLETE", (2, 0, 0, 0), 2, 2, 0, None),
    ("2026-10-06", True, None, "COMPLETE", (2, 0, 0, 0), 2, 2, 0, None),
    ("2026-10-07", True, None, "COMPLETE", (2, 0, 0, 0), 2, 2, 0, None),
    # 規則: PARTIAL でも計算できた owner がいれば数える。欠落 = 試行 3 − 記録 2 + 記録の無い符号 0
    ("2026-10-08", True, None, "PARTIAL", (2, 0, 0, 0), 3, 2, 1, None),
    (
        "2026-10-09",
        False,
        "NO_INDEX_DAY_SKIP",
        "NO_INDEX_DAY_SKIP",
        (0,) * 4,
        None,
        None,
        None,
        None,
    ),
    ("2026-10-13", False, "NO_INDEX", "NO_INDEX", (0, 0, 0, 0), None, None, None, None),
    ("2026-10-14", False, "EMPTY", "EMPTY", (0, 0, 0, 0), 0, 0, 0, None),
    # 規則: 計算できた owner が 0(SKIPPED 1・FAILED 1)なら数えない
    ("2026-10-15", False, "NO_COMPUTED_OWNER", "COMPLETE", (0, 1, 1, 0), 2, 2, 0, None),
    # 規則: 読取に失敗した日は数えない(0 件の日と区別する = 理由 READ_FAILED と種別を持つ)
    ("2026-10-16", False, "READ_FAILED", None, (0, 0, 0, 0), None, None, None, "ACCESS_DENIED"),
    ("2026-10-19", True, None, "COMPLETE", (2, 0, 0, 0), 2, 2, 0, None),
    ("2026-10-20", True, None, "COMPLETE", (2, 0, 0, 0), 2, 2, 0, None),
]


def _day_tuple(day: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        day["business_date"],
        day["counted"],
        day["not_counted"],
        day["state"],
        (
            day["computed_owners"],
            day["skipped_owners"],
            day["failed_owners"],
            day["other_owners"],
        ),
        day["attempted"],
        day["recorded"],
        day["missing_count"],
        day["read_failure"],
    )


def _run_a(monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[Any, _Source]:
    records, fail = set_a()
    source = _Source(records, fail)
    return _run(monkeypatch, source, "2026-10-05", "2026-10-20", *extra), source


def test_set_a_days_are_classified_by_the_hand_written_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, fail = set_a()
    code, payload = _json(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    assert [_day_tuple(d) for d in payload["days"]] == [row[:9] for row in _A_ROWS]
    assert code == 1  # 読取に失敗した日が 1 日ある(数えない日として出力に含めた上で終了コード 1)


def test_set_a_top_level_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    records, fail = set_a()
    code, payload = _json(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    assert code == 1
    assert payload["expected_days_total"] == 11  # 祝日(10/12)を含まない
    assert payload["counted_days"] == 6  # 10/05・06・07・08・19・20
    assert payload["reached"] is False
    assert payload["first_counted_date"] == "2026-10-05"
    assert payload["tenth_counted_date"] is None
    # 以後の期待される営業日が無く(最後に読んだ日 = 10/20)、残り 4 日を数える日にはできない
    assert payload["projected_tenth_date_if_all_remaining_days_count"] is None
    assert payload["not_counted"] == {
        "EMPTY": 1,
        "NO_COMPUTED_OWNER": 1,
        "NO_INDEX": 1,
        "NO_INDEX_DAY_SKIP": 1,
        "READ_FAILED": 1,
    }
    # 数える日 6 + 数えない日 5 = 全 11 日(母数は数える日だけでも全日でもなく両方を合わせる)
    assert payload["counted_days"] + sum(payload["not_counted"].values()) == 11
    assert payload["target_days"] == 10


def test_set_a_holiday_is_neither_read_nor_a_not_counted_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, fail = set_a()
    source = _Source(records, fail)
    _, payload = _json(monkeypatch, source, "2026-10-05", "2026-10-20")
    assert all(_HOLIDAY not in audit_id for audit_id in source.requested)
    assert _HOLIDAY not in [d["business_date"] for d in payload["days"]]
    # 10/10・10/11・10/17・10/18(週末)も同様に現れない
    assert len(payload["days"]) == 11


def test_set_a_versions_are_not_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    records, fail = set_a()
    _, payload = _json(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    assert payload["versions_periods"] == 2
    first, second = payload["periods"]
    # 規則: 版は記録が読めた owner のいる日で決まる(SKIPPED / FAILED の日 10/15 も V1 の記録を持つ)
    got_first = (first["model_version"], first["first_date"], first["last_date"])
    assert (*got_first, first["day_count"]) == (
        syn.V1_MODEL,
        "2026-10-05",
        "2026-10-15",
        5,
    )
    assert (
        second["model_version"],
        second["first_date"],
        second["last_date"],
        second["day_count"],
    ) == (syn.V2_MODEL, "2026-10-19", "2026-10-20", 2)


def _measure(period: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return next(m for m in period["measures"] if m["key"] == key)


def test_set_a_candidate_counts_per_version(monkeypatch: pytest.MonkeyPatch) -> None:
    records, fail = set_a()
    _, payload = _json(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    v1, v2 = payload["periods"]
    # V1(10/05・06・07・08)の候補: 3 + 3 + 6 + 2 = 14。10/15 の owner は候補を持たない
    raer = _measure(v1, "raer_states")
    assert raer["denominator"] == 14
    assert raer["buckets"] == {"AVAILABLE": 7, "PROVISIONAL": 6, "UNAVAILABLE": 1}
    reasons = _measure(v1, "allocation_reasons")
    assert reasons["buckets"] == {
        "ALLOCATED": 8,
        "BELOW_HURDLE": 1,
        "BELOW_MINIMUM_TRADING_UNIT": 1,
        "INSUFFICIENT_CASH_REMAINING": 2,
        "RAER_UNAVAILABLE": 1,
        "SECTOR_UNKNOWN": 1,
    }
    # V2(10/19・20)の候補: 2 owner × 1 候補 × 2 日 = 4(全て算出・配分)
    assert _measure(v2, "raer_states")["denominator"] == 4
    assert _measure(v2, "raer_states")["buckets"] == {"AVAILABLE": 4}
    assert _measure(v2, "allocation_reasons")["buckets"] == {"ALLOCATED": 4}


def test_set_a_table_lines_match_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _run_a(monkeypatch)
    assert result.exit_code == 1
    lines = result.stdout.splitlines()
    assert "数える日: 6 / 10(目標に未到達)" in lines
    assert (
        "数えない日(理由別): EMPTY=1, NO_COMPUTED_OWNER=1, NO_INDEX=1, NO_INDEX_DAY_SKIP=1, "
        "READ_FAILED=1 / 全 11 日" in lines
    )
    expected = [
        "  2026-10-05 COMPLETE ○ - | 2/0/0/0 | 0/2",
        "  2026-10-08 PARTIAL ○ - | 2/0/0/0 | 1/3",
        "  2026-10-09 NO_INDEX_DAY_SKIP × NO_INDEX_DAY_SKIP | 0/0/0/0 | -",
        "  2026-10-13 NO_INDEX × NO_INDEX | 0/0/0/0 | -",
        "  2026-10-14 EMPTY × EMPTY | 0/0/0/0 | 0/0",
        "  2026-10-15 COMPLETE × NO_COMPUTED_OWNER | 0/1/1/0 | 0/2",
        "  2026-10-16 - × READ_FAILED | 0/0/0/0 | - 読取失敗=ACCESS_DENIED",
    ]
    for line in expected:
        assert line in lines, line
    assert "PARTIAL の日(欠落件数 / 母数 = 試みた owner 数):" in lines
    assert "  2026-10-08: 欠落 1 / 3" in lines


# =================================================================================================
# B 全日が数える日: 目標に達する・指標が構成どおり
# =================================================================================================


def test_set_b_reaches_the_target_with_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    code, payload = _json(monkeypatch, _Source(set_b()), "2026-10-05", "2026-10-19")
    assert code == 0
    assert payload["counted_days"] == 10
    assert payload["expected_days_total"] == 10
    assert payload["reached"] is True
    assert payload["first_counted_date"] == "2026-10-05"
    assert payload["tenth_counted_date"] == "2026-10-19"
    assert payload["not_counted"] == {}
    assert payload["versions_periods"] == 1
    period = payload["periods"][0]
    assert (period["first_date"], period["last_date"], period["day_count"]) == (
        "2026-10-05",
        "2026-10-19",
        10,
    )


def test_set_b_measures_follow_the_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    _, payload = _json(monkeypatch, _Source(set_b()), "2026-10-05", "2026-10-19")
    period = payload["periods"][0]
    # 候補: owner-a 3 × 10 日 + owner-b 1 × 10 日 = 40。算出不能 = i % 3 の合計 (0+1+2) × 3 + 0 = 9
    raer = _measure(period, "raer_states")
    assert raer["denominator"] == 40
    assert raer["buckets"] == {"AVAILABLE": 21, "PROVISIONAL": 10, "UNAVAILABLE": 9}
    unavailable = _measure(period, "raer_unavailable_rate")
    assert (unavailable["count"], unavailable["denominator"]) == (9, 40)
    assert unavailable["rate"] == "0.2250"
    # 鮮度: owner-a の経過日数は i % 4(0・1・2・3 が 3・3・2・2 日)、owner-b は常に 0(10 日)
    staleness = _measure(period, "cash_staleness_days")
    assert staleness["denominator"] == 20
    assert sorted(staleness["buckets"].values(), reverse=True) == [13, 3, 2, 2]


# =================================================================================================
# C 目標の境界: 9 日 → 未達(3) / 10 日 → 達成(0) / 11 日 → 達成(0)
# =================================================================================================


def test_boundary_nine_days_is_not_reached_exit_three(monkeypatch: pytest.MonkeyPatch) -> None:
    records = set_b()
    for key in [k for k in records if "2026-10-19" in k]:
        del records[key]  # 最後の 1 日を『索引なし』にする
    code, payload = _json(monkeypatch, _Source(records), "2026-10-05", "2026-10-19")
    assert code == 3
    assert payload["counted_days"] == 9
    assert payload["reached"] is False
    assert payload["tenth_counted_date"] is None
    assert payload["not_counted"] == {"NO_INDEX": 1}


def test_boundary_ten_days_is_reached_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    code, payload = _json(monkeypatch, _Source(set_b()), "2026-10-05", "2026-10-19")
    assert (code, payload["counted_days"], payload["reached"]) == (0, 10, True)


def test_boundary_eleven_days_is_reached_and_tenth_date_is_the_tenth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _merge(set_b(), _full("2026-10-20"))
    code, payload = _json(monkeypatch, _Source(records), "2026-10-05", "2026-10-20")
    assert code == 0
    assert payload["counted_days"] == 11
    assert payload["reached"] is True
    assert payload["first_counted_date"] == "2026-10-05"
    assert payload["tenth_counted_date"] == "2026-10-19"  # 11 日目(10/20)ではなく 10 日目


def test_verdicts_are_ordered_by_date_even_if_records_were_inserted_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordered = set_b()
    reversed_records = dict(reversed(list(ordered.items())))
    _, forward = _json(monkeypatch, _Source(ordered), "2026-10-05", "2026-10-19")
    _, backward = _json(monkeypatch, _Source(reversed_records), "2026-10-05", "2026-10-19")
    assert forward == backward
    dates = [d["business_date"] for d in forward["days"]]
    assert dates == sorted(dates)
    assert forward["first_counted_date"] == dates[0]
    assert forward["tenth_counted_date"] == dates[9]


def test_the_same_business_date_is_counted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """日の数え方は『1 営業日 = 1 回の読取』に依存する。重複した日が二重に数えられない。"""
    source = _Source(set_b())
    _, payload = _json(monkeypatch, source, "2026-10-05", "2026-10-19")
    dates = [d["business_date"] for d in payload["days"]]
    assert len(dates) == len(set(dates)) == 10
    index_reads = [a for a in source.requested if a.startswith("allocation_shadow_index:")]
    assert len(index_reads) == len(set(index_reads)) == 10  # 索引は 1 日 1 回だけ読む


# =================================================================================================
# D 数えない理由の網羅
# =================================================================================================

_D1_EXPECTED = {
    "2026-10-05": "UNSUPPORTED",
    "2026-10-06": "NO_INDEX_DAY_SKIP",
    "2026-10-07": "NO_INDEX",
    "2026-10-08": "EMPTY",
    "2026-10-09": "UNPARSED",
    "2026-10-13": "NO_COMPUTED_OWNER",
}


def test_d1_each_not_counted_reason_appears_once_exit_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, payload = _json(monkeypatch, _Source(set_d1()), "2026-10-05", "2026-10-13")
    assert {d["business_date"]: d["not_counted"] for d in payload["days"]} == _D1_EXPECTED
    assert payload["not_counted"] == {reason: 1 for reason in _D1_EXPECTED.values()}
    assert payload["counted_days"] == 0
    assert code == 3  # 読取の失敗は無く、目標に未達


def test_d2_adding_a_read_failure_day_turns_the_exit_code_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _merge(set_d1(), _full("2026-10-14"))
    code, payload = _json(monkeypatch, _Source(records, ["2026-10-14"]), "2026-10-05", "2026-10-14")
    reasons = {d["business_date"]: d["not_counted"] for d in payload["days"]}
    assert reasons["2026-10-14"] == "READ_FAILED"
    assert {k: v for k, v in reasons.items() if k != "2026-10-14"} == _D1_EXPECTED
    assert payload["not_counted"]["READ_FAILED"] == 1
    assert code == 1  # 読取の失敗は『未達 (3)』より優先


def test_precedence_empty_day_with_an_unknown_reason_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """規則: 索引あり・記録 0 件(EMPTY)は UNPARSED より先に当てはまる。"""
    day = "2026-10-05"
    index = syn.index_record(day, [], attempted=1, unrecorded=1, reasons={"NOT_A_KNOWN_REASON": 1})
    records = syn.day_records(day, {}, index=index)
    _, payload = _json(monkeypatch, _Source(records), day, day)
    assert payload["days"][0]["not_counted"] == "EMPTY"


def test_precedence_unparsed_wins_over_no_computed_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """規則: 解釈できない項目が 1 件以上なら、計算できた owner が 0 でも UNPARSED を採る。"""
    day = "2026-10-05"
    skipped = syn.owner_record(day, "owner-a", outcome="SKIPPED", skip_reason="TIME_BUDGET")
    owners = {"owner-a": skipped}
    index = syn.index_record(
        day, ["owner-a"], attempted=2, unrecorded=1, reasons={"NOT_A_KNOWN_REASON": 1}
    )
    records = syn.day_records(day, owners, index=index)
    _, payload = _json(monkeypatch, _Source(records), day, day)
    assert payload["days"][0]["not_counted"] == "UNPARSED"


def test_not_read_is_reachable_only_by_direct_call_and_all_eight_reasons_are_covered() -> None:
    """NOT_READ は CLI の経路では出ない(読めなかった日は READ_FAILED)。直接呼出しで 1 日。"""
    report = build_ten_day_report(summarize([]), ["2026-10-05", "2026-10-06"])
    assert [v.not_counted for v in report.verdicts] == [NotCounted.NOT_READ, NotCounted.NOT_READ]
    assert dict(report.not_counted_counts) == {"NOT_READ": 2}
    covered = set(_D1_EXPECTED.values()) | {"READ_FAILED", "NOT_READ"}
    assert covered == {reason.value for reason in NotCounted}  # 8 区分を網羅している


# =================================================================================================
# 秘密・決定性・再実行
# =================================================================================================

_MUST_NOT_APPEAR = (
    syn.CANARY_STOCK_CODE,
    syn.CANARY_CASH,
    syn.CANARY_PRICE,
    syn.ref("owner-a"),
    syn.digest("owner-a"),
    syn.digest("owner-b"),
    "owner-a",
    "owner-b",
)


@pytest.mark.parametrize("fmt", ["json", "table"])
def test_no_canary_or_ref_reaches_output(monkeypatch: pytest.MonkeyPatch, fmt: str) -> None:
    records, fail = set_a()
    source = _Source(records, fail)
    assert any(syn.CANARY_STOCK_CODE in json.dumps(r) for r in records.values())  # 仕込んである
    result = _run(monkeypatch, source, "2026-10-05", "2026-10-20", "--format", fmt)
    out = result.stdout + result.stderr
    for leaked in _MUST_NOT_APPEAR:
        assert leaked not in out, leaked


def test_owner_table_appears_only_with_by_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    records, fail = set_a()
    _, plain = _json(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    assert "by_owner" not in plain
    _, with_owner = _json(
        monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20", "--by-owner"
    )
    assert "by_owner" in with_owner
    table = _run(
        monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20", "--by-owner"
    ).stdout
    assert "owner 別" in table


def test_two_runs_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    records, fail = set_a()
    first = _run(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    second = _run(monkeypatch, _Source(records, fail), "2026-10-05", "2026-10-20")
    assert (first.exit_code, first.stdout) == (second.exit_code, second.stdout)


def test_the_run_only_reads_and_never_writes_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    records, fail = set_a()
    before = json.dumps(records, sort_keys=True)
    source = _Source(records, fail)
    _run(monkeypatch, source, "2026-10-05", "2026-10-20")
    assert json.dumps(source.records, sort_keys=True) == before
    assert not hasattr(source, "put") and not hasattr(source, "write")


# =================================================================================================
# 審査(サブちゃん …6266674835)の SHOULD-A / SHOULD-B と観察への対応
# =================================================================================================


def _records_from(source: _Source, days: Sequence[str]) -> list[DayRecord]:
    """CLI と同じ読み方(read_day → build_screen)で、読めた日の入力を作る。"""
    return [
        DayRecord(day, build_screen(read_day(source, dt.date.fromisoformat(day)), {}))
        for day in days
    ]


def test_a_duplicated_expected_day_is_not_counted_twice() -> None:
    """SHOULD-A: CLI は暦から日付を作るので重複しない。実データ側の組み立てで 1 日が 2 回入っても、
    1 日を 2 日と数えない(『10 営業日に達した』が偽になる型。例外は出ない)。直接呼出し。"""
    report = build_ten_day_report(summarize([]), ["2026-10-05", "2026-10-05"])
    assert [v.business_date for v in report.verdicts] == ["2026-10-05"]
    assert report.expected_days_total == 1
    assert dict(report.not_counted_counts) == {"NOT_READ": 1}


def test_the_projected_tenth_date_is_the_nth_remaining_expected_day() -> None:
    """SHOULD-B: 10/29 の判断で見る『あと何日で 10 営業日に達するか』。終了コードも件数も変わらない
    field なので、非 None になる構成で固定する。期待される営業日 10 日のうち 6 日を読み(全て数える
    日)、残り 4 日は未読(NOT_READ。CLI の経路では出ないので直接呼出し)。"""
    days = syn.business_days("2026-10-05", "2026-10-19", [_HOLIDAY])  # 10 営業日
    source = _Source(set_b())
    read = _records_from(source, days[:6])  # 10/05・06・07・08・09・13
    report = build_ten_day_report(summarize(read), days)
    assert report.counted_days == 6
    assert report.reached is False
    # 残りの営業日 = 10/14・15・16・19。10 日目に達するには 4 日が要る = 残りの 4 日目 = 10/19
    assert report.projected_tenth_date == "2026-10-19"
    # 5 日しか数えていなければ残り 5 日が要るが残りは 5 日(10/13・14・15・16・19)= 5 日目 = 10/19
    five = build_ten_day_report(summarize(_records_from(source, days[:5])), days)
    assert five.counted_days == 5
    assert five.projected_tenth_date == "2026-10-19"
    # 4 日しか数えていなければ 6 日が要るが残りは 6 日(10/09〜)= 6 日目 = 10/19(境界)
    four = build_ten_day_report(summarize(_records_from(source, days[:4])), days)
    assert four.projected_tenth_date == "2026-10-19"
    # 残りの日数が足りなければ見込みは出さない(数える日 6 のまま、期待される営業日が 9 日しかない)
    short = build_ten_day_report(summarize(read), days[:9])
    assert short.projected_tenth_date is None


def test_a_ref_in_the_index_without_its_owner_record_counts_as_missing_through_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """観察: 欠落件数の『索引にあるが記録の無い符号』の項を通しでも固定する。
    索引は owner-a・owner-b の 2 件(試行 2・記録 2)を載せるが、owner-b の記録が無い日。
    欠落 = 試行 2 − 記録 2 + 記録の無い符号 1 = 1。owner-a が計算できているので数える日。"""
    day = "2026-10-05"
    records = syn.day_records(
        day,
        {"owner-a": syn.owner_record(day, "owner-a")},
        index=syn.index_record(day, ["owner-a", "owner-b"]),
    )
    code, payload = _json(monkeypatch, _Source(records), day, day)
    row = payload["days"][0]
    assert (row["state"], row["counted"], row["computed_owners"], row["other_owners"]) == (
        "PARTIAL",
        True,
        1,
        1,
    )
    assert (row["attempted"], row["recorded"], row["missing_count"]) == (2, 2, 1)
    assert code == 3  # 数える日 1 / 10 = 未達(読取の失敗なし)


def test_no_projection_when_the_trailing_days_were_read_but_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHOULD-B の残り(サブちゃん …6267098254): 『最後に読んだ日』と『最後の数える日』は別。
    末尾の営業日が『読めたが数えない日』(EMPTY)なら、その日は既に読み終えていて、これから数える
    日になりうる日ではない。10 営業日のうち先頭 8 日 = 数える日・末尾 2 日 = 記録 0 件・目標 10:
    数える日 8 / 未達だが、見込みの日は立たない(None)。見込みを捏造すると『まだ見込みがある』と
    楽観に振れ、評価の打ち切りの判断を誤らせる(終了コードも件数も内訳も変わらない)。"""
    days = syn.business_days("2026-10-05", "2026-10-19", [_HOLIDAY])  # 10 営業日
    records = {
        key: record
        for key, record in set_b().items()
        if not any(day in key for day in days[8:])  # 末尾 2 日の記録を外し、空の索引に差し替える
    }
    for day in days[8:]:
        records.update(_empty_day(day))
    code, payload = _json(monkeypatch, _Source(records), days[0], days[-1])
    assert code == 3
    assert payload["counted_days"] == 8
    assert payload["reached"] is False
    assert payload["not_counted"] == {"EMPTY": 2}
    assert payload["projected_tenth_date_if_all_remaining_days_count"] is None
