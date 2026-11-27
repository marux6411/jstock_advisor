"""Issue #281: decode 失敗の走査単位の集計行を、失敗 0 件の平常時も INFO で 1 行出す。

背景: `emit_failure_summary()` は failed = 0 のとき何も出さなかった。そのため『失敗 0 件だった』と
『走査していない』を区別できず、監視が『この行が無い』ことから健全性を読めなかった。

固定するもの(MANAGER 判断 = #281 issuecomment-5747861637):
  ・failed = 0 → INFO で `persistence decode summary collection=… scanned=N failed=0`
  ・failed > 0 → WARNING(文面・level は従来どおり)
  ・出すのは collection 名・走査件数・失敗件数だけ(item_id もレコードの中身も出ない)
  ・record_failure_policy は logger の level を宣言する(root が WARNING でも INFO が出る。#413)
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from jstock_advisor.infrastructure import record_failure_policy as policy_module
from jstock_advisor.infrastructure.record_failure_policy import (
    RecordFailurePolicy,
    decode_records,
    emit_failure_summary,
)

_SENTINEL = "sentinel-record-content-281"
_LOGGER = policy_module.logger.name
_HEAD = "persistence decode summary"


def _decode(raw: Any) -> str:
    if raw == "bad":
        raise ValueError(_SENTINEL)
    return str(raw)


def _items(*values: Any) -> list[tuple[str, Any]]:
    return [(f"id-{_SENTINEL}-{i}", v) for i, v in enumerate(values)]


def _summaries(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER and r.getMessage().startswith(_HEAD)]


def test_a_clean_scan_logs_one_info_summary_with_zero_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        decode_records(_items("a", "b", "c"), _decode, collection="notification_log")

    records = _summaries(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert records[0].getMessage() == (
        "persistence decode summary collection=notification_log scanned=3 failed=0"
    )


def test_an_empty_scan_is_also_recorded_as_scanned_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        decode_records([], _decode, collection="notification_log")

    assert [r.getMessage() for r in _summaries(caplog)] == [
        "persistence decode summary collection=notification_log scanned=0 failed=0"
    ]


@pytest.mark.parametrize("policy", [RecordFailurePolicy.LENIENT])
def test_a_scan_with_failures_still_warns_with_the_same_text(
    caplog: pytest.LogCaptureFixture, policy: RecordFailurePolicy
) -> None:
    with caplog.at_level(logging.INFO):
        decode_records(
            _items("a", "bad", "b"), _decode, collection="notification_log", policy=policy
        )

    records = _summaries(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == (
        "persistence decode summary collection=notification_log scanned=3 failed=1"
    )


def test_the_summary_is_not_visible_when_only_warning_and_above_is_collected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        decode_records(_items("a"), _decode, collection="notification_log")

    assert _summaries(caplog) == []


@pytest.mark.parametrize("scanned_values", [("a",), ("a", "bad")])
def test_neither_the_item_id_nor_the_record_content_reaches_the_summary(
    caplog: pytest.LogCaptureFixture, scanned_values: tuple[str, ...]
) -> None:
    with caplog.at_level(logging.DEBUG):
        decode_records(
            _items(*scanned_values),
            _decode,
            collection="notification_log",
            policy=RecordFailurePolicy.LENIENT,
        )

    for record in _summaries(caplog):
        text = record.getMessage() + str(record.args) + str(record.exc_info)
        assert _SENTINEL not in text
        assert "id-" not in text


def test_the_logger_level_is_declared_so_info_is_emitted_under_a_warning_root() -> None:
    """Lambda の root logger の既定は WARNING。module が level を宣言していれば INFO が出る。"""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.WARNING)
    try:
        logger = logging.getLogger(_LOGGER)
        assert logger.level == logging.INFO
        assert logger.isEnabledFor(logging.INFO)
    finally:
        root.setLevel(previous)


def test_emit_failure_summary_takes_any_iterable_of_failures_and_counts_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        emit_failure_summary("c", 5, iter(()))

    assert [r.getMessage() for r in _summaries(caplog)] == [
        "persistence decode summary collection=c scanned=5 failed=0"
    ]
