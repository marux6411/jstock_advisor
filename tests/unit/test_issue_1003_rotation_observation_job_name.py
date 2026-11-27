"""#1003: P の異常通知の対象名を専用の名称『資金入替の観測』にする(語彙の追加の census)。

USER 確定(#122 …6214116489): 『通知名称を「資金入替の観測」に変更。通知の発火条件、重要度、通知先は
変更しない』。この変更は **語彙を 1 つ足し、対応表の 1 行を向け直すだけ** であることを、変更前の
literal との差で固定する(意味論の変更ではなく『語彙の追加』の census):

  T1 IncidentJob の追加は ROTATION_OBSERVATION の 1 件だけ(既存 13 件の名前・値の書き換えは 0)
  T2 _INTERNAL_NAME_TO_JOB の変更は rotation-observation の 1 行だけ(他は変更前と完全一致)
  T3 本文(LINE・GitHub の異常 Issue)に『資金入替の観測』が出て、内部名は出ない
  T4 対応表に無い名前は従来どおり『その他の処理』(P は落ちない)
  + 名前の引き方: 接頭辞は完全一致で除く(最初の `-` までを落とす等の緩い除去で引かない)
"""

from __future__ import annotations

import datetime as dt

import pytest

from jstock_advisor.domain.notification import incident_message
from jstock_advisor.domain.notification.incident_github_issue_message import (
    IncidentFailureStage,
    IncidentIssueNotice,
    build_incident_issue_body,
    build_incident_issue_title,
)
from jstock_advisor.domain.notification.incident_message import (
    IncidentJob,
    IncidentNotice,
    build_incident_message,
    resolve_incident_job,
)

# 変更前(origin/main 5496975)の IncidentJob の全 13 件(名前 -> 値)。固定の literal
_BEFORE_JOBS = {
    "BUY_CANDIDATES": "買い候補チェック",
    "HOLDINGS_WATCHLIST": "保有株チェック",
    "DISCLOSURE_CHECK": "開示チェック",
    "EVALUATION": "過去の推奨の評価",
    "WATCHLIST_SCREENING": "ウォッチリスト自動追加",
    "WEEKLY_REVIEW": "週次レビュー",
    "MONTHLY_REVIEW": "月次レビュー",
    "QUARTERLY_REVIEW": "四半期レビュー",
    "LINE_WEBHOOK": "LINE の応答",
    "INCIDENT_NOTIFIER": "異常通知の中継処理",
    "ASYNC_INVOKE_FAILURE": "非同期実行の失敗",
    "SHAREHOLDER_BENEFIT_REGISTRY": "株主優待データの確認",
    "OTHER": "その他の処理",
}

# 変更前の _INTERNAL_NAME_TO_JOB(内部名 -> メンバー名)。rotation-observation だけが変わる
_BEFORE_NAME_TO_JOB = {
    "buy-candidates": "BUY_CANDIDATES",
    "buy-candidate-worker": "BUY_CANDIDATES",
    "holdings-watchlist": "HOLDINGS_WATCHLIST",
    "holdings-watchlist-worker": "HOLDINGS_WATCHLIST",
    "disclosure-check": "DISCLOSURE_CHECK",
    "evaluation": "EVALUATION",
    "watchlist-dispatcher": "WATCHLIST_SCREENING",
    "watchlist-worker": "WATCHLIST_SCREENING",
    "watchlist-terminal-failure-handler": "WATCHLIST_SCREENING",
    "watchlist-batch-reconciler": "WATCHLIST_SCREENING",
    "weekly-review": "WEEKLY_REVIEW",
    "monthly-review": "MONTHLY_REVIEW",
    "quarterly-review": "QUARTERLY_REVIEW",
    "line-webhook": "LINE_WEBHOOK",
    "incident-notifier": "INCIDENT_NOTIFIER",
    "rotation-observation": "OTHER",
    "shareholder-benefit-registry": "SHAREHOLDER_BENEFIT_REGISTRY",
}

_NEW_NAME = "資金入替の観測"
_AT = dt.datetime(2026, 10, 19, 0, 3, tzinfo=dt.UTC)  # 09:03 JST


# =================================================================================================
# T1 語彙の追加は 1 件だけ
# =================================================================================================


def test_t1_exactly_one_member_is_added_and_no_existing_one_is_rewritten() -> None:
    now = {job.name: job.value for job in IncidentJob}
    assert {k: v for k, v in now.items() if k in _BEFORE_JOBS} == _BEFORE_JOBS
    assert set(now) - set(_BEFORE_JOBS) == {"ROTATION_OBSERVATION"}
    assert now["ROTATION_OBSERVATION"] == _NEW_NAME
    assert set(_BEFORE_JOBS) <= set(now)  # 既存を消していない


def test_t1_the_new_label_does_not_collide_with_an_existing_label() -> None:
    others = [v for k, v in _BEFORE_JOBS.items()]
    assert all(_NEW_NAME not in other and other not in _NEW_NAME for other in others)
    assert len({job.value for job in IncidentJob}) == len(list(IncidentJob))  # 値の重複(別名)なし


# =================================================================================================
# T2 対応表の変更は 1 行だけ
# =================================================================================================


def test_t2_only_the_rotation_observation_row_changes() -> None:
    now = {k: v.name for k, v in incident_message._INTERNAL_NAME_TO_JOB.items()}
    assert set(now) == set(_BEFORE_NAME_TO_JOB)  # 行の増減なし
    changed = {k for k in now if now[k] != _BEFORE_NAME_TO_JOB[k]}
    assert changed == {"rotation-observation"}
    assert now["rotation-observation"] == "ROTATION_OBSERVATION"


def test_t2_the_queue_name_table_is_untouched() -> None:
    assert {k: v.name for k, v in incident_message._QUEUE_NAME_TO_JOB.items()} == {
        "watchlist-terminal-failure-dlq": "WATCHLIST_SCREENING",
        "buy-candidate-terminal-failure-dlq": "BUY_CANDIDATES",
        "holdings-watchlist-terminal-failure-dlq": "HOLDINGS_WATCHLIST",
        "async-invoke-failure-dlq": "ASYNC_INVOKE_FAILURE",
    }


# =================================================================================================
# T3 本文に出る(LINE・GitHub の異常 Issue)/ 内部名は出ない
# =================================================================================================


@pytest.mark.parametrize(
    "internal_name", ["rotation-observation", "jstock-advisor-rotation-observation"]
)
def test_t3_the_line_message_names_the_rotation_observation(internal_name: str) -> None:
    job = resolve_incident_job(internal_name)
    text = build_incident_message(IncidentNotice(job=job, occurred_at=_AT))
    assert f"対象: {_NEW_NAME}\n" in text
    assert "rotation-observation" not in text
    assert "その他の処理" not in text


def test_t3_the_github_issue_names_the_rotation_observation() -> None:
    notice = IncidentIssueNotice(
        job=resolve_incident_job("rotation-observation"),
        occurred_at=_AT,
        fingerprint="a" * 64,
        occurrence_count=1,
        failure_stage=IncidentFailureStage.CLOUDWATCH_ALARM,
    )
    title = build_incident_issue_title(notice)
    body = build_incident_issue_body(notice)
    assert f"{_NEW_NAME}で異常を検知しました" in title
    assert f"対象: {_NEW_NAME}" in body
    assert "rotation-observation" not in title + body


# =================================================================================================
# T4 / 名前の引き方
# =================================================================================================


@pytest.mark.parametrize(
    "unknown",
    [
        "some-new-internal-job",
        "jstock-rotation-observation",  # 接頭辞が違う(stack 接頭辞 jstock-advisor- と一致しない)
        "rotation-observation-extra",
        "x-rotation-observation",
        "rotation_observation",
        "Rotation-Observation",
        "",
    ],
)
def test_t4_a_name_that_is_not_exactly_the_known_one_falls_back_to_the_generic_job(
    unknown: str,
) -> None:
    assert resolve_incident_job(unknown) is IncidentJob.OTHER
    job = resolve_incident_job(unknown)
    text = build_incident_message(IncidentNotice(job=job, occurred_at=_AT))
    assert "対象: その他の処理\n" in text


def test_t4_non_string_names_fall_back_to_the_generic_job() -> None:
    for value in (None, 123, b"rotation-observation", ["rotation-observation"]):
        assert resolve_incident_job(value) is IncidentJob.OTHER
