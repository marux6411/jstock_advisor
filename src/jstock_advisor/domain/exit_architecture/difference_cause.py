"""『なぜ Arbiter < E2 か』の分類(Issue #878 PR-3c-i。dormant)。

現行の E2(profit_taking)の最終の強さと、Arbiter の `Decision`・adapter の結果から、差の原因を
**構造化された語彙**で返す純粋関数。文字列の解析はしない
(trace・proposals・Decision の型だけを読む)。
差を『説明できる差』と『説明できない差』に分け、後者を黙って許さない。

分類(rev2 の区分 E / C / U / X に対応)
  SAME                          差なし(区分 E)
  NO_FULL_EVIDENCE              E2 は FULL・Arbiter は FULL の独立根拠が無く PARTIAL(区分 C)。
                                どの経路かを `trigger_kind` で添える(FULL を主張した提案のうち、
                                origin が最大の TriggerKind。同点は固定の順序)
  USER_TARGET_ONLY              E2 の強さが、ユーザー設定の目標だけに由来する(区分 U。通知のみで
                                Arbiter の入力にしない = pass_through がある)
  ARBITER_STRONGER              Arbiter が E2 より強い(区分 X。失敗。FULL / PARTIAL が増える経路が
                                無いという不変条件に反する)
  UNEXPLAINED                   上のどれでもない差(失敗)

語彙の持ち主: `DiffCause` = ここ。#880 R1a の `Difference`
(domain/backtest/replay_classification.py。
MATCH / NEW_ONLY / …)は『差の形』で別の軸。値は重ならない(契約テストで固定)。
どこからも import されない。package の `__init__.py` へ re-export しない(設計条件 A)。
`DiffCause` の語彙が将来、利用者に見える面へ出るなら、その時点で外部仕様 = USER の確認が要る。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from jstock_advisor.domain.exit_architecture.arbiter import (
    ORIGIN_OF_TRIGGER,
    TRIGGER_TIE_ORDER,
)
from jstock_advisor.domain.exit_architecture.decision import Decision
from jstock_advisor.domain.exit_architecture.profit_taking_adapter import ProfitTakingAdaptation
from jstock_advisor.domain.exit_architecture.vocabulary import (
    Strength,
    SuppressionReason,
    TriggerKind,
)


class DiffCause(StrEnum):
    SAME = "SAME"
    NO_FULL_EVIDENCE = "NO_FULL_EVIDENCE"
    USER_TARGET_ONLY = "USER_TARGET_ONLY"
    ARBITER_STRONGER = "ARBITER_STRONGER"
    UNEXPLAINED = "UNEXPLAINED"


#: 失敗として扱う原因(説明できる差ではない)
FAILURE_CAUSES: Final[frozenset[DiffCause]] = frozenset(
    {DiffCause.ARBITER_STRONGER, DiffCause.UNEXPLAINED}
)


@dataclass(frozen=True)
class DifferenceResult:
    """分類の結果。trigger_kind は NO_FULL_EVIDENCE のときだけ持つ(下がった FULL の経路)。"""

    cause: DiffCause
    trigger_kind: TriggerKind | None = None

    def __post_init__(self) -> None:
        if (self.cause is DiffCause.NO_FULL_EVIDENCE) != (self.trigger_kind is not None):
            raise ValueError("trigger_kind は NO_FULL_EVIDENCE のときだけ持つ")

    @property
    def is_failure(self) -> bool:
        return self.cause in FAILURE_CAUSES


def _max_proposal_strength(adaptation: ProfitTakingAdaptation) -> Strength:
    strengths = [proposal.claimed_strength for proposal in adaptation.proposals]
    return max(strengths, default=Strength.NONE)


def _full_trigger_kind(adaptation: ProfitTakingAdaptation) -> TriggerKind | None:
    """FULL を主張した提案のうち、origin が最大の TriggerKind(同点は TRIGGER_TIE_ORDER の先頭)。"""
    full = [p for p in adaptation.proposals if p.claimed_strength is Strength.FULL]
    if not full:
        return None
    best = max(
        full,
        key=lambda p: (
            int(ORIGIN_OF_TRIGGER[p.trigger_kind]),
            len(TRIGGER_TIE_ORDER) - TRIGGER_TIE_ORDER.index(p.trigger_kind),
        ),
    )
    return best.trigger_kind


def classify_difference(
    e2_final: Strength, decision: Decision, adaptation: ProfitTakingAdaptation
) -> DifferenceResult:
    """E2 の最終の強さ `e2_final` と Arbiter の `decision` の差の原因を分類する(純粋関数)。

    e2_final   現行の E2 の最終の強さ(NONE・WATCH・PARTIAL・FULL)。呼び出し側が trace から求める
    decision   Arbiter の結果
    adaptation E2 の trace を写した adapter の結果(proposals と、通知のみの pass_through)
    """
    if not isinstance(e2_final, Strength):
        raise TypeError("e2_final は Strength")
    if not isinstance(decision, Decision) or not isinstance(adaptation, ProfitTakingAdaptation):
        raise TypeError("decision は Decision・adaptation は ProfitTakingAdaptation")
    arbiter = decision.strength
    if arbiter == e2_final:
        return DifferenceResult(DiffCause.SAME)
    if arbiter > e2_final:
        return DifferenceResult(DiffCause.ARBITER_STRONGER)
    # ここから Arbiter < E2
    if adaptation.pass_through and e2_final > _max_proposal_strength(adaptation):
        # E2 の強さは、Arbiter に渡した提案のどれよりも強い = ユーザー目標だけに由来する
        return DifferenceResult(DiffCause.USER_TARGET_ONLY)
    if (
        e2_final is Strength.FULL
        and arbiter is Strength.PARTIAL
        and any(s.reason is SuppressionReason.NO_FULL_EVIDENCE for s in decision.suppressed)
    ):
        trigger_kind = _full_trigger_kind(adaptation)
        if trigger_kind is not None:
            return DifferenceResult(DiffCause.NO_FULL_EVIDENCE, trigger_kind)
    return DifferenceResult(DiffCause.UNEXPLAINED)
