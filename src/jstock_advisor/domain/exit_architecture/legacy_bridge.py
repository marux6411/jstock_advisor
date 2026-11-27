"""Arbiter の Decision を、現行のエンジンの語彙(action・origin 名)へ写す(#878 PR-3c-i。dormant)。

新旧の結果を並べて比べるための橋。**強度(sell_intensity)は求めない**: `resolve_sell_intensity` は
package の外(`domain/signals/profit_taking.py`)にあり、規則を複製しない・package から呼ばない
(rev4 の C-2)。強度は呼び出し側(契約テスト・PR-4b)が
`resolve_sell_intensity(origin 名, momentum)` から求める。

どこからも import されない。`review_flag` / `review_class`(人の確認を要する軸)は action へ変換しない
(数量を伴わない確認を PARTIAL / FULL に自動変換しない = UJ-15)。
package の `__init__.py` へ re-export しない(設計条件 A)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jstock_advisor.domain.entities.enums import RecommendationType
from jstock_advisor.domain.exit_architecture.arbiter import ORIGIN_OF_TRIGGER
from jstock_advisor.domain.exit_architecture.decision import Decision
from jstock_advisor.domain.exit_architecture.vocabulary import ExitAction, Strength

_ACTION_OF: Final[dict[ExitAction, RecommendationType]] = {
    ExitAction.PARTIAL: RecommendationType.PARTIAL_PROFIT_TAKE,
    ExitAction.FULL: RecommendationType.FULL_PROFIT_TAKE,
}


@dataclass(frozen=True)
class LegacyView:
    """現行のエンジンの語彙で見た Decision。

    action       現行の RecommendationType(HOLD / WATCH / PARTIAL_PROFIT_TAKE / FULL_PROFIT_TAKE)
    origin_name  成立経路の名前(現行の `_RawLevelOrigin` の名前と同じ。HOLD は None)
    """

    action: RecommendationType
    origin_name: str | None


def bridge(decision: Decision) -> LegacyView:
    """Decision -> LegacyView(純粋関数)。

    HOLD は strength が WATCH なら WATCH、それ以外は HOLD。PARTIAL / FULL は利確の action。
    origin 名は勝った候補の TriggerKind から決める(無ければ None)。
    """
    if not isinstance(decision, Decision):
        raise TypeError("decision は Decision")
    if decision.action is ExitAction.HOLD:
        action = (
            RecommendationType.WATCH
            if decision.strength is Strength.WATCH
            else RecommendationType.HOLD
        )
    else:
        action = _ACTION_OF[decision.action]
    origin_name = (
        ORIGIN_OF_TRIGGER[decision.trigger_kind].name if decision.trigger_kind is not None else None
    )
    return LegacyView(action=action, origin_name=origin_name)
