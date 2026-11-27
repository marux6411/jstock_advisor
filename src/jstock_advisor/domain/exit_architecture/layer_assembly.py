"""mirror の層の組み立て(Issue #878 PR-3c-i。dormant)。

現行のエンジンの結果(保有判断スコア = E3)から、Arbiter への入力 `LayerVerdicts` を
組み立てる純粋関数。
**現行のエンジンに無い層は UNDETERMINED のまま**にし、値を作らない(= `hold_optimal` が成立しない)。
どこからも import されない(配線なし・保存なし・flag なし)。設計: #878 rev2(…6096309230)・rev7・
rev8(…6262530133)・サブちゃんの設計レビュー PASS(…6262490525)。

## 層ごとの扱い(mirror)

* L0 信頼性: 入力なし(= 上限なし。現行のエンジンに L0 が無い)
* L1 投資前提: E3 の結果があれば #882 の adapter(`adapt_holding_decision_to_thesis`)で写す。
  無ければ UNDETERMINED(NOT_EVALUATED)。HOLD にしない
* L2 将来期待リターン: 構成要素なし・valuation_exhaustion / severely_low = UNDETERMINED
  (COMPONENT_NOT_IMPLEMENTED)
* L3 価格の動き: UNDETERMINED(NOT_EVALUATED)。#877 の分類は REGIME_SHADOW(PR-3d)でのみ使う
* L4 機会費用: UNDETERMINED(COMPONENT_NOT_IMPLEMENTED)

## 『読めない記録』の数え方(設計条件 B)

E3 の結果の hard_gate が内部で食い違っている(`InconsistentHardGateError`)ときは、例外を伝播させず
その日の L1 を UNDETERMINED にし、**戻り値の `unreadable_records`(件数)** で数える。
語彙(`UndeterminedReason`)に member を足さない(vocabulary.py を変更しない)。`detail` の自由文には
数えさせず、固定の文言だけを置く。

## 守ること(契約テストで固定)

* package の `__init__.py` へ re-export しない(設計条件 A。本番到達の 7 件の 1 つ)
* services / infrastructure / config / 時計 / I/O を import しない
* `build_candidates`・`arbiter.py`・`vocabulary.py` を変更しない
"""

from __future__ import annotations

from dataclasses import dataclass

from jstock_advisor.domain.entities.holding_decision import HoldingDecisionResult
from jstock_advisor.domain.exit_architecture.determination import Determination
from jstock_advisor.domain.exit_architecture.thesis_adapter import (
    InconsistentHardGateError,
    ThesisAdaptation,
    ThesisMappingPolicy,
    adapt_holding_decision_to_thesis,
)
from jstock_advisor.domain.exit_architecture.verdicts import (
    ExpectedReturnVerdict,
    LayerVerdicts,
    PortfolioContext,
    RegimeVerdict,
    ReliabilityVerdict,
    RotationVerdict,
    ThesisVerdict,
)
from jstock_advisor.domain.exit_architecture.vocabulary import (
    RegimeState,
    ReliabilityClass,
    ThesisState,
    UndeterminedReason,
)

#: 固定の文言(集計の根拠にしない。件数は `unreadable_records` で数える)
DETAIL_NO_E3_RESULT = "mirror: 保有判断スコアの結果が無い"
DETAIL_UNREADABLE_RECORD = "mirror: 保有判断スコアの結果が読めない記録"
DETAIL_L2_NOT_IMPLEMENTED = "mirror: 現行のエンジンに L2 の評価が無い"
DETAIL_L3_NOT_EVALUATED = "mirror: L3 は shadow(PR-3d)でのみ評価する"
DETAIL_L4_NOT_IMPLEMENTED = "mirror: 現行のエンジンに L4 の評価が無い"


@dataclass(frozen=True)
class LayerAssembly:
    """組み立ての結果。

    layers              Arbiter への入力
    unreadable_records  読めない記録の件数(0 または 1。呼び出し側が日・保有ごとに合算する)
    thesis_adaptation   L1 を写した adapter の結果(supporting_only・unmapped を含む)。
                        E3 の結果が無い・読めないときは None
    """

    layers: LayerVerdicts
    unreadable_records: int
    thesis_adaptation: ThesisAdaptation | None

    def __post_init__(self) -> None:
        if isinstance(self.unreadable_records, bool) or self.unreadable_records not in (0, 1):
            raise ValueError("unreadable_records は 0 または 1")
        if self.unreadable_records == 1 and self.thesis_adaptation is not None:
            raise ValueError("読めない記録には thesis_adaptation がない")


def _undetermined_thesis(detail: str) -> ThesisVerdict:
    # L1 の信頼クラスは型として必須だが、Arbiter が見る信頼上限は L0(inputs)だけ。L0 は空 = 上限なし
    return ThesisVerdict(
        thesis_state=Determination[ThesisState].undetermined(
            UndeterminedReason.NOT_EVALUATED, detail
        ),
        reliability=ReliabilityClass.RELIABLE,
    )


def assemble_layers(
    hd_result: HoldingDecisionResult | None,
    policy: ThesisMappingPolicy,
    context: PortfolioContext,
) -> LayerAssembly:
    """mirror の `LayerVerdicts` を組み立てる(純粋関数。入力を変更しない)。

    Raises:
        TypeError: context が PortfolioContext でない(構成の誤り。値を捨てない)
    """
    if not isinstance(context, PortfolioContext):
        raise TypeError("context は PortfolioContext")
    thesis: ThesisVerdict
    adaptation: ThesisAdaptation | None = None
    unreadable = 0
    if hd_result is None:
        thesis = _undetermined_thesis(DETAIL_NO_E3_RESULT)
    else:
        try:
            adaptation = adapt_holding_decision_to_thesis(hd_result, policy)
            thesis = adaptation.verdict
        except InconsistentHardGateError:
            # 成り立たない記録。黙って捨てず、その日の L1 を UNDETERMINED にして件数で数える
            thesis = _undetermined_thesis(DETAIL_UNREADABLE_RECORD)
            adaptation = None
            unreadable = 1
    layers = LayerVerdicts(
        reliability=ReliabilityVerdict(inputs=()),
        thesis=thesis,
        expected_return=ExpectedReturnVerdict(
            components=(),
            valuation_exhaustion=Determination[bool].undetermined(
                UndeterminedReason.COMPONENT_NOT_IMPLEMENTED, DETAIL_L2_NOT_IMPLEMENTED
            ),
            severely_low=Determination[bool].undetermined(
                UndeterminedReason.COMPONENT_NOT_IMPLEMENTED, DETAIL_L2_NOT_IMPLEMENTED
            ),
            reliability=ReliabilityClass.RELIABLE,
        ),
        regime=RegimeVerdict(
            state=Determination[RegimeState].undetermined(
                UndeterminedReason.NOT_EVALUATED, DETAIL_L3_NOT_EVALUATED
            ),
            previous_state=Determination[RegimeState].undetermined(
                UndeterminedReason.NOT_EVALUATED, DETAIL_L3_NOT_EVALUATED
            ),
            current_gain_pct=Determination[float].undetermined(
                UndeterminedReason.NOT_EVALUATED, DETAIL_L3_NOT_EVALUATED
            ),
            peak_gain_pct=Determination[float].undetermined(
                UndeterminedReason.NOT_EVALUATED, DETAIL_L3_NOT_EVALUATED
            ),
        ),
        rotation=RotationVerdict(
            gap_clear=Determination[bool].undetermined(
                UndeterminedReason.COMPONENT_NOT_IMPLEMENTED, DETAIL_L4_NOT_IMPLEMENTED
            ),
        ),
        context=context,
    )
    return LayerAssembly(layers=layers, unreadable_records=unreadable, thesis_adaptation=adaptation)
