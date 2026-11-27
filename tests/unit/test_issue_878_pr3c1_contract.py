"""#878 PR-3c-i(mirror の層の組み立て・Decision の橋・差の原因の分類。dormant)の契約テスト。

設計: #878 rev7(…6262077372)・rev8(…6262530133)・サブちゃんの設計レビュー PASS(…6262490525。
PR-3c-i は USER の 10/13 の例外に当たらない = 条件 A・B つき)。

固定するもの
  ・assemble_layers: L0 は入力なし / L1 は E3 の結果があれば #882 の adapter・無ければ
    UNDETERMINED /
    L2〜L4 は UNDETERMINED(値を作らない)/ hold_optimal が成立しない / FULL は FE が無いと PARTIAL
  ・★ 条件 B ★: 『読めない記録』は例外を伝播させず L1 を UNDETERMINED にし、
    戻り値の件数 field で数える。
    UndeterminedReason は 6 値のまま(member を足さない)・detail の自由文に数えさせない
  ・bridge: Decision -> (action, origin 名)。強度は求めない・review_flag を action に変換しない
  ・classify_difference: SAME / ARBITER_STRONGER(失敗)/ USER_TARGET_ONLY /
    NO_FULL_EVIDENCE(経路つき)
    / UNEXPLAINED(失敗)。R1a の Difference と語彙が重ならない
  ・★ 条件 A ★: package の `__init__.py` は何も import しない・新 module は
    どこからも import されない
  ・既存の build_candidates・arbiter・vocabulary に依存する分岐を持たない(source を読まない)
fixture は合成のみ。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from jstock_advisor.domain.backtest.replay_classification import Difference
from jstock_advisor.domain.entities.enums import RecommendationType
from jstock_advisor.domain.entities.holding_decision import HoldingDecisionHardGate
from jstock_advisor.domain.exit_architecture import arbiter as arbiter_module
from jstock_advisor.domain.exit_architecture import difference_cause as dc
from jstock_advisor.domain.exit_architecture import layer_assembly as la
from jstock_advisor.domain.exit_architecture import legacy_bridge as lb
from jstock_advisor.domain.exit_architecture.arbiter import (
    ORIGIN_OF_TRIGGER,
    ArbiterInput,
    arbitrate,
    undetermined_layers,
)
from jstock_advisor.domain.exit_architecture.decision import Decision, SuppressedCandidate
from jstock_advisor.domain.exit_architecture.determination import Determination
from jstock_advisor.domain.exit_architecture.profit_taking_adapter import (
    PassThroughKind,
    PassThroughNotice,
    ProfitTakingAdaptation,
)
from jstock_advisor.domain.exit_architecture.verdicts import PortfolioContext
from jstock_advisor.domain.exit_architecture.vocabulary import (
    ExitAction,
    ExitClass,
    ExitLayer,
    ReviewFlag,
    Strength,
    SuppressionReason,
    ThesisState,
    TriggerKind,
    UndeterminedReason,
)
from jstock_advisor.domain.signals.profit_taking import _RawLevelOrigin
from tests.unit.test_arbiter_contract import (
    _decide,
    _fe1_layers,
    _inp,
    _layers,
    _policy,
    _proposal,
    _soft,
)
from tests.unit.test_thesis_adapter_contract import _POLICY, _result

_SRC = Path(arbiter_module.__file__).resolve().parents[2]
_PKG = _SRC / "domain" / "exit_architecture"
_NEW = ("layer_assembly.py", "legacy_bridge.py", "difference_cause.py")
_CTX = PortfolioContext(concentrated=False, trading_unit_feasible=True, earnings_window_near=False)


def _assemble(hd: Any = None, ctx: PortfolioContext = _CTX) -> la.LayerAssembly:
    return la.assemble_layers(hd, _POLICY, ctx)


def _mirror_decision(*proposals: Any, hd: Any = None) -> Decision:
    layers = _assemble(hd if hd is not None else _result()).layers
    return arbitrate(ArbiterInput(layers, tuple(proposals), _soft(), _policy())).unwrap()


# =================================================================================================
# assemble_layers
# =================================================================================================


def test_without_an_e3_result_l1_is_undetermined_and_nothing_is_unreadable() -> None:
    out = _assemble(None)
    state = out.layers.thesis.thesis_state
    assert not state.is_determined and state.reason is UndeterminedReason.NOT_EVALUATED
    assert state.detail == la.DETAIL_NO_E3_RESULT
    assert out.unreadable_records == 0 and out.thesis_adaptation is None


def test_with_an_e3_result_l1_is_written_by_the_882_adapter() -> None:
    out = _assemble(_result())
    assert out.layers.thesis.thesis_state == Determination.of(ThesisState.INTACT)
    assert out.unreadable_records == 0
    assert out.thesis_adaptation is not None and out.thesis_adaptation.verdict is out.layers.thesis


def test_l0_has_no_input_so_there_is_no_cap() -> None:
    assert _assemble(_result()).layers.reliability.inputs == ()


def test_l2_l3_l4_are_undetermined_with_the_existing_reasons_and_no_value_is_made() -> None:
    layers = _assemble(_result()).layers
    er, regime, rotation = layers.expected_return, layers.regime, layers.rotation
    assert er.components == ()
    for item, reason in (
        (er.valuation_exhaustion, UndeterminedReason.COMPONENT_NOT_IMPLEMENTED),
        (er.severely_low, UndeterminedReason.COMPONENT_NOT_IMPLEMENTED),
        (regime.state, UndeterminedReason.NOT_EVALUATED),
        (regime.previous_state, UndeterminedReason.NOT_EVALUATED),
        (regime.current_gain_pct, UndeterminedReason.NOT_EVALUATED),
        (regime.peak_gain_pct, UndeterminedReason.NOT_EVALUATED),
        (rotation.gap_clear, UndeterminedReason.COMPONENT_NOT_IMPLEMENTED),
    ):
        assert not item.is_determined and item.reason is reason and item.value is None


def test_the_context_is_passed_through_unchanged() -> None:
    ctx = PortfolioContext(
        concentrated=True, trading_unit_feasible=False, earnings_window_near=True
    )
    assert _assemble(_result(), ctx).layers.context is ctx


@pytest.mark.parametrize("bad", [None, {"concentrated": True}, True])
def test_a_context_that_is_not_a_portfolio_context_is_rejected(bad: Any) -> None:
    with pytest.raises(TypeError):
        la.assemble_layers(_result(), _POLICY, bad)


@pytest.mark.parametrize(("triggered", "codes"), [(True, ()), (False, ("GOING_CONCERN_DOUBT",))])
def test_an_unreadable_record_is_counted_and_never_propagates_or_becomes_a_hold(
    triggered: bool, codes: tuple[str, ...]
) -> None:
    broken = _result().model_copy(
        update={"hard_gate": HoldingDecisionHardGate(triggered=triggered, reason_codes=codes)}
    )
    out = _assemble(broken)
    state = out.layers.thesis.thesis_state
    assert out.unreadable_records == 1  # ★ 戻り値の件数 field で数える(条件 B)★
    assert not state.is_determined and state.reason is UndeterminedReason.NOT_EVALUATED
    assert state.detail == la.DETAIL_UNREADABLE_RECORD  # 固定の文言。集計の根拠にしない
    assert out.thesis_adaptation is None
    decision = _decide(_inp(layers=out.layers))
    assert decision.action is ExitAction.HOLD and decision.hold_optimal is False


def test_the_unreadable_message_never_contains_the_record() -> None:
    broken = _result().model_copy(
        update={"hard_gate": HoldingDecisionHardGate(triggered=True, reason_codes=())}
    )
    detail = _assemble(broken).layers.thesis.thesis_state.detail
    assert "test-result-id" not in detail and "test-holding-id" not in detail


def test_the_assembly_rejects_an_inconsistent_unreadable_count() -> None:
    layers = _assemble(None).layers
    for bad in (2, -1, True):
        with pytest.raises(ValueError):
            la.LayerAssembly(layers, bad, None)  # type: ignore[arg-type]
    adaptation = _assemble(_result()).thesis_adaptation
    with pytest.raises(ValueError):
        la.LayerAssembly(layers, 1, adaptation)


def test_the_assembly_is_deterministic_and_does_not_change_its_input() -> None:
    hd = _result(reasons=())
    before = hd.model_dump()
    assert _assemble(hd) == _assemble(hd)
    assert hd.model_dump() == before


def test_a_mirror_without_proposals_is_never_hold_optimal() -> None:
    """層が確定していないのに『HOLD が最適』が出ない(L2〜L4 を確定と偽る案の不採用の根拠)。"""
    for hd in (None, _result()):
        layers = _assemble(hd).layers
        decision = _decide(_inp(layers=layers))
        assert decision.action is ExitAction.HOLD and decision.hold_optimal is False
        expected = {ExitLayer.L2_EXPECTED_RETURN, ExitLayer.L3_PRICE_REGIME}
        expected |= {ExitLayer.L4_OPPORTUNITY_COST}
        if hd is None:
            expected.add(ExitLayer.L1_THESIS)
        assert set(undetermined_layers(layers)) == expected


def test_full_falls_to_partial_without_a_full_evidence() -> None:
    proposal = _proposal(TriggerKind.FULL_STRONG_CRITICAL, Strength.FULL)  # L1・RISK_EXIT
    decision = _mirror_decision(proposal)
    assert decision.action is ExitAction.PARTIAL
    assert [s.reason for s in decision.suppressed] == [SuppressionReason.NO_FULL_EVIDENCE]


# =================================================================================================
# 条件 B: 語彙を足さない
# =================================================================================================


def test_the_undetermined_reasons_are_exactly_the_six_values() -> None:
    assert sorted(r.value for r in UndeterminedReason) == [
        "COMPONENT_NOT_IMPLEMENTED",
        "COVERAGE_INSUFFICIENT",
        "GUARD_NOT_MET",
        "INPUT_MISSING",
        "NOT_EVALUATED",
        "RELIABILITY_UNUSABLE",
    ]


def test_the_new_modules_only_use_reasons_that_already_exist() -> None:
    used: set[str] = set()
    for node in ast.walk(_tree("layer_assembly.py")):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "UndeterminedReason"
        ):
            used.add(node.attr)
    assert used == {"NOT_EVALUATED", "COMPONENT_NOT_IMPLEMENTED"}


# =================================================================================================
# bridge
# =================================================================================================


def _decision(kind: TriggerKind, strength: Strength) -> Decision:
    return _decide(_inp([_proposal(kind, strength)], layers=_layers()))


@pytest.mark.parametrize(
    ("strength", "action"),
    [
        (Strength.PARTIAL, RecommendationType.PARTIAL_PROFIT_TAKE),
        (Strength.WATCH, RecommendationType.WATCH),
    ],
)
def test_bridge_maps_the_action_of_a_real_decision(
    strength: Strength, action: RecommendationType
) -> None:
    view = lb.bridge(_decision(TriggerKind.PRICE_UPSIDE_MATRIX, strength))
    assert view.action is action


def test_bridge_maps_hold_and_a_full_decision() -> None:
    assert lb.bridge(_decide(_inp())).action is RecommendationType.HOLD
    full = _decide(
        _inp([_proposal(TriggerKind.FULL_STRONG_CRITICAL, Strength.FULL)], layers=_fe1_layers())
    )
    assert full.action is ExitAction.FULL
    view = lb.bridge(full)
    assert view.action is RecommendationType.FULL_PROFIT_TAKE
    assert view.origin_name == _RawLevelOrigin.FUNDAMENTAL_CRITICAL_RISK.name


def test_bridge_origin_names_are_the_current_engines_origin_names() -> None:
    legacy = {origin.name for origin in _RawLevelOrigin}
    seen = set()
    for kind in ORIGIN_OF_TRIGGER:
        view = lb.bridge(_decision(kind, Strength.PARTIAL))
        assert view.origin_name == ORIGIN_OF_TRIGGER[kind].name and view.origin_name in legacy
        seen.add(view.origin_name)
    assert len(seen) == 5  # 5 つの origin を全て実際に通した(空振りでない)


def test_a_hold_has_no_origin_name() -> None:
    assert lb.bridge(_decide(_inp())).origin_name is None


def test_bridge_does_not_turn_a_review_flag_into_an_action() -> None:
    held = Decision(
        ExitAction.HOLD,
        ExitClass.NONE,
        Strength.NONE,
        review_flag=ReviewFlag.URGENT_REVIEW,
    )
    assert lb.bridge(held).action is RecommendationType.HOLD


def test_bridge_returns_only_the_action_and_the_origin_name_no_intensity() -> None:
    assert [f for f in lb.LegacyView.__dataclass_fields__] == ["action", "origin_name"]
    source = (_PKG / "legacy_bridge.py").read_text(encoding="utf-8")
    assert "resolve_sell_intensity" not in source.split('"""', 2)[2]  # docstring の外で呼ばない
    assert "signals" not in "".join(_imports("legacy_bridge.py"))


def test_bridge_rejects_a_non_decision() -> None:
    with pytest.raises(TypeError):
        lb.bridge("HOLD")  # type: ignore[arg-type]


# =================================================================================================
# classify_difference
# =================================================================================================


def _adaptation(*proposals: Any, user_target: bool = False) -> ProfitTakingAdaptation:
    notices = (PassThroughNotice(PassThroughKind.USER_TARGET_PRICE),) if user_target else ()
    return ProfitTakingAdaptation(tuple(proposals), None, notices, ())


def _no_proposal_decision(strength: Strength = Strength.NONE) -> Decision:
    return Decision(ExitAction.HOLD, ExitClass.NONE, strength)


def test_same_strength_is_same() -> None:
    result = dc.classify_difference(
        Strength.PARTIAL,
        _decision(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL),
        _adaptation(),
    )
    assert result == dc.DifferenceResult(dc.DiffCause.SAME) and not result.is_failure


def test_a_stronger_arbiter_is_a_failure() -> None:
    result = dc.classify_difference(
        Strength.WATCH, _decision(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL), _adaptation()
    )
    assert result.cause is dc.DiffCause.ARBITER_STRONGER and result.is_failure


def test_a_difference_that_comes_only_from_the_user_target_is_user_target_only() -> None:
    proposal = _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL)
    result = dc.classify_difference(
        Strength.FULL,
        _decision(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL),
        _adaptation(proposal, user_target=True),
    )
    assert result.cause is dc.DiffCause.USER_TARGET_ONLY and not result.is_failure


def test_without_a_user_target_the_same_gap_is_unexplained() -> None:
    proposal = _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL)
    result = dc.classify_difference(
        Strength.FULL,
        _decision(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL),
        _adaptation(proposal),
    )
    assert result.cause is dc.DiffCause.UNEXPLAINED and result.is_failure


def test_full_to_partial_with_no_full_evidence_names_the_path() -> None:
    proposal = _proposal(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)
    decision = _decision(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)
    assert decision.action is ExitAction.PARTIAL
    result = dc.classify_difference(Strength.FULL, decision, _adaptation(proposal))
    assert result == dc.DifferenceResult(
        dc.DiffCause.NO_FULL_EVIDENCE, TriggerKind.FULL_MODERATE_CONDITIONS
    )


def test_the_path_is_the_full_proposal_with_the_greatest_origin_ties_by_the_fixed_order() -> None:
    low = _proposal(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)  # OTHER_CONDITIONS
    high = _proposal(TriggerKind.FAIR_VALUE_STRONG, Strength.FULL)  # FAIR_VALUE_STRONG
    decision = _decide(_inp([low, high], layers=_layers()))
    assert decision.action is ExitAction.PARTIAL
    for order in ((low, high), (high, low)):  # 入力の並びに依存しない
        result = dc.classify_difference(Strength.FULL, decision, _adaptation(*order))
        assert result.trigger_kind is TriggerKind.FAIR_VALUE_STRONG
    same_origin = [
        _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.FULL),
        _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.FULL),
    ]
    assert (
        dc.classify_difference(Strength.FULL, decision, _adaptation(*same_origin)).trigger_kind
        is TriggerKind.PRICE_UPSIDE_MATRIX
    )


def test_full_to_partial_without_the_no_full_evidence_trace_is_unexplained() -> None:
    proposal = _proposal(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)
    partial = Decision(
        ExitAction.PARTIAL,
        ExitClass.VALUE_EXIT,
        Strength.PARTIAL,
        primary_layer=ExitLayer.L2_EXPECTED_RETURN,
        trigger_kind=TriggerKind.FULL_MODERATE_CONDITIONS,
    )
    result = dc.classify_difference(Strength.FULL, partial, _adaptation(proposal))
    assert result.cause is dc.DiffCause.UNEXPLAINED


def test_a_no_full_evidence_without_a_full_proposal_is_unexplained_not_a_made_up_path() -> None:
    decision = _decision(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)
    result = dc.classify_difference(Strength.FULL, decision, _adaptation())
    assert result.cause is dc.DiffCause.UNEXPLAINED and result.trigger_kind is None


def test_every_pair_of_strengths_gets_a_classification_that_matches_the_order() -> None:
    proposal = _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL)
    for e2 in Strength:
        for arb in Strength:
            decision = (
                _no_proposal_decision(arb)
                if arb <= Strength.WATCH
                else _decision(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL)
            )
            result = dc.classify_difference(e2, decision, _adaptation(proposal))
            if decision.strength == e2:
                assert result.cause is dc.DiffCause.SAME
            elif decision.strength > e2:
                assert result.cause is dc.DiffCause.ARBITER_STRONGER
            else:
                assert result.cause in set(dc.DiffCause) - {
                    dc.DiffCause.SAME,
                    dc.DiffCause.ARBITER_STRONGER,
                }


def test_a_gap_equal_to_the_strongest_proposal_is_not_attributed_to_the_user_target() -> None:
    """境界: E2 の強さが提案の最大と等しいなら、ユーザー目標だけに由来するとは言えない。"""
    proposal = _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.PARTIAL)
    arbiter_hold = _no_proposal_decision(
        Strength.NONE
    )  # 提案があるのに Arbiter は HOLD(例: 層が未確定)
    result = dc.classify_difference(
        Strength.PARTIAL, arbiter_hold, _adaptation(proposal, user_target=True)
    )
    assert result.cause is dc.DiffCause.UNEXPLAINED


def test_a_tie_in_origin_is_broken_by_the_fixed_trigger_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """origin が同じ FULL 候補が 2 つ並んだとき、経路は TRIGGER_TIE_ORDER の先頭になる。

    現行の語彙では FULL を主張できる経路の origin は全て異なるため、origin を揃えて固定する
    (将来、経路が増えて origin が重なったときの決定性を今のうちに固定する)。
    """
    low = _proposal(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)
    high = _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.FULL)
    decision = _decide(_inp([low, high], layers=_layers()))
    monkeypatch.setitem(
        dc.ORIGIN_OF_TRIGGER,
        TriggerKind.FULL_MODERATE_CONDITIONS,
        ORIGIN_OF_TRIGGER[TriggerKind.PRICE_UPSIDE_MATRIX],
    )
    for order in ((low, high), (high, low)):
        result = dc.classify_difference(Strength.FULL, decision, _adaptation(*order))
        assert result.trigger_kind is TriggerKind.PRICE_UPSIDE_MATRIX  # 順序の先頭


def test_a_no_full_evidence_trace_with_a_decision_below_partial_is_unexplained() -> None:
    """反例(サブちゃんの SHOULD-1): E2 = FULL・Arbiter = WATCH・trace に NO_FULL_EVIDENCE。

    FULL -> PARTIAL で NO_FULL_EVIDENCE を記録した後、信頼性の降格で WATCH まで落ちる経路。
    『FULL の独立根拠が無いから PARTIAL に下がった』では説明できない(Arbiter は PARTIAL ですらない)
    ので、NO_FULL_EVIDENCE ではなく UNEXPLAINED。
    """
    proposal = _proposal(TriggerKind.FULL_MODERATE_CONDITIONS, Strength.FULL)
    traced = SuppressedCandidate(
        ExitClass.VALUE_EXIT, Strength.FULL, SuppressionReason.NO_FULL_EVIDENCE
    )
    watch = Decision(ExitAction.HOLD, ExitClass.NONE, Strength.WATCH, suppressed=(traced,))
    result = dc.classify_difference(Strength.FULL, watch, _adaptation(proposal))
    assert result == dc.DifferenceResult(dc.DiffCause.UNEXPLAINED)


def test_the_path_ignores_a_stronger_origin_that_did_not_claim_full() -> None:
    """反例(サブちゃんの SHOULD-2): FAIR_VALUE_PARTIAL_GATE(PARTIAL・origin 4)+ PRICE_UPSIDE_MATRIX
    (FULL・origin 2)。『下がった FULL の経路』は FULL を主張した側(PRICE_UPSIDE_MATRIX)だけ。
    """
    partial = _proposal(TriggerKind.FAIR_VALUE_PARTIAL_GATE, Strength.PARTIAL)
    full = _proposal(TriggerKind.PRICE_UPSIDE_MATRIX, Strength.FULL)
    assert (
        ORIGIN_OF_TRIGGER[TriggerKind.FAIR_VALUE_PARTIAL_GATE]
        > ORIGIN_OF_TRIGGER[TriggerKind.PRICE_UPSIDE_MATRIX]
    )  # 前提: FULL を主張していない側のほうが origin は大きい
    decision = _decide(_inp([partial, full], layers=_layers()))
    assert decision.action is ExitAction.PARTIAL
    assert any(s.reason is SuppressionReason.NO_FULL_EVIDENCE for s in decision.suppressed)
    for order in ((partial, full), (full, partial)):
        result = dc.classify_difference(Strength.FULL, decision, _adaptation(*order))
        assert result == dc.DifferenceResult(
            dc.DiffCause.NO_FULL_EVIDENCE, TriggerKind.PRICE_UPSIDE_MATRIX
        )


def _expected_cause(
    e2: Strength, arbiter: Strength, *, user_target: bool, max_proposal: Strength, traced_full: bool
) -> dc.DiffCause:
    """区分ごとの完全一致の期待値(実装とは独立に、設計の定義から書く)。"""
    if arbiter == e2:
        return dc.DiffCause.SAME
    if arbiter > e2:
        return dc.DiffCause.ARBITER_STRONGER
    if user_target and e2 > max_proposal:
        return dc.DiffCause.USER_TARGET_ONLY
    if e2 is Strength.FULL and arbiter is Strength.PARTIAL and traced_full:
        return dc.DiffCause.NO_FULL_EVIDENCE
    return dc.DiffCause.UNEXPLAINED


def _decision_with(strength: Strength, *, traced_full: bool) -> Decision:
    suppressed = (
        (
            SuppressedCandidate(
                ExitClass.VALUE_EXIT, Strength.FULL, SuppressionReason.NO_FULL_EVIDENCE
            ),
        )
        if traced_full
        else ()
    )
    if strength is Strength.PARTIAL:
        return Decision(
            ExitAction.PARTIAL,
            ExitClass.VALUE_EXIT,
            Strength.PARTIAL,
            primary_layer=ExitLayer.L2_EXPECTED_RETURN,
            trigger_kind=TriggerKind.FULL_MODERATE_CONDITIONS,
            suppressed=suppressed,
        )
    if strength is Strength.FULL:
        return _decide(
            _inp(
                [_proposal(TriggerKind.FULL_STRONG_CRITICAL, Strength.FULL)],
                layers=_fe1_layers(),
            )
        )
    return Decision(ExitAction.HOLD, ExitClass.NONE, strength, suppressed=suppressed)


@pytest.mark.parametrize("user_target", [False, True])
@pytest.mark.parametrize("traced_full", [False, True])
@pytest.mark.parametrize("max_proposal", [Strength.NONE, Strength.PARTIAL, Strength.FULL])
def test_every_classification_matches_the_exact_expected_cause(
    user_target: bool, traced_full: bool, max_proposal: Strength
) -> None:
    """区分ごとの完全一致(『3 値のどれか』の集合所属ではなく、1 組ずつ期待する原因を固定する)。"""
    proposals = (
        ()
        if max_proposal is Strength.NONE
        else (
            _proposal(
                TriggerKind.FULL_MODERATE_CONDITIONS
                if max_proposal is Strength.FULL
                else TriggerKind.PRICE_UPSIDE_MATRIX,
                max_proposal,
            ),
        )
    )
    adaptation = _adaptation(*proposals, user_target=user_target)
    for e2 in Strength:
        for arbiter in Strength:
            decision = _decision_with(arbiter, traced_full=traced_full)
            expected = _expected_cause(
                e2,
                decision.strength,
                user_target=user_target,
                max_proposal=max_proposal,
                traced_full=traced_full,
            )
            got = dc.classify_difference(e2, decision, adaptation).cause
            has_full_proposal = max_proposal is Strength.FULL
            if expected is dc.DiffCause.NO_FULL_EVIDENCE and not has_full_proposal:
                expected = (
                    dc.DiffCause.UNEXPLAINED
                )  # 経路を作り話にしない(FULL を主張した提案が無い)
            assert got is expected, (
                e2.name,
                arbiter.name,
                user_target,
                traced_full,
                max_proposal.name,
            )


def test_the_difference_result_holds_a_path_only_for_no_full_evidence() -> None:
    with pytest.raises(ValueError):
        dc.DifferenceResult(dc.DiffCause.SAME, TriggerKind.PRICE_UPSIDE_MATRIX)
    with pytest.raises(ValueError):
        dc.DifferenceResult(dc.DiffCause.NO_FULL_EVIDENCE)


def test_the_failure_causes_are_exactly_the_stronger_and_the_unexplained() -> None:
    assert {dc.DiffCause.ARBITER_STRONGER, dc.DiffCause.UNEXPLAINED} == dc.FAILURE_CAUSES


def test_the_vocabulary_is_the_agreed_five_and_does_not_overlap_the_r1a_difference() -> None:
    assert sorted(c.value for c in dc.DiffCause) == [
        "ARBITER_STRONGER",
        "NO_FULL_EVIDENCE",
        "SAME",
        "UNEXPLAINED",
        "USER_TARGET_ONLY",
    ]
    assert {c.value for c in dc.DiffCause}.isdisjoint({d.value for d in Difference})


@pytest.mark.parametrize("bad", [(1, None, None), ("FULL", None, None), (Strength.FULL, 1, None)])
def test_classify_rejects_wrong_types(bad: Any) -> None:
    e2, decision, adaptation = bad
    with pytest.raises(TypeError):
        dc.classify_difference(e2, decision or _no_proposal_decision(), adaptation or _adaptation())


# =================================================================================================
# 条件 A・dormant・既存の網に触れない(AST)
# =================================================================================================


def _tree(name: str) -> ast.Module:
    return ast.parse((_PKG / name).read_text(encoding="utf-8"))


def _imports(name: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
    return found


def test_the_package_init_imports_nothing_so_no_new_module_is_reachable_through_it() -> None:
    """★ 条件 A ★ `__init__.py` が submodule を import すると新 module が本番到達になる(#606)。"""
    tree = ast.parse((_PKG / "__init__.py").read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    ]


@pytest.mark.parametrize("name", _NEW)
def test_nothing_in_src_imports_the_new_module(name: str) -> None:
    stem = name.removesuffix(".py")
    importers = sorted(
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if path != _PKG / name and f"exit_architecture.{stem}" in path.read_text(encoding="utf-8")
    )
    assert importers == []  # 到達の根が無い(R2 / PR-4b で配線するとき、意図して更新する)


@pytest.mark.parametrize("name", _NEW)
def test_the_new_modules_do_not_read_the_source_attribute_or_the_gate(name: str) -> None:
    attributes = {n.attr for n in ast.walk(_tree(name)) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(_tree(name)) if isinstance(n, ast.Name)}
    assert "source" not in attributes and "ProposalSource" not in names
    assert "build_candidates" not in names | attributes and "_layer_determined" not in names


def test_the_only_names_taken_from_the_arbiter_are_the_two_read_only_tables() -> None:
    taken: set[str] = set()
    for name in _NEW:
        for node in ast.walk(_tree(name)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.endswith(".arbiter")
            ):
                taken.update(alias.name for alias in node.names)
    assert taken == {"ORIGIN_OF_TRIGGER", "TRIGGER_TIE_ORDER"}


def test_the_new_modules_import_no_services_config_or_clock() -> None:
    forbidden = ("services", "infrastructure", "config", "lambda_handlers", "providers", "cli")
    for name in _NEW:
        for module in _imports(name):
            assert not module.startswith(tuple(f"jstock_advisor.{f}" for f in forbidden)), name
            assert module.split(".")[0] in {
                "__future__",
                "dataclasses",
                "enum",
                "typing",
                "jstock_advisor",
            }, name
