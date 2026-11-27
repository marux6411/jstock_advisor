"""#878 PR-4a: ProfitTakingOutcome の非永続 field `exit_trace` の契約(dormant・誰にも読まれない)。

実判定の 1 回の評価の経過(ExitTrace)だけを、判定へ使わずに持つ。設計 = #878 rev10 §3・rev11 §2・
§3(…6278955325 / …6279142222)。実装の機構は、service が evaluate_profit_taking_traced を呼ぶ案 i
ではなく、domain の `evaluate_profit_taking` の省略可能な出力先 `trace_sink` を実判定の 1 回だけが
渡す形(既存の約 35 件のテストが service の `evaluate_profit_taking` を patch して実判定を固定して
いるため。意図 = 『反実仮想は構造的に trace を作れない』は同じ)。

固定すること:
* domain: sink を渡すと、その 1 回の ExitTrace が 1 件追加される(traced 版の結果と一致)。
  渡さなければ従来と同一。例外のときは追加しない
* service: 実判定の 1 回だけが sink を渡す(AST)。service は traced を参照しない。反実仮想は渡せない
* outcome: HOLD でも推奨ありでも exit_trace が入る。等価比較・repr に含まれない。data_error はNone
* 判定の同一性: exit_trace の有無で、outcome・監査記録(gate_trace を含む)は完全一致
* ★ `exit_trace` を読む src は 0 件 ★(PR-4b の配線が census の差分として見える)
"""

from __future__ import annotations

import ast
import dataclasses
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from jstock_advisor.domain.entities.enums import RecommendationType
from jstock_advisor.domain.signals.profit_taking import (
    ExitTrace,
    InvalidProfitTakingInputError,
    ProfitTakingResult,
    evaluate_profit_taking,
    evaluate_profit_taking_traced,
)
from jstock_advisor.services import profit_taking_service as service_module
from jstock_advisor.services.profit_taking_service import ProfitTakingOutcome, ProfitTakingService
from tests.support import profit_taking_cases as pc
from tests.unit.test_issue_720_profit_taking_gate_trace import _analyze
from tests.unit.test_profit_taking_service import _canned_result

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "jstock_advisor"
_SERVICE_PATH = _SRC / "services" / "profit_taking_service.py"


def _trace_of(case_kwargs: dict[str, Any]) -> ExitTrace:
    _, trace = evaluate_profit_taking_traced(**case_kwargs)
    return trace


# =================================================================================================
# domain: trace_sink
# =================================================================================================


def test_the_sink_receives_exactly_the_trace_of_that_evaluation() -> None:
    checked = 0
    for case in pc.sampled_cases(300):
        try:
            expected_result, expected_trace = evaluate_profit_taking_traced(**case.kwargs)
        except InvalidProfitTakingInputError:
            continue
        sink: list[ExitTrace] = []
        result = evaluate_profit_taking(**case.kwargs, trace_sink=sink)
        assert sink == [expected_trace]
        assert result == expected_result == evaluate_profit_taking(**case.kwargs)
        checked += 1
    assert checked >= 250


def test_without_a_sink_nothing_changes_and_the_default_is_none() -> None:
    import inspect

    parameter = inspect.signature(evaluate_profit_taking).parameters["trace_sink"]
    assert parameter.default is None
    case = next(iter(pc.sampled_cases(1)))
    assert evaluate_profit_taking(**case.kwargs) == evaluate_profit_taking(
        **case.kwargs, trace_sink=None
    )


def test_an_invalid_input_leaves_the_sink_empty() -> None:
    case = next(iter(pc.sampled_cases(1)))
    kwargs = dict(case.kwargs)
    kwargs["average_purchase_price"] = kwargs["average_purchase_price"] * 0
    sink: list[ExitTrace] = []
    with pytest.raises(InvalidProfitTakingInputError):
        evaluate_profit_taking(**kwargs, trace_sink=sink)
    assert sink == []


# =================================================================================================
# service: 実判定の 1 回だけが sink を渡す(構造)
# =================================================================================================


def _service_tree() -> ast.Module:
    return ast.parse(_SERVICE_PATH.read_text(encoding="utf-8"))


def test_the_service_never_references_the_traced_function() -> None:
    names = {n.id for n in ast.walk(_service_tree()) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(_service_tree()) if isinstance(n, ast.Attribute)}
    imported = {
        alias.name
        for n in ast.walk(_service_tree())
        if isinstance(n, ast.ImportFrom)
        for alias in n.names
    }
    assert "evaluate_profit_taking_traced" not in names | attrs | imported


def test_only_one_call_site_passes_a_sink_and_the_counterfactual_gets_the_function_unchanged() -> (
    None
):
    tree = _service_tree()
    sink_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_evaluate"
        and any(k.arg == "trace_sink" for k in n.keywords)
    ]
    assert len(sink_calls) == 1  # 実判定の 1 回
    # 反実仮想: `_evaluate` は callable としてそのまま build_profit_taking_gate_trace へ渡す
    gate_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "build_profit_taking_gate_trace"
    ]
    assert len(gate_calls) == 1
    passed = [a.id for a in gate_calls[0].args if isinstance(a, ast.Name)]
    assert "_evaluate" in passed
    # `_evaluate` の本体は untraced の wrapper を呼び続ける(戻り値の型は ProfitTakingResult のまま)
    evaluate_defs = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_evaluate"
    ]
    assert len(evaluate_defs) == 1
    inner = [
        c.func.id
        for c in ast.walk(evaluate_defs[0])
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    ]
    assert inner == ["evaluate_profit_taking"]
    returns = evaluate_defs[0].returns
    assert isinstance(returns, ast.Name) and returns.id == "ProfitTakingResult"


# =================================================================================================
# outcome: field の性質・値
# =================================================================================================


def _field() -> dataclasses.Field[Any]:
    return next(f for f in dataclasses.fields(ProfitTakingOutcome) if f.name == "exit_trace")


def test_the_field_is_non_persistent_default_none_and_hidden_from_equality_and_repr() -> None:
    field = _field()
    assert field.default is None
    assert field.compare is False
    assert field.repr is False
    base = ProfitTakingOutcome("0001", None, None)
    with_trace = dataclasses.replace(
        base, exit_trace=_trace_of(next(iter(pc.sampled_cases(1))).kwargs)
    )
    assert base == with_trace
    assert "exit_trace" not in repr(with_trace)


_SENTINEL = _trace_of(next(iter(pc.sampled_cases(1))).kwargs)


def _without_sink(call: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in call.items() if k != "trace_sink"}


def _filling_fake(calls: list[dict[str, Any]], action: RecommendationType):  # type: ignore[no-untyped-def]
    """sink を渡された呼び出しにだけ ExitTrace を入れる偽の evaluate(本物の wrapper と同じ作法)。"""

    def _evaluate(**kwargs: Any) -> ProfitTakingResult:
        calls.append(kwargs)
        sink = kwargs.get("trace_sink")
        if sink is not None:
            sink.append(_SENTINEL)
        return dataclasses.replace(_canned_result(action), final_action=action, origin="NONE")

    return _evaluate


def _ignoring_fake(calls: list[dict[str, Any]], action: RecommendationType):  # type: ignore[no-untyped-def]
    def _evaluate(**kwargs: Any) -> ProfitTakingResult:
        calls.append(kwargs)
        return dataclasses.replace(_canned_result(action), final_action=action, origin="NONE")

    return _evaluate


@pytest.mark.parametrize(
    "action",
    [RecommendationType.HOLD, RecommendationType.WATCH, RecommendationType.FULL_PROFIT_TAKE],
)
def test_the_outcome_carries_the_trace_of_the_real_judgment_only(
    monkeypatch: pytest.MonkeyPatch, action: RecommendationType
) -> None:
    calls: list[dict[str, Any]] = []
    outcome, _, _ = _analyze(monkeypatch, shares=300, fake=_filling_fake(calls, action))

    assert outcome.exit_trace is _SENTINEL
    # 実判定(最初の呼び出し)だけが sink を持ち、反実仮想の再評価は持たない
    assert isinstance(calls[0]["trace_sink"], list)
    assert all(c.get("trace_sink") is None for c in calls[1:])


def test_the_outcome_has_no_trace_when_the_evaluation_did_not_provide_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _, _ = _analyze(
        monkeypatch, shares=300, fake=_ignoring_fake([], RecommendationType.HOLD)
    )
    assert outcome.exit_trace is None


def test_an_invalid_input_outcome_has_no_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raises(**kwargs: Any) -> ProfitTakingResult:
        raise InvalidProfitTakingInputError("average_purchase_price", Decimal("0"), "boom")

    outcome, _, _ = _analyze_allowing_no_audit(monkeypatch, _raises)
    assert outcome.exit_trace is None
    assert outcome.recommendation is None


def _analyze_allowing_no_audit(
    monkeypatch: pytest.MonkeyPatch, fake: Any
) -> tuple[ProfitTakingOutcome, None, None]:
    from tests.unit.test_profit_taking_service import _CONFIG, _NOW, _holding, _providers

    monkeypatch.setattr(service_module, "evaluate_profit_taking", fake)
    service = ProfitTakingService(providers=_providers(None, None), config=_CONFIG)
    return service.analyze(_holding("2914"), _NOW), None, None


# =================================================================================================
# 判定の同一性: exit_trace の有無で outcome・監査記録(gate_trace を含む)は完全一致
# =================================================================================================


@pytest.mark.parametrize("shares", [100, 300])
@pytest.mark.parametrize("action", [RecommendationType.WATCH, RecommendationType.FULL_PROFIT_TAKE])
def test_the_judgment_the_audit_record_and_the_gate_trace_are_identical_with_or_without_a_trace(
    monkeypatch: pytest.MonkeyPatch, shares: int, action: RecommendationType
) -> None:
    calls_with: list[dict[str, Any]] = []
    calls_without: list[dict[str, Any]] = []
    outcome_a, values_a, _ = _analyze(
        monkeypatch, shares=shares, fake=_filling_fake(calls_with, action)
    )
    outcome_b, values_b, _ = _analyze(
        monkeypatch, shares=shares, fake=_ignoring_fake(calls_without, action)
    )

    # recommendation_id は実行ごとの乱数。それ以外の全 field が一致(exit_trace は compare=False)
    assert outcome_a.recommendation is not None and outcome_b.recommendation is not None
    ignore = {"recommendation_id"}
    assert outcome_a.recommendation.model_dump(
        exclude=ignore
    ) == outcome_b.recommendation.model_dump(exclude=ignore)
    assert dataclasses.replace(outcome_a, recommendation=None) == dataclasses.replace(
        outcome_b, recommendation=None
    )
    assert outcome_a.exit_trace is not None and outcome_b.exit_trace is None
    assert values_a == values_b  # gate_trace(profit_taking_gate_trace)を含む監査記録の全体
    assert "profit_taking_gate_trace" in values_a
    # 評価の呼び出し回数・各呼び出しの条件入力も同じ(trace_sink 以外の引数が同一)
    assert len(calls_with) == len(calls_without)
    assert [_without_sink(c) for c in calls_with] == [_without_sink(c) for c in calls_without]


def test_a_service_run_with_the_real_evaluation_fills_a_real_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本物の evaluate_profit_taking(patch しない `_fake_evaluate` ではなく透過の spy)を通す。"""
    seen: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> ProfitTakingResult:
        seen.append({k: v for k, v in kwargs.items() if k != "trace_sink"})
        return evaluate_profit_taking(**kwargs)

    outcome, _, _ = _analyze(monkeypatch, shares=300, fake=spy)
    assert isinstance(outcome.exit_trace, ExitTrace)
    # 実判定の kwargs で traced を直接呼んだ結果の trace と一致する
    assert outcome.exit_trace == _trace_of(seen[0])


# =================================================================================================
# ★ exit_trace を読む src は 0 件 ★(PR-4b の配線が census の差分として見える)
# =================================================================================================


def _py_files() -> list[Path]:
    return sorted(p for p in _SRC.rglob("*.py"))


def test_no_src_reads_exit_trace() -> None:
    readers: list[str] = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "exit_trace":
                readers.append(f"{path.relative_to(_SRC).as_posix()}:{node.lineno}")
            if isinstance(node, ast.Constant) and node.value == "exit_trace":
                readers.append(f"{path.relative_to(_SRC).as_posix()}:{node.lineno}")
    assert readers == []


def test_exit_trace_is_written_only_by_the_profit_taking_service() -> None:
    writers: list[str] = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "exit_trace":
                writers.append(path.relative_to(_SRC).as_posix())
    assert (
        writers == ["services/profit_taking_service.py"] * 3
    )  # HOLD・fail-closed・推奨ありの 3 経路


def test_the_service_does_not_reach_the_exit_architecture_package() -> None:
    """#606・C0 を変えない(本 PR は exit_architecture を一切参照しない)。"""
    tree = _service_tree()
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
        alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names
    ]
    assert [m for m in modules if "exit_architecture" in m] == []


# =================================================================================================
# 審査(サブちゃん …6279590950 / HANAKO …6279487565)の条件: 形 C の網・None のとき従来と完全に同一・
# sink を読まない
# =================================================================================================

_DOMAIN_PATH = _SRC / "domain" / "signals" / "profit_taking.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, name
    return found[0]


def _parents(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def test_the_evaluate_closure_takes_a_keyword_only_sink_defaulting_to_none() -> None:
    """形 C: `_evaluate(inputs, *, trace_sink=None)`。既定 None = 反実仮想は sink を持たない。"""
    args = _function(_service_tree(), "_evaluate").args
    assert [a.arg for a in args.args] == ["inputs"]
    assert [a.arg for a in args.kwonlyargs] == ["trace_sink"]
    assert len(args.kw_defaults) == 1
    default = args.kw_defaults[0]
    assert isinstance(default, ast.Constant) and default.value is None


def test_the_counterfactual_call_expression_never_mentions_the_sink() -> None:
    tree = _service_tree()
    gate_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "build_profit_taking_gate_trace"
    ]
    assert len(gate_calls) == 1
    mentions = [
        n
        for n in ast.walk(gate_calls[0])
        if (isinstance(n, ast.Name) and "sink" in n.id)
        or (isinstance(n, ast.keyword) and n.arg is not None and "sink" in n.arg)
    ]
    assert mentions == []


def _sink_uses_are_write_only(function: ast.FunctionDef) -> list[str]:
    """`trace_sink` の使われ方のうち、許す形(`is not None` の比較・`.append(...)` の呼び出し・
    keyword 引数としての受け渡し)以外を返す。中身を読む・長さを見る・分岐に使う形は全て違反。"""
    parents = _parents(function)
    violations: list[str] = []
    for node in ast.walk(function):
        if not (isinstance(node, ast.Name) and node.id == "trace_sink"):
            continue
        parent = parents[node]
        if isinstance(parent, ast.keyword):
            continue  # 受け渡し
        if (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], ast.IsNot)
            and isinstance(parent.comparators[0], ast.Constant)
            and parent.comparators[0].value is None
        ):
            continue  # `trace_sink is not None`
        if isinstance(parent, ast.Attribute) and parent.attr == "append":
            call = parents[parent]
            if isinstance(call, ast.Call) and call.func is parent:
                continue  # `trace_sink.append(...)`
        violations.append(f"line {node.lineno}: {ast.dump(parent)[:60]}")
    return violations


def test_the_sink_is_write_only_in_the_wrapper_and_in_the_closure() -> None:
    """出力口が入力に化けない(判定が前回の trace に依存しない)。中身を読む分岐が無い。"""
    assert (
        _sink_uses_are_write_only(
            _function(ast.parse(_DOMAIN_PATH.read_text("utf-8")), "evaluate_profit_taking")
        )
        == []
    )
    assert _sink_uses_are_write_only(_function(_service_tree(), "_evaluate")) == []


def test_the_wrapper_with_no_sink_is_structurally_identical_to_the_pre_change_wrapper() -> None:
    """None のとき従来と完全に同一(構造の主張): 追加された引数は末尾の `trace_sink`(既定 None)だけ。
    追加された文は `if trace_sink is not None:` の 1 分岐だけ。traced 版への引数(従来の 14 個)は
    同じ順序のまま不変・`return result` は不変。"""
    function = _function(ast.parse(_DOMAIN_PATH.read_text("utf-8")), "evaluate_profit_taking")
    params = [a.arg for a in function.args.args]
    assert params[-1] == "trace_sink" and len(params) == 15
    assert function.args.kwonlyargs == []
    default = function.args.defaults[-1]
    assert isinstance(default, ast.Constant) and default.value is None
    legacy = params[:-1]
    assert legacy[0] == "current_price" and legacy[-1] == "is_benefit_eligible"
    docstring_free = [
        s
        for s in function.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    assert [type(s).__name__ for s in docstring_free] == ["Assign", "If", "Return"]
    assign, branch, ret = docstring_free
    assert isinstance(assign, ast.Assign) and isinstance(assign.value, ast.Call)
    assert isinstance(assign.targets[0], ast.Tuple)
    assert [t.id for t in assign.targets[0].elts if isinstance(t, ast.Name)] == ["result", "trace"]
    assert [a.id for a in assign.value.args if isinstance(a, ast.Name)] == legacy
    assert assign.value.keywords == []
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Name) and ret.value.id == "result"
    assert isinstance(branch, ast.If) and len(branch.body) == 1 and branch.orelse == []


def test_exactly_one_trace_is_collected_even_though_the_counterfactual_ran_many_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """『trace は 1 本だけ在る』を件数で固定(形 A の事故 = N+1 本が混ざる、を捕まえる)。"""
    calls: list[dict[str, Any]] = []
    _analyze(
        monkeypatch, shares=100, fake=_filling_fake(calls, RecommendationType.FULL_PROFIT_TAKE)
    )
    assert len(calls) >= 3  # 実判定 1 + 反実仮想の再評価(遮断ごと + JOINT)
    sink = calls[0]["trace_sink"]
    assert len(sink) == 1
    assert all(c.get("trace_sink") is None for c in calls[1:])


# =================================================================================================
# 審査(サブちゃん …6280537592)の SHOULD-1(案 A)/ SHOULD-2 への対応
# =================================================================================================


def test_the_partial_path_carries_the_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """PARTIAL(売却数量が確定する正常な経路)でも、実判定の trace が入る。"""
    calls: list[dict[str, Any]] = []
    outcome, _, _ = _analyze(
        monkeypatch,
        shares=300,
        fake=_filling_fake(calls, RecommendationType.PARTIAL_PROFIT_TAKE),
    )
    assert outcome.recommendation is not None
    assert outcome.exit_trace is _SENTINEL


def test_the_fail_closed_partial_path_still_carries_the_trace_of_the_real_judgment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ SHOULD-1(案 A)★: PARTIAL なのに売却数量が確定しない fail-closed の経路
    (recommendation = None・data_error あり)は実判定の後なので、trace を持つ。
    `exit_trace is None` は『実判定が走らなかった』(snapshot 取得失敗・不正入力)だけを意味し、
    『走ったが fail-closed で棄却した』と区別できる。"""
    calls: list[dict[str, Any]] = []

    def _fail_closed_fake(**kwargs: Any) -> ProfitTakingResult:
        calls.append(kwargs)
        sink = kwargs.get("trace_sink")
        if sink is not None:
            sink.append(_SENTINEL)
        action = RecommendationType.PARTIAL_PROFIT_TAKE
        # sell_intensity = None の PARTIAL は、サービスの不変条件チェックで fail-closed になる
        return dataclasses.replace(
            _canned_result(action, sell_intensity=None), final_action=action, origin="NONE"
        )

    outcome, _, _ = _analyze(monkeypatch, shares=300, fake=_fail_closed_fake)
    assert outcome.recommendation is None
    assert outcome.data_error is not None and "fail-closed" in outcome.data_error
    assert outcome.exit_trace is _SENTINEL  # 判定を棄却したことと、trace の有無は別


def test_exit_trace_is_none_only_when_the_real_judgment_did_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """評価の前の data_error(不正入力)は None。評価が走った経路は、trace を供給する限り値を持つ。"""

    def _raises(**kwargs: Any) -> ProfitTakingResult:
        raise InvalidProfitTakingInputError("average_purchase_price", Decimal("0"), "boom")

    outcome, _, _ = _analyze_allowing_no_audit(monkeypatch, _raises)
    assert outcome.exit_trace is None


def test_a_non_empty_sink_is_only_appended_to_and_the_result_is_unchanged() -> None:
    """SHOULD-2: 空でない sink を渡しても、結果は従来と同一で、追加はちょうど 1 件(len が 1 → 2)。
    sink の中身を読む分岐があれば(例: 空のときだけ追加)、挙動の面でもここで落ちる。"""
    case = next(iter(pc.sampled_cases(1)))
    expected_result, expected_trace = evaluate_profit_taking_traced(**case.kwargs)
    pre_existing = _SENTINEL
    sink: list[ExitTrace] = [pre_existing]
    result = evaluate_profit_taking(**case.kwargs, trace_sink=sink)
    assert result == expected_result
    assert len(sink) == 2
    assert sink[0] is pre_existing
    assert sink[1] == expected_trace
