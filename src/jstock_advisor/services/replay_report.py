"""replay の表(9 つの節)を、R1a の純粋関数から組み立てる(Issue #880 R1b。F-33)。

入力は `replay_input_loader` が作った R1a の型と、事前登録の本体の値。**集計(件数・割合・行列)は行う
が、判断(OLD / NEW)は再計算しない**(判断は入力のファイルで受け取る)。出力は文字列の表だけ。

## 守ること(契約テストで固定)

* import は標準ライブラリと `jstock_advisor.domain.backtest` と、同じ package の loader だけ
* ★ 閾値・窓・最小セル数・税と手数料は、すべて事前登録の本体の値(`ReplayParameters`)★。
  既定を置かない
* ★ 数値の丸めは `render_number` の 1 か所だけ ★(`quantize` / `round` / `float` はここ以外に無い)。
  桁数は事前登録の値。Fraction は整数だけで ROUND_HALF_EVEN に丸める
* 識別子(保有の参照・銘柄)・個別の価格・数量を出力に含めない(件数・割合・要約だけ)
* 最小セル数に満たないセルは件数だけ(割合・平均・中央値を出さない = R1a の規則のまま)
* UNDETERMINED は別に数える(欠損として落とさない)
* 事前登録の照合が一致しないとき、★ 第 1〜8 節(結果の表)は出さない ★。第 9 節の照合の結果
  (欠落・余り・値の違い)だけを出す。注意書きは CLI の層が付ける(この module は付けない)
"""

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction

from jstock_advisor.domain.backtest.outcome_window import (
    OutcomeStatus,
    WindowOutcome,
    compute_outcomes,
    sale_effect,
)
from jstock_advisor.domain.backtest.preregistration_check import (
    RegistrationCheck,
    ValuesComparison,
    check_report,
    compare_used_values,
    content_hash,
)
from jstock_advisor.domain.backtest.replay_classification import (
    Difference,
    Judgement,
    ReplayAction,
    difference_counts,
    reason_counts,
    transition_matrix,
)
from jstock_advisor.domain.backtest.replay_events import (
    Observation,
    account_population,
    build_events,
    co_fire_counts,
    layer_counts,
    phi_coefficient,
)
from jstock_advisor.domain.backtest.replay_reproducibility import may_proceed, reproducibility
from jstock_advisor.domain.backtest.replay_tables import rate_table, summarize
from jstock_advisor.services.replay_input_loader import (
    Citation,
    PairRecord,
    PriceSeries,
    ReplayParameters,
)

_NONE_MARK = "-"
_SUPPRESSED_NOTE = "最小セル数未満(件数のみ)"


# --- 数値の表示(丸めはここだけ)------------------------------------------------------------------


def render_number(value: Fraction | Decimal | float | None, digits: int) -> str:
    """数値を固定の桁数の文字列にする(ROUND_HALF_EVEN)。None は『-』。丸めはこの関数だけ。"""
    if value is None:
        return _NONE_MARK
    if isinstance(value, Decimal):
        quantum = Decimal(1).scaleb(-digits)
        return format(value.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")
    exact = value if isinstance(value, Fraction) else Fraction(value)
    scale = 10**digits
    scaled, remainder = divmod(abs(exact.numerator) * scale, exact.denominator)
    twice = 2 * remainder
    if twice > exact.denominator or (twice == exact.denominator and scaled % 2 == 1):
        scaled += 1
    sign = "-" if exact < 0 and scaled != 0 else ""
    whole, fraction = divmod(scaled, scale)
    return f"{sign}{whole}" if digits == 0 else f"{sign}{whole}.{fraction:0{digits}d}"


def _count_or_rate(label: str, n: int, rate: Fraction | None, digits: int) -> str:
    if rate is None:
        return f"{label}: {n} 件({_SUPPRESSED_NOTE})"
    return f"{label}: {n} 件({render_number(rate * 100, digits)}%)"


# --- 第 9 節の照合の部分 -------------------------------------------------------------------------


def registration_lines(
    check: RegistrationCheck,
    comparison: ValuesComparison | None,
    registration_id: str,
    actual_hash: str,
) -> list[str]:
    """第 9 節のうち、事前登録の照合の部分(見出しなし)。"""
    lines = [
        f"  登録 ID: {registration_id}",
        f"  本体のハッシュ(SHA-256): {actual_hash}",
        f"  照合の結果: {check.value}",
    ]
    if comparison is not None and not comparison.ok:
        lines.append(f"  欠落(登録にあり、使った値に無い): {', '.join(comparison.missing) or '-'}")
        lines.append(
            f"  余り(使った値にあり、登録に無い): {', '.join(comparison.unexpected) or '-'}"
        )
        lines.append(f"  値の違い: {', '.join(comparison.differing) or '-'}")
    return lines


def mismatch_report(body: dict[str, object], citation: Citation, check: RegistrationCheck) -> str:
    """照合が一致しないときの出力(第 1〜8 節は出さない。第 9 節の照合の結果だけ)。"""
    registration_id = str(body.get("registration_id", _NONE_MARK))
    comparison: ValuesComparison | None
    try:
        comparison = compare_used_values(body, citation.used)
    except ValueError:
        comparison = None
    lines = [
        "事前登録の照合が一致しないため、結果の表(第 1〜8 節)は出しません。",
        "登録に無い値・登録と違う値では計算しません。",
        "",
        "[9] 再現性・事前登録の照合",
        *registration_lines(check, comparison, registration_id, _hash_or_mark(body)),
        "  再現性: 照合が一致するまで出しません",
    ]
    return "\n".join(lines) + "\n"


def _hash_or_mark(body: dict[str, object]) -> str:
    try:
        return content_hash(body)
    except ValueError:
        return _NONE_MARK


# --- 結果の表(第 1〜8 節 + 第 9 節の再現性)------------------------------------------------


def _analyzed_pairs(
    observations: tuple[Observation, ...], pairs: tuple[PairRecord, ...]
) -> list[PairRecord]:
    analyzed = {(o.holding_ref, o.day_index) for o in observations if o.analyzed}
    return [p for p in pairs if (p.holding_ref, p.day_index) in analyzed]


def _judgement_pairs(pairs: list[PairRecord]) -> list[tuple[Judgement, Judgement]]:
    return [(p.old, p.new) for p in pairs]


def _new_label(pair: PairRecord) -> str:
    return "UNDETERMINED" if pair.new.undetermined else pair.new.action.value


def _section_population(observations: tuple[Observation, ...]) -> list[str]:
    account = account_population(observations)
    lines = [
        "[1] 母集団の勘定",
        f"  全部の行: {account.total}",
    ]
    lines += [f"  除外 {reason.value}: {count}" for reason, count in account.excluded]
    lines += [
        f"  除外の合計: {account.excluded_total}",
        f"  症例(held-out。集計に使わない別枠): {account.held_out}",
        f"  分析対象: {account.analyzed}(うち初回評価 {account.first_evaluations})",
    ]
    return lines


def _section_layers(params: ReplayParameters, observations: tuple[Observation, ...]) -> list[str]:
    events = build_events(observations, max_gap=params.max_gap)
    counts = layer_counts(observations, events)
    return [
        "[2] 3 層の件数(分析対象)",
        f"  観測: {counts.observations}",
        f"  event: {counts.events}",
        f"  保有: {counts.holdings}(保有数が少ないときは統計的な有意を主張しない)",
    ]


def _section_classification(params: ReplayParameters, pairs: list[PairRecord]) -> list[str]:
    digits = params.rate_digits
    differences = dict(difference_counts(_judgement_pairs(pairs)))
    counts = {kind.value: differences[kind] for kind in Difference}
    lines = ["[3] OLD と NEW の分類(分析対象の対)", f"  対の数: {len(pairs)}"]
    for cell in rate_table(counts, min_cell=params.min_cell):
        lines.append("  " + _count_or_rate(cell.label, cell.n, cell.rate, digits))
    return lines


def _section_matrix(pairs: list[PairRecord]) -> list[str]:
    matrix = transition_matrix(_judgement_pairs(pairs))
    lines = ["[4] 遷移行列(OLD の action → NEW の action。件数)"]
    for old_action in ReplayAction:
        row = [f"{new.value}={n}" for o, new, n in matrix.cells if o is old_action]
        lines.append(f"  {old_action.value}: " + " / ".join(row))
    lines.append(f"  NEW が判定不能(別に数える): {matrix.undetermined}")
    return lines


def _section_reasons(params: ReplayParameters, pairs: list[PairRecord]) -> list[str]:
    codes = [p.reason_code for p in pairs if p.reason_code is not None]
    result = reason_counts(codes, params.reason_vocabulary)
    lines = ["[5] 理由コードの件数(NEW の理由)"]
    lines += [f"  {code}: {n}" for code, n in result.counts]
    lines.append(f"  認識できないコード: {result.unrecognized}")
    return lines


def _outcome_groups(
    params: ReplayParameters, pairs: list[PairRecord], prices: dict[str, PriceSeries]
) -> tuple[dict[tuple[str, int], list[tuple[PairRecord, WindowOutcome]]], int]:
    """NEW の action ごと・窓ごとの転帰。価格の列が無い・範囲外の対の数も返す(落とさず数える)。"""
    groups: dict[tuple[str, int], list[tuple[PairRecord, WindowOutcome]]] = defaultdict(list)
    unavailable = 0
    for pair in pairs:
        series = prices.get(pair.holding_ref)
        index = None if series is None else pair.day_index - series.first_day_index
        if series is None or index is None or not 0 <= index < len(series.bars):
            unavailable += 1
            continue
        outcomes = compute_outcomes(
            series.bars, index, params.outcome_spec, prior_high=pair.prior_high
        )
        for outcome in outcomes:
            groups[(_new_label(pair), outcome.window)].append((pair, outcome))
    return groups, unavailable


def _summary_line(label: str, values: list[Decimal], params: ReplayParameters) -> str:
    cell = summarize(values, min_cell=params.min_cell)
    digits = params.rate_digits
    if cell.suppressed:
        return f"    {label}: n={cell.n}({_SUPPRESSED_NOTE})"
    return (
        f"    {label}: n={cell.n} 平均={render_number(cell.mean, digits)} "
        f"中央値={render_number(cell.median, digits)}"
    )


def _section_outcomes(
    params: ReplayParameters,
    groups: dict[tuple[str, int], list[tuple[PairRecord, WindowOutcome]]],
    unavailable: int,
) -> list[str]:
    lines = [
        "[6] 転帰(判断の後の値動き。NEW の action 別・窓別)",
        f"  価格の列が無い・範囲外の対(値を作らない): {unavailable}",
    ]
    for (label, window), items in sorted(groups.items()):
        complete = [o for _, o in items if o.status is OutcomeStatus.COMPLETE]
        insufficient = len(items) - len(complete)
        lines.append(
            f"  NEW={label} 窓={window} 営業日: 完全 {len(complete)}"
            f" / 系列が足りない {insufficient}"
        )
        lines.append(
            _summary_line(
                "forward_return",
                [o.forward_return for o in complete if o.forward_return is not None],
                params,
            )
        )
        lines.append(
            _summary_line(
                "max_drawdown",
                [o.max_drawdown for o in complete if o.max_drawdown is not None],
                params,
            )
        )
        lines.append(
            _summary_line("rebound", [o.rebound for o in complete if o.rebound is not None], params)
        )
        high = sum(1 for o in complete if o.new_high)
        lines.append(f"    新高値の更新: {high} 件 / {len(complete)} 件")
    return lines


def _section_sale_effect(
    params: ReplayParameters,
    pairs: list[PairRecord],
    groups: dict[tuple[str, int], list[tuple[PairRecord, WindowOutcome]]],
) -> list[str]:
    lines = [
        "[7] 売却の効果(HOLD 100% との差。起点の評価額を 1 とした相対値。税・手数料は事前登録の値)"
    ]
    for basis in ("OLD", "NEW"):
        per_window: dict[tuple[str, int], dict[str, list[Decimal]]] = defaultdict(
            lambda: {"早売り": [], "回避した下落": [], "売り遅れ": []}
        )
        for (_, window), items in groups.items():
            for pair, outcome in items:
                judgement = pair.old if basis == "OLD" else pair.new
                ratio = pair.old_sold_ratio if basis == "OLD" else pair.new_sold_ratio
                if judgement.action is ReplayAction.HOLD:
                    continue  # 判定不能も action は HOLD(Judgement の不変条件)
                effect = sale_effect(
                    outcome, sold_ratio=ratio, sale_cost_rate=params.sale_cost_rate
                )
                if effect is None:
                    continue
                bucket = per_window[(judgement.action.value, window)]
                bucket["早売り"].append(effect.missed_upside)
                bucket["回避した下落"].append(effect.avoided_loss)
                bucket["売り遅れ"].append(effect.further_loss_on_remaining)
        for (action, window), values in sorted(per_window.items()):
            lines.append(f"  {basis}={action} 窓={window} 営業日:")
            for name, series in values.items():
                lines.append(_summary_line(name, series, params))
    return lines


def _section_co_fire(params: ReplayParameters, pairs: list[PairRecord]) -> list[str]:
    counts = co_fire_counts([p.fires for p in pairs])
    lines = [
        "[8] 同時発火率(2 つの層が同じ観測日に悪化した件数)",
        f"  両方: {counts.both} / 第 1 のみ: {counts.only_first} / 第 2 のみ: {counts.only_second}"
        f" / どちらも無し: {counts.neither}",
        f"  どちらかが判定不能(別に数える): {counts.undetermined}",
    ]
    if counts.determined < params.min_cell:
        lines.append(f"  phi: {_NONE_MARK}({_SUPPRESSED_NOTE})")
    else:
        lines.append(f"  phi: {render_number(phi_coefficient(counts), params.rate_digits)}")
    return lines


def _section_reproducibility(params: ReplayParameters, pairs: list[PairRecord]) -> list[str]:
    result = reproducibility([(p.recorded, p.recomputed) for p in pairs])
    digits = params.rate_digits
    lines = [
        "[9] 再現性・事前登録の照合",
        "  再現性(記録された判断と、再計算した判断の一致)",
        f"  対の数: {result.total} / 一致: {result.matched} / 不一致: {result.mismatched}",
    ]
    if result.total >= params.min_cell:
        lines.append(f"  一致率: {render_number(result.rate, digits)}")
    else:
        lines.append(f"  一致率: {_NONE_MARK}({_SUPPRESSED_NOTE})")
    lines += [f"  不一致の内訳: {a} → {b}: {n}" for a, b, n in result.mismatches]
    verdict = "進める" if may_proceed(result, required=params.required_agreement) else "進めない"
    lines.append(f"  比較に進めるか(登録された一致率に対して): {verdict}")
    return lines


def build_report(
    *,
    body: dict[str, object],
    citation: Citation,
    params: ReplayParameters,
    observations: tuple[Observation, ...],
    pairs: tuple[PairRecord, ...],
    prices: dict[str, PriceSeries],
) -> str:
    """表の全体(第 1〜9 節)。事前登録の照合が一致しないときは第 9 節の照合の結果だけ。"""
    check = check_report(
        body,
        cited_registration_id=citation.registration_id,
        cited_hash=citation.content_hash,
        used=citation.used,
    )
    if check is not RegistrationCheck.MATCH:
        return mismatch_report(body, citation, check)
    analyzed = _analyzed_pairs(observations, pairs)
    groups, unavailable = _outcome_groups(params, analyzed, prices)
    sections = [
        _section_population(observations),
        _section_layers(params, observations),
        _section_classification(params, analyzed),
        _section_matrix(analyzed),
        _section_reasons(params, analyzed),
        _section_outcomes(params, groups, unavailable),
        _section_sale_effect(params, analyzed, groups),
        _section_co_fire(params, analyzed),
        [
            *_section_reproducibility(params, analyzed),
            "  事前登録の照合",
            *registration_lines(check, None, str(body["registration_id"]), content_hash(body)),
        ],
    ]
    return "\n\n".join("\n".join(lines) for lines in sections) + "\n"


def registration_check(body: dict[str, object], citation: Citation) -> RegistrationCheck:
    """引用(登録 ID・ハッシュ・使った値)と登録の本体の照合。判定の順序は R1a の `check_report`。"""
    return check_report(
        body,
        cited_registration_id=citation.registration_id,
        cited_hash=citation.content_hash,
        used=citation.used,
    )


def check_only_report(body: dict[str, object], citation: Citation) -> tuple[str, RegistrationCheck]:
    """`check-prereg` の出力(第 9 節の照合の結果だけ)。"""
    check = check_report(
        body,
        cited_registration_id=citation.registration_id,
        cited_hash=citation.content_hash,
        used=citation.used,
    )
    try:
        comparison: ValuesComparison | None = compare_used_values(body, citation.used)
    except ValueError:
        comparison = None
    lines = [
        "[9] 事前登録の照合",
        *registration_lines(
            check, comparison, str(body.get("registration_id", _NONE_MARK)), _hash_or_mark(body)
        ),
    ]
    return "\n".join(lines) + "\n", check
