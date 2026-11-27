"""`services/replay_report.py`(Issue #880 R1b)の契約テスト。

固定するもの: 9 つの節の中身(件数・UNDETERMINED の別計上・最小セル数未満は件数のみ)・
識別子/価格/数量を出力に含めない・数値の丸めは `render_number` の 1 か所・★ 事前登録の照合が
一致しないときは第 1〜8 節を出さず、第 9 節の照合の結果(欠落・余り・値の違い)だけを出す ★
(「第 1〜8 節が無い」と「不一致の内容が在る」を別々に固定する)・注意書きはこの module では
付けない(CLI の層の責務)。fixture は合成のみ。
"""

from __future__ import annotations

import ast
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from jstock_advisor.domain.backtest.preregistration_check import RegistrationCheck
from jstock_advisor.services import replay_input_loader as loader
from jstock_advisor.services import replay_report as report
from tests.support import replay_fixtures as fx

_REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "jstock_advisor" / "services" / "replay_report.py"
)
_SECTION = re.compile(r"^\[(\d)\] ", re.MULTILINE)


def _build(
    *,
    prereg_overrides: dict[str, Any] | None = None,
    cite_overrides: dict[str, Any] | None = None,
    mutate_pairs: Any = None,
) -> str:
    body_dict = fx.prereg_body(**(prereg_overrides or {}))
    body, params = loader.load_prereg(fx.to_json(body_dict))
    citation = loader.load_citation(
        fx.to_json(fx.citation_for(body_dict, **(cite_overrides or {})))
    )
    observations = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=params.held_out_refs, max_lines=100
    )
    pair_rows = fx.pair_rows()
    if mutate_pairs is not None:
        mutate_pairs(pair_rows)
    pairs = loader.load_pairs(fx.to_lines(pair_rows), max_lines=100)
    prices = loader.load_prices(fx.to_lines(fx.price_rows()), max_lines=100)
    return report.build_report(
        body=body,
        citation=citation,
        params=params,
        observations=observations,
        pairs=pairs,
        prices=prices,
    )


def _sections(text: str) -> list[int]:
    return [int(n) for n in _SECTION.findall(text)]


def _section(text: str, number: int) -> str:
    parts = re.split(r"(?m)^(?=\[\d\] )", text)
    found = [p for p in parts if p.startswith(f"[{number}] ")]
    assert len(found) == 1, number
    return found[0]


# =================================================================================================
# 9 つの節
# =================================================================================================


def test_all_nine_sections_appear_once_in_order_when_the_registration_matches() -> None:
    assert _sections(_build()) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_section_1_accounts_for_every_row_and_shows_zero_reasons() -> None:
    text = _section(_build(), 1)
    assert "全部の行: 8" in text
    for reason, count in {
        "LEGACY": 1,
        "EVALUATED_AT_MISMATCH": 0,  # 件数 0 の区分も省略しない
        "BEFORE_PLAN": 0,
        "BASELINE_INTEGRITY_ERROR": 0,
        "DATA_INSUFFICIENT": 1,
    }.items():
        assert f"除外 {reason}: {count}" in text
    assert "除外の合計: 2" in text
    assert "症例(held-out。集計に使わない別枠): 1" in text
    assert "分析対象: 5(うち初回評価 1)" in text  # 8 = 2 + 1 + 5


def test_section_2_reports_the_three_layers_and_warns_about_few_holdings() -> None:
    text = _section(_build(), 2)
    assert "観測: 5" in text and "event: 3" in text and "保有: 2" in text
    assert "統計的な有意を主張しない" in text


def test_section_3_counts_each_kind_and_never_shows_a_rate_below_the_minimum_cell() -> None:
    text = _section(_build(), 3)
    for kind in ("MATCH", "NEW_ONLY", "OLD_ONLY", "STRENGTH_DIFF", "CLASS_DIFF", "UNDETERMINED"):
        assert f"{kind}: " in text  # 0 件も省略しない
    assert "対の数: 5" in text
    assert "%" not in text  # 各セル 1 件 < 最小セル数 2 → 割合を出さない
    assert text.count("最小セル数未満(件数のみ)") == 6


def test_section_3_shows_a_rate_once_the_cell_reaches_the_minimum() -> None:
    text = _section(_build(prereg_overrides={"min_cell": 1}), 3)
    assert "MATCH: 1 件(20.00%)" in text  # 1 / 5


def test_section_4_keeps_undetermined_out_of_the_hold_column() -> None:
    text = _section(_build(), 4)
    assert "HOLD: HOLD=1 / PARTIAL=0 / FULL=1" in text
    assert "FULL: HOLD=0 / PARTIAL=1 / FULL=1" in text
    assert "PARTIAL: HOLD=0 / PARTIAL=0 / FULL=0" in text  # NEW が判定不能の対は HOLD に入れない
    assert "NEW が判定不能(別に数える): 1" in text


def test_section_5_counts_unrecognized_codes_instead_of_dropping_them() -> None:
    text = _section(_build(), 5)
    assert "R_A: 1" in text and "R_B: 1" in text and "認識できないコード: 1" in text


def test_section_6_groups_by_the_new_action_and_suppresses_small_cells() -> None:
    text = _section(_build(), 6)
    assert "NEW=FULL 窓=1 営業日: 完全 2 / 系列が足りない 0" in text
    assert "forward_return: n=2 平均=0.01 中央値=0.01" in text
    assert "NEW=PARTIAL 窓=2 営業日: 完全 1 / 系列が足りない 0" in text
    assert "NEW=UNDETERMINED" in text  # 判定不能も別に出す
    assert "n=1(最小セル数未満(件数のみ))" in text
    assert "価格の列が無い・範囲外の対(値を作らない): 0" in text


def test_section_6_counts_pairs_without_a_price_series_instead_of_inventing_values() -> None:
    body_dict = fx.prereg_body()
    body, params = loader.load_prereg(fx.to_json(body_dict))
    citation = loader.load_citation(fx.to_json(fx.citation_for(body_dict)))
    observations = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=params.held_out_refs, max_lines=100
    )
    pairs = loader.load_pairs(fx.to_lines(fx.pair_rows()), max_lines=100)
    prices = {
        k: v
        for k, v in loader.load_prices(fx.to_lines(fx.price_rows()), max_lines=100).items()
        if k != "h2"
    }
    text = report.build_report(
        body=body,
        citation=citation,
        params=params,
        observations=observations,
        pairs=pairs,
        prices=prices,
    )
    assert "価格の列が無い・範囲外の対(値を作らない): 2" in _section(text, 6)  # h2 の 2 対


def test_section_6_reports_a_short_series_as_insufficient() -> None:
    def shorten(rows: list[dict[str, Any]]) -> None:
        pass

    body_dict = fx.prereg_body(outcome_windows=[1, 2, 20])
    body, params = loader.load_prereg(fx.to_json(body_dict))
    citation = loader.load_citation(fx.to_json(fx.citation_for(body_dict)))
    observations = loader.load_observations(
        fx.to_lines(fx.observation_rows()), held_out_refs=params.held_out_refs, max_lines=100
    )
    pairs = loader.load_pairs(fx.to_lines(fx.pair_rows()), max_lines=100)
    prices = loader.load_prices(fx.to_lines(fx.price_rows()), max_lines=100)
    text = report.build_report(
        body=body,
        citation=citation,
        params=params,
        observations=observations,
        pairs=pairs,
        prices=prices,
    )
    assert re.search(r"窓=20 営業日: 完全 0 / 系列が足りない \d+", _section(text, 6))


def test_section_7_compares_old_and_new_sales_against_holding_everything() -> None:
    text = _section(_build(), 7)
    assert "OLD=FULL 窓=1 営業日:" in text and "NEW=FULL 窓=1 営業日:" in text
    assert "OLD=PARTIAL" in text and "NEW=PARTIAL" in text
    assert "早売り: n=2" in text and "回避した下落: n=2" in text and "売り遅れ: n=2" in text
    assert "OLD=HOLD" not in text and "NEW=HOLD" not in text  # HOLD は売らない


def test_section_8_counts_undetermined_apart_and_hides_phi_for_a_small_total() -> None:
    text = _section(_build(), 8)
    assert "両方: 1 / 第 1 のみ: 1 / 第 2 のみ: 1 / どちらも無し: 1" in text
    assert "どちらかが判定不能(別に数える): 1" in text
    assert "phi: 0.00" in text  # 判定できた 4 件 ≥ 最小セル数 2
    small = _section(_build(prereg_overrides={"min_cell": 5}), 8)
    assert "phi: -(最小セル数未満(件数のみ))" in small


def test_section_8_shows_phi_when_the_determined_total_equals_the_minimum() -> None:
    """境界: 判定できた 4 件 = 最小セル数 4 のときは出す(未満のときだけ隠す)。"""
    assert "phi: 0.00" in _section(_build(prereg_overrides={"min_cell": 4}), 8)


def test_section_9_shows_the_rate_when_the_pair_total_equals_the_minimum() -> None:
    """境界: 対の数 5 = 最小セル数 5 のときは一致率を出し、6 のときは隠す。"""
    assert "一致率: 0.80" in _section(_build(prereg_overrides={"min_cell": 5}), 9)
    assert "一致率: -(最小セル数未満(件数のみ))" in _section(
        _build(prereg_overrides={"min_cell": 6}), 9
    )


def test_section_7_leaves_out_the_new_undetermined_pair() -> None:
    """NEW が判定不能の対(action は HOLD)は売却の効果へ入れない。売りとして数えない。"""
    text = _section(_build(), 7)
    assert "UNDETERMINED" not in text
    assert "NEW=PARTIAL 窓=1 営業日:" in text  # PARTIAL の 1 対だけ(判定不能の 1 対は入らない)
    assert (
        text.count("n=1(最小セル数未満(件数のみ))") == 12
    )  # OLD PARTIAL・NEW PARTIAL × 窓 2 × 指標 3


def test_section_9_reports_reproducibility_and_the_registration_check() -> None:
    text = _section(_build(), 9)
    assert "対の数: 5 / 一致: 4 / 不一致: 1" in text
    assert "一致率: 0.80" in text
    assert "不一致の内訳: PARTIAL → HOLD: 1" in text
    assert "比較に進めるか(登録された一致率に対して): 進めない" in text
    assert "登録 ID: REG-TEST-1" in text and "照合の結果: MATCH" in text
    assert re.search(r"本体のハッシュ\(SHA-256\): [0-9a-f]{64}", text)


def test_section_9_lets_the_comparison_proceed_when_the_registered_agreement_is_met() -> None:
    text = _section(_build(prereg_overrides={"required_agreement": "4/5"}), 9)
    assert "比較に進めるか(登録された一致率に対して): 進める" in text


def test_section_9_never_proceeds_with_zero_pairs() -> None:
    def drop_all(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["day_index"] = 10_000  # 観測に無い → 読込の検査で拒否される

    # 分析対象の対が 0 件になる入力: 全ての観測を除外にする
    body_dict = fx.prereg_body()
    body, params = loader.load_prereg(fx.to_json(body_dict))
    citation = loader.load_citation(fx.to_json(fx.citation_for(body_dict)))
    rows = fx.observation_rows()
    for row in rows:
        row["exclusion"] = "LEGACY"
    observations = loader.load_observations(
        fx.to_lines(rows), held_out_refs=params.held_out_refs, max_lines=100
    )
    pairs = loader.load_pairs(fx.to_lines(fx.pair_rows()), max_lines=100)
    prices = loader.load_prices(fx.to_lines(fx.price_rows()), max_lines=100)
    text = report.build_report(
        body=body,
        citation=citation,
        params=params,
        observations=observations,
        pairs=pairs,
        prices=prices,
    )
    section = _section(text, 9)
    assert "対の数: 0 / 一致: 0 / 不一致: 0" in section
    assert "一致率: -(最小セル数未満(件数のみ))" in section
    assert "進めない" in section


# =================================================================================================
# 出力に含めないもの・決定性
# =================================================================================================


def test_the_output_has_no_identifier_price_or_quantity() -> None:
    text = _build()
    for forbidden in ("h1", "h2", "h3", "h4", "h-case", "100.5"):
        assert forbidden not in text, forbidden
    without_hash = re.sub(r"[0-9a-f]{64}", "", text).replace(
        "HOLD 100%", ""
    )  # ハッシュの 16 進数は価格ではない
    prices = {str(100 + i) for i in range(14)} | {str(101 + i) for i in range(14)}
    tokens = set(re.findall(r"(?<![\d.])\d{3}(?![\d.])", without_hash))
    assert tokens & prices == set()


def test_the_output_is_deterministic() -> None:
    assert _build() == _build()


def test_the_output_has_no_notice_sentences_the_cli_layer_adds_them() -> None:
    text = _build()
    assert "公開面へ貼らない" not in text and "実際の投資判断ではありません" not in text


# =================================================================================================
# 事前登録の照合が一致しないとき: 第 1〜8 節を出さず、第 9 節の照合の結果だけを出す(別々に固定)
# =================================================================================================


def _mismatch_text(**cite_overrides: Any) -> str:
    return _build(cite_overrides=cite_overrides)


@pytest.mark.parametrize(
    ("cite_overrides", "expected"),
    [
        ({"content_hash": "0" * 64}, RegistrationCheck.HASH_MISMATCH),
        ({"registration_id": "REG-OTHER"}, RegistrationCheck.ID_MISMATCH),
    ],
)
def test_a_mismatch_prints_no_result_section_1_to_8(
    cite_overrides: dict[str, Any], expected: RegistrationCheck
) -> None:
    text = _mismatch_text(**cite_overrides)
    assert _sections(text) == [
        9
    ]  # 第 1〜8 節が無い(これだけでは「内容が在る」は言えない → 下のテスト)
    for number in range(1, 9):
        assert f"[{number}] " not in text
    assert "結果の表(第 1〜8 節)は出しません" in text
    assert expected.value  # 期待する語彙が存在する


@pytest.mark.parametrize(
    ("cite_overrides", "expected"),
    [
        ({"content_hash": "0" * 64}, "HASH_MISMATCH"),
        ({"registration_id": "REG-OTHER"}, "ID_MISMATCH"),
    ],
)
def test_a_mismatch_still_names_what_did_not_match(
    cite_overrides: dict[str, Any], expected: str
) -> None:
    text = _mismatch_text(**cite_overrides)
    assert f"照合の結果: {expected}" in text  # 不一致の内容が在る(「節が無い」とは別に固定)
    assert "登録 ID: REG-TEST-1" in text
    assert re.search(r"本体のハッシュ\(SHA-256\): [0-9a-f]{64}", text)


def test_a_values_difference_reports_missing_unexpected_and_differing_separately() -> None:
    body_dict = fx.prereg_body()
    used = {p["name"]: p["value"] for p in body_dict["parameters"]}
    del used["max_gap"]  # 欠落
    used["extra_value"] = 1  # 余り
    used["min_cell"] = 99  # 値の違い
    text = _mismatch_text(used=used)
    assert "照合の結果: VALUES_DIFFER" in text
    assert "欠落(登録にあり、使った値に無い): max_gap" in text
    assert "余り(使った値にあり、登録に無い): extra_value" in text
    assert "値の違い: min_cell" in text
    assert _sections(text) == [9]


def test_a_mismatch_computes_nothing_from_the_data() -> None:
    """不一致のとき、データ由来の数(母集団・分類など)が 1 つも出ない。"""
    text = _mismatch_text(content_hash="0" * 64)
    for needle in ("全部の行", "対の数", "遷移行列", "転帰", "同時発火率", "認識できないコード"):
        assert needle not in text
    assert "再現性: 照合が一致するまで出しません" in text


def test_check_only_report_has_only_section_9() -> None:
    body_dict = fx.prereg_body()
    body, _ = loader.load_prereg(fx.to_json(body_dict))
    citation = loader.load_citation(fx.to_json(fx.citation_for(body_dict)))
    text, check = report.check_only_report(body, citation)
    assert check is RegistrationCheck.MATCH
    assert _sections(text) == [9] and "照合の結果: MATCH" in text
    bad = loader.load_citation(fx.to_json(fx.citation_for(body_dict, content_hash="0" * 64)))
    text, check = report.check_only_report(body, bad)
    assert check is RegistrationCheck.HASH_MISMATCH and "HASH_MISMATCH" in text


# =================================================================================================
# 数値の丸め(1 か所)
# =================================================================================================


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (Fraction(1, 8), 2, "0.12"),  # ROUND_HALF_EVEN: 0.125 → 0.12
        (Fraction(3, 8), 2, "0.38"),  # 0.375 → 0.38
        (Fraction(5, 8), 2, "0.62"),  # 0.625 → 0.62
        (Fraction(7, 8), 2, "0.88"),  # 0.875 → 0.88
        (Fraction(1, 2), 0, "0"),  # 0.5 → 0(偶数へ)
        (Fraction(3, 2), 0, "2"),
        (Fraction(-1, 8), 2, "-0.12"),
        (Fraction(-1, 1000), 2, "0.00"),  # 負のゼロを作らない
        (Fraction(0), 3, "0.000"),
        (Fraction(1), 2, "1.00"),
        (Fraction(1, 3), 4, "0.3333"),
        (Fraction(2, 3), 4, "0.6667"),
        (Fraction(12345, 100), 1, "123.4"),  # 123.45 は偶数側へ
        (Decimal("0.125"), 2, "0.12"),
        (Decimal("0.135"), 2, "0.14"),
        (Decimal("-0.005"), 2, "-0.00"),
        (Decimal("2"), 3, "2.000"),
        (0.5, 1, "0.5"),
        (-0.25, 1, "-0.2"),
        (None, 2, "-"),
    ],
)
def test_render_number_rounds_half_even_at_the_registered_digits(
    value: Any, digits: int, expected: str
) -> None:
    assert report.render_number(value, digits) == expected


def test_rounding_happens_only_inside_render_number() -> None:
    tree = ast.parse(_REPORT_PATH.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "render_number" in functions

    def offending(node: ast.AST) -> list[str]:
        found: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id in {"float", "round"}:
                    found.append(func.id)
                if isinstance(func, ast.Attribute) and func.attr in {"quantize", "__round__"}:
                    found.append(func.attr)
            if isinstance(child, ast.Constant) and isinstance(child.value, float):
                found.append("float literal")
        return found

    inside = offending(functions["render_number"])
    assert "quantize" in inside  # 空振りでない: 丸めはここにある
    others = [
        (name, offending(node)) for name, node in functions.items() if name != "render_number"
    ]
    assert [(n, o) for n, o in others if o] == []


def test_the_module_defines_no_numeric_default_and_adds_no_notice() -> None:
    tree = ast.parse(_REPORT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for default in [*node.args.defaults, *[d for d in node.args.kw_defaults if d]]:
                assert not (
                    isinstance(default, ast.Constant)
                    and isinstance(default.value, int | float)
                    and not isinstance(default.value, bool)
                ), node.name
    source = _REPORT_PATH.read_text(encoding="utf-8")
    assert "NOTICES" not in source and "公開面" not in source  # 注意書きは CLI の層
