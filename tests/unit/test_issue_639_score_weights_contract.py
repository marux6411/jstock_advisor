"""Issue #639: ScoreWeights(配点)と BenefitUtilityCoefficients(利用可能性係数。config 版)の契約。

背景: #538 の調査で、この 2 つのモデルには検査が一切無く、NaN / ±inf、負値、合計のずれが起動時に
検出されなかった。契約(設計 = #639 の Phase B 設計と、訂正・HANAKO 決定の反映):

  ScoreWeights                  各配点は有限で 0 より大きい / 合計は 100(誤差 1e-6)
  BenefitUtilityCoefficients    各係数は有限で 0 以上 1.0 以下(閉区間)/ 合計の契約は無い

固定するもの:
  ・上の契約(NaN・±inf・負値・0・合計のずれ・上限超過・境界)
  ・現行 config(scoring_weights.yaml)が契約を満たし、load_config で読めること(起動時に落ちない)
  ・YAML 経由でも拒否されること(config の差し替えで契約を破ると起動時に落ちる = fail-fast)
  ・過去に永続化された配点の snapshot からの復元は、不正なら現在の配点へ fallback する既存の経路
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jstock_advisor.config.loader import DEFAULT_CONFIG_DIR, load_config
from jstock_advisor.config.models import BenefitUtilityCoefficients, ScoreWeights

_WEIGHT_FIELDS = list(ScoreWeights.model_fields)
_COEFFICIENT_FIELDS = list(BenefitUtilityCoefficients.model_fields)

_VALID_WEIGHTS: dict[str, float] = {
    "total_yield_attractiveness": 20,
    "dividend_sustainability": 20,
    "financial_health": 20,
    "undervaluation": 20,
    "shareholder_benefit_value": 10,
    "earnings_stability": 5,
    "price_stability": 5,
}
_VALID_COEFFICIENTS: dict[str, float] = {
    "cash_equivalent": 1.0,
    "versatile_point": 0.9,
    "in_house_service": 0.7,
    "in_house_product": 0.6,
    "discount_voucher": 0.3,
    "lottery_or_commemorative": 0.0,
}


def _weights(**override: float) -> ScoreWeights:
    return ScoreWeights(**{**_VALID_WEIGHTS, **override})


def _coefficients(**override: float) -> BenefitUtilityCoefficients:
    return BenefitUtilityCoefficients(**{**_VALID_COEFFICIENTS, **override})


# =================================================================================================
# ScoreWeights
# =================================================================================================


def test_the_current_weights_are_accepted() -> None:
    weights = _weights()

    assert sum(weights.model_dump().values()) == pytest.approx(100.0)


def test_the_fields_are_exactly_the_seven_weights() -> None:
    assert list(_VALID_WEIGHTS) == _WEIGHT_FIELDS


@pytest.mark.parametrize("field", _WEIGHT_FIELDS)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_weight_is_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=field):
        _weights(**{field: bad})


@pytest.mark.parametrize("field", _WEIGHT_FIELDS)
@pytest.mark.parametrize("bad", [-0.01, -20.0, 0.0, -0.0])
def test_a_zero_or_negative_weight_is_rejected(field: str, bad: float) -> None:
    # 他の項目で合計 100 に調整しても、当該の配点が 0 以下なら拒否される(合計の検査より前に分かる)
    others = [name for name in _WEIGHT_FIELDS if name != field]
    adjusted = {name: _VALID_WEIGHTS[name] for name in others}
    adjusted[others[0]] += _VALID_WEIGHTS[field] - bad
    with pytest.raises(ValidationError, match=field):
        ScoreWeights(**{**adjusted, field: bad})


@pytest.mark.parametrize("delta", [-0.01, 0.01, -1.0, 1.0, -19.0, 100.0])
def test_a_total_other_than_100_is_rejected(delta: float) -> None:
    with pytest.raises(ValidationError, match="合計は 100"):
        _weights(total_yield_attractiveness=20 + delta)


def test_a_total_within_the_float_tolerance_is_accepted() -> None:
    _weights(total_yield_attractiveness=20 + 1e-9)
    _weights(total_yield_attractiveness=20 - 1e-9)


def test_other_distributions_that_total_100_are_accepted() -> None:
    _weights(
        total_yield_attractiveness=40,
        dividend_sustainability=10,
        financial_health=10,
        undervaluation=10,
        shareholder_benefit_value=10,
        earnings_stability=10,
        price_stability=10,
    )
    _weights(
        total_yield_attractiveness=0.5,
        dividend_sustainability=0.5,
        financial_health=0.5,
        undervaluation=0.5,
        shareholder_benefit_value=0.5,
        earnings_stability=0.5,
        price_stability=97,
    )


def test_the_message_does_not_include_a_value_that_should_not_be_echoed() -> None:
    # 値そのものではなく、どの項目が契約を破ったかを伝える
    with pytest.raises(ValidationError) as caught:
        _weights(financial_health=math.nan)
    assert "scoring.weights.financial_health" in str(caught.value)


# =================================================================================================
# BenefitUtilityCoefficients(config 版)
# =================================================================================================


def test_the_current_coefficients_are_accepted() -> None:
    coefficients = _coefficients()

    assert coefficients.lottery_or_commemorative == 0.0  # 0 は許容(現行値)
    assert coefficients.cash_equivalent == 1.0  # 上限ちょうどは許容


def test_the_fields_are_exactly_the_six_coefficients() -> None:
    assert list(_VALID_COEFFICIENTS) == _COEFFICIENT_FIELDS


@pytest.mark.parametrize("field", _COEFFICIENT_FIELDS)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_coefficient_is_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=field):
        _coefficients(**{field: bad})


@pytest.mark.parametrize("field", _COEFFICIENT_FIELDS)
@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 1.0000001, 2.0])
def test_a_coefficient_outside_zero_to_one_is_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=field):
        _coefficients(**{field: bad})


@pytest.mark.parametrize("field", _COEFFICIENT_FIELDS)
@pytest.mark.parametrize("edge", [0.0, 1.0, 0.5])
def test_the_boundaries_and_the_middle_are_accepted(field: str, edge: float) -> None:
    _coefficients(**{field: edge})


def test_the_coefficients_have_no_total_contract() -> None:
    # カテゴリごとに独立した割引率。合計は 6 でも 0 でもよい
    _coefficients(**dict.fromkeys(_COEFFICIENT_FIELDS, 1.0))
    _coefficients(**dict.fromkeys(_COEFFICIENT_FIELDS, 0.0))


# =================================================================================================
# 現行 config と YAML 経由
# =================================================================================================


def test_the_shipped_config_satisfies_the_contract_and_loads() -> None:
    config = load_config()

    weights = config.scoring.weights.model_dump()
    assert all(v > 0 and math.isfinite(v) for v in weights.values())
    assert sum(weights.values()) == pytest.approx(100.0, abs=1e-6)
    coefficients = config.scoring.shareholder_benefit_value.utility_coefficients_default
    assert all(0.0 <= v <= 1.0 for v in coefficients.model_dump().values())


def _config_with(tmp_path: Path, old: str, new: str) -> Path:
    broken = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, broken)
    path = broken / "scoring_weights.yaml"
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")
    return broken


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  financial_health: 20 ", "  financial_health: .nan "),  # NaN
        ("  financial_health: 20 ", "  financial_health: .inf "),  # +inf
        ("  financial_health: 20 ", "  financial_health: 0 "),  # 0
        ("  financial_health: 20 ", "  financial_health: -5 "),  # 負値
        ("  financial_health: 20 ", "  financial_health: 21 "),  # 合計 101
        ("  price_stability: 5 ", "  price_stability: 4 "),  # 合計 99
        ("    cash_equivalent: 1.0 ", "    cash_equivalent: 1.5 "),  # 係数が上限超過
        ("    discount_voucher: 0.3 ", "    discount_voucher: -0.1 "),  # 係数が負
        ("    discount_voucher: 0.3 ", "    discount_voucher: .nan "),  # 係数が NaN
    ],
)
def test_a_config_that_breaks_the_contract_fails_at_load(
    tmp_path: Path, old: str, new: str
) -> None:
    broken = _config_with(tmp_path, old, new)

    with pytest.raises(ValidationError):
        load_config(broken)


def test_an_unmodified_copy_of_the_config_still_loads(tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, copied)

    load_config(copied)


# =================================================================================================
# 過去の snapshot からの復元(既存の経路は変わらない)
# =================================================================================================


def test_a_stored_snapshot_that_breaks_the_contract_falls_back_to_the_current_weights() -> None:
    from jstock_advisor.services.watchlist_judgment_summary_formatter import _resolve_weights

    class _Recommendation:
        def __init__(self, snapshot: Any) -> None:
            self.config_values_used = {"scoring_weights": snapshot}

    current = _weights()
    valid_snapshot = {**_VALID_WEIGHTS, "financial_health": 10, "undervaluation": 30}
    broken_snapshot = {**_VALID_WEIGHTS, "financial_health": 10}  # 合計 90

    used = _resolve_weights(_Recommendation(valid_snapshot), current)  # type: ignore[arg-type]
    fell_back = _resolve_weights(_Recommendation(broken_snapshot), current)  # type: ignore[arg-type]

    assert used.financial_health == 10 and used.undervaluation == 30
    assert fell_back is current
