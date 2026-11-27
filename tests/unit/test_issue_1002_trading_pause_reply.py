"""Issue #1002: LINE の売買登録失敗時の返信で、設定の取得失敗と利用者が意図した停止を区別する。

## 何を固定するのか

```
TradingPauseService に PauseState(NOT_PAUSED / PAUSED / PAUSED_FETCH_FAILED)を返す入口
`resolve_buy_sell_pause_state()` を足し、ConversationService の 4 箇所(会話の開始・入力の途中・
買付の確定直前・売却の確定直前)が状態に応じた返信を返す。
  T1  通常の一時停止の返信は **文字列リテラルの全文一致**で不変(4 箇所それぞれ)
  T2  取得失敗の返信は USER が確定した文言の全文一致(4 箇所それぞれ)
  T3  取得失敗でも fail-safe(確定の処理に進まない)
  T4  通知は 1 応答(1 instance)で最大 1 回・publish の失敗は握る
  T5  NOT_PAUSED のとき返信は None(通常の応答に進む)
  T6  is_buy_sell_paused() の戻り値は従来と同じ(取得失敗 = True)
  T7  state 入口は取得を 1 回しか呼ばない
  T8  新文と通知の文言は同じ語を共有し、第 3 文は通常停止の文と同一
  T9  ★ WATCH のとき設定の取得 0 回・通知 0 回(短絡評価。サブちゃんの MUST-1)
  T10 ★ `_FALLBACK_PAUSE_BUY_SELL` を参照し続ける(knob が効く。サブちゃんの SHOULD-1 案 A)
```

## 検査していない範囲

```
・実際の SNS publish・LINE 送信・DynamoDB(スタブ・差し替え)
・Production(取得失敗が自然発生したときだけ新しい返信が出る負の経路。人為的な障害注入はしない)
```

fixture は架空値のみ。Production・AWS へは触れない。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from jstock_advisor.domain.entities.enums import ConversationAction
from jstock_advisor.domain.notification.incident_message import IncidentContent
from jstock_advisor.infrastructure.aws import conversation_state_store
from jstock_advisor.infrastructure.aws import trading_pause_config as _repo
from jstock_advisor.services import conversation_service as cs_mod
from jstock_advisor.services import trading_pause_service as mod
from jstock_advisor.services.conversation_service import ConversationReply, ConversationService
from jstock_advisor.services.trading_pause_service import PauseState, TradingPauseService

_NOW = dt.datetime(2026, 10, 18, 9, 0, tzinfo=dt.UTC)
_TOPIC_ENV = "INCIDENT_NOTIFICATION_TOPIC_ARN"

# 文字列リテラルで固定する(定数を import して比較しない = 定数ごと書き換えられても落ちる)
_NORMAL_PAUSED = (
    "ただいまシステムメンテナンス中のため、購入・売却操作を一時的に停止しています。"
    "しばらくしてからもう一度お試しください。"
)
_FETCH_FAILED = (
    "売買登録の停止設定を読み込めなかったため、購入・売却の登録を停止しています。"
    "ご自身で設定した一時停止ではありません。"
    "しばらくしてからもう一度お試しください。"
)
_THIRD_SENTENCE = "しばらくしてからもう一度お試しください。"


class _StubPause:
    """resolve_buy_sell_pause_state だけを持つ停止サービス(呼び出し回数を数える)。"""

    def __init__(self, state: PauseState) -> None:
        self.state = state
        self.calls = 0

    def resolve_buy_sell_pause_state(self, now: dt.datetime | None = None) -> PauseState:
        self.calls += 1
        return self.state


class _Untouchable:
    """属性に触れたら失敗する(停止中は確定の処理に進まないことの検査用)。"""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"state.{name} に触れた(停止中は確定の処理に進んではいけない)")


def _service(pause: Any) -> ConversationService:
    return ConversationService(
        portfolio_service=MagicMock(),
        transaction_history_service=MagicMock(),
        watchlist_service=MagicMock(),
        stock_display_name_resolver=MagicMock(),
        trading_pause_service=pause,
        holdings_view_service=MagicMock(),
        watchlist_view_service=MagicMock(),
        target_view_service=MagicMock(),
        stock_analysis_view_service=MagicMock(),
        holding_repository=MagicMock(),
        available_cash_service=MagicMock(),
    )


def _four_sites(service: ConversationService) -> list[ConversationReply]:
    """停止中に返信を返す 4 箇所(開始・入力・買付確定・売却確定)の返信。"""
    untouched: Any = _Untouchable()
    return [
        service._start("U1", ConversationAction.BUY, _NOW),
        service._handle_trade_input("U1", ConversationAction.BUY, "8306,100,1000,2026-10-18", _NOW),
        service._commit_buy("U1", untouched, _NOW),
        service._commit_sell("U1", untouched, _NOW),
    ]


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    envelopes: list[dict[str, Any]] = []
    monkeypatch.setenv(_TOPIC_ENV, "synthetic-topic")
    monkeypatch.setattr(mod, "publish_incident_envelope", lambda e, **_kw: envelopes.append(e))
    return envelopes


@pytest.fixture
def fetch_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """設定の取得を失敗させ、取得の回数を数える。"""
    calls: list[int] = []

    def _boom(_store_dir: Path | None = None) -> None:
        calls.append(1)
        raise RuntimeError("synthetic fetch failure")

    monkeypatch.setattr(_repo, "get", _boom)
    return calls


# =================================================================================================
# T1 / T2: 4 箇所の返信(通常停止は不変・取得失敗は確定文言)
# =================================================================================================


def test_t1_a_normal_pause_replies_with_the_unchanged_text_at_all_four_sites() -> None:
    replies = _four_sites(_service(_StubPause(PauseState.PAUSED)))
    assert [r.text for r in replies] == [_NORMAL_PAUSED] * 4


def test_t2_a_fetch_failure_replies_with_the_confirmed_text_at_all_four_sites() -> None:
    replies = _four_sites(_service(_StubPause(PauseState.PAUSED_FETCH_FAILED)))
    assert [r.text for r in replies] == [_FETCH_FAILED] * 4


def test_t2_the_second_sentence_is_the_core_of_the_decision() -> None:
    sentences = [s for s in _FETCH_FAILED.split("。") if s]
    assert len(sentences) == 3
    assert sentences[1] == "ご自身で設定した一時停止ではありません"
    assert "ご自身で設定した一時停止ではありません。" in cs_mod._TRADING_PAUSED_FETCH_FAILED
    assert cs_mod._TRADING_PAUSED_FETCH_FAILED == _FETCH_FAILED
    assert cs_mod._TRADING_PAUSED == _NORMAL_PAUSED


# =================================================================================================
# T3 / T4 / T9: 実サービスで取得失敗を仕込む(fail-safe・通知の回数・WATCH の短絡)
# =================================================================================================


def test_t3_a_fetch_failure_keeps_the_fail_safe_at_the_commit_sites(
    store_dir: Path, fetch_calls: list[int], published: list[dict[str, Any]]
) -> None:
    service = _service(TradingPauseService(store_dir=store_dir))
    untouched: Any = _Untouchable()  # 触れたら失敗 = 確定の処理(保有・購入ロットの書込)に進まない
    assert service._commit_buy("U1", untouched, _NOW).text == _FETCH_FAILED
    assert service._commit_sell("U1", untouched, _NOW).text == _FETCH_FAILED
    service._portfolio.assert_not_called()  # type: ignore[attr-defined]


def test_t4_one_notification_at_most_per_instance_across_the_four_sites(
    store_dir: Path, fetch_calls: list[int], published: list[dict[str, Any]]
) -> None:
    service = _service(TradingPauseService(store_dir=store_dir))
    replies = _four_sites(service)
    assert [r.text for r in replies] == [_FETCH_FAILED] * 4
    assert len(published) == 1


def test_t4_a_failed_publish_is_swallowed_and_the_reply_is_unchanged(
    store_dir: Path, monkeypatch: pytest.MonkeyPatch, fetch_calls: list[int]
) -> None:
    def _raise(_envelope: dict[str, Any], **_kw: Any) -> None:
        raise TimeoutError("synthetic publish timeout")

    monkeypatch.setenv(_TOPIC_ENV, "synthetic-topic")
    monkeypatch.setattr(mod, "publish_incident_envelope", _raise)
    service = _service(TradingPauseService(store_dir=store_dir))
    assert [r.text for r in _four_sites(service)] == [_FETCH_FAILED] * 4


def test_t9_watch_does_not_fetch_the_setting_and_does_not_notify(
    store_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetch_calls: list[int],
    published: list[dict[str, Any]],
) -> None:
    """取得失敗を仕込んで WATCH を 1 通送る → 取得 0 回・通知 0 回(返信水準ではなく回数で固定)。"""
    started: list[str] = []
    monkeypatch.setattr(
        conversation_state_store,
        "start_or_replace",
        lambda user_id, action, now: started.append(str(action)),
    )
    service = _service(TradingPauseService(store_dir=store_dir))
    reply = service._start("U1", ConversationAction.WATCH, _NOW)
    assert reply.text not in (_NORMAL_PAUSED, _FETCH_FAILED)
    assert started != []  # WATCH の会話は始まる(停止の対象外)
    assert fetch_calls == []
    assert published == []


def test_t9_control_buy_fetches_once_and_notifies_once(
    store_dir: Path, fetch_calls: list[int], published: list[dict[str, Any]]
) -> None:
    service = _service(TradingPauseService(store_dir=store_dir))
    assert service._start("U1", ConversationAction.BUY, _NOW).text == _FETCH_FAILED
    assert len(fetch_calls) == 1
    assert len(published) == 1


# =================================================================================================
# T5 / T6 / T7 / T10: 状態・wrapper・取得回数・knob
# =================================================================================================


def test_t5_not_paused_returns_no_pause_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(
        conversation_state_store,
        "start_or_replace",
        lambda user_id, action, now: started.append(str(action)),
    )
    service = _service(_StubPause(PauseState.NOT_PAUSED))
    assert service._pause_reply() is None
    reply = service._start("U1", ConversationAction.BUY, _NOW)
    assert reply.text not in (_NORMAL_PAUSED, _FETCH_FAILED)
    assert started != []


def test_t6_is_buy_sell_paused_keeps_its_return_values(
    store_dir: Path, monkeypatch: pytest.MonkeyPatch, published: list[dict[str, Any]]
) -> None:
    service = TradingPauseService(store_dir=store_dir)
    assert service.is_buy_sell_paused(_NOW) is False  # 未初期化
    assert service.resolve_buy_sell_pause_state(_NOW) is PauseState.NOT_PAUSED
    service.init_config(True, "tester", "setup", _NOW)
    assert service.is_buy_sell_paused(_NOW) is True
    assert service.resolve_buy_sell_pause_state(_NOW) is PauseState.PAUSED
    cfg = service.get_config()
    assert cfg is not None
    service.update_config(cfg.config_version, False, "tester", "setup", _NOW)
    assert service.is_buy_sell_paused(_NOW) is False
    assert service.resolve_buy_sell_pause_state(_NOW) is PauseState.NOT_PAUSED

    def _boom(_s: Path | None = None) -> None:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(_repo, "get", _boom)
    assert service.is_buy_sell_paused(_NOW) is True  # 取得失敗 = 停止(従来どおり)
    assert service.resolve_buy_sell_pause_state(_NOW) is PauseState.PAUSED_FETCH_FAILED


def test_t7_the_state_entry_fetches_exactly_once(
    store_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_get = _repo.get

    def _counting(store: Path | None = None) -> Any:
        calls.append(1)
        return real_get(store)

    monkeypatch.setattr(_repo, "get", _counting)
    TradingPauseService(store_dir=store_dir).resolve_buy_sell_pause_state(_NOW)
    assert len(calls) == 1
    calls.clear()
    TradingPauseService(store_dir=store_dir).is_buy_sell_paused(_NOW)
    assert len(calls) == 1


def test_t10_the_fallback_constant_is_still_the_knob(
    store_dir: Path, monkeypatch: pytest.MonkeyPatch, fetch_calls: list[int]
) -> None:
    service = TradingPauseService(store_dir=store_dir, notify_fetch_failures=False)
    assert service.resolve_buy_sell_pause_state(_NOW) is PauseState.PAUSED_FETCH_FAILED
    assert service.is_buy_sell_paused(_NOW) is True
    monkeypatch.setattr(mod, "_FALLBACK_PAUSE_BUY_SELL", False)
    assert service.resolve_buy_sell_pause_state(_NOW) is PauseState.NOT_PAUSED
    assert service.is_buy_sell_paused(_NOW) is False


# =================================================================================================
# T8: 文言の共有(通知との語・通常停止との第 3 文)
# =================================================================================================


def test_t8_the_reply_shares_its_wording_with_the_notification_and_the_normal_pause() -> None:
    notice = IncidentContent.LINE_WEBHOOK_TRADING_PAUSE_CONFIG_FETCH_FAILED.value
    for phrase in ("読み込めなかった", "ご自身で設定した一時停止ではありません"):
        assert phrase in notice
        assert phrase in _FETCH_FAILED
    # 第 3 文は通常停止の文と同一(通常停止の文を将来触ると取り残されるのを防ぐ)
    assert _FETCH_FAILED.endswith(_THIRD_SENTENCE)
    assert cs_mod._TRADING_PAUSED.endswith(_THIRD_SENTENCE)
    assert cs_mod._TRADING_PAUSED_FETCH_FAILED.endswith(_THIRD_SENTENCE)
