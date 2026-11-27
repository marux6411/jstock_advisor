"""Issue #958: 資金配分 Shadow の CLI で、client の生成が失敗したときの失敗表示の契約テスト。

背景: `--aws-profile` に存在しない・壊れた profile を渡すと、boto3 が client の生成
(`build_client`)の途中で例外(`ProfileNotFound` ほか)を送出する。この例外は読取の失敗
(`ReadFailure`)ではないため、固定の文言 + 終了コードの契約から漏れて traceback になっていた。

固定するもの(Issue 本文の受入条件):
  1 存在しない profile → 固定の文言・既存の読取失敗の終了コード(1)・traceback なし
  2 client の生成の他の例外(設定の不正・SDK の初期化失敗)でも同様。出力に profile 名・例外の文面・
    パスが含まれない(sentinel の漏れ走査)
  3 既存の読取失敗(権限・表なし・接続・その他)の文言・終了コードは変わらない
  4 範囲: CLI の失敗表示だけ。引数の検査の失敗(終了コード 2)は、client の生成の前のまま
対象のコマンド: today / summary / detail(3 つとも `_build_source` を共有する)。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jstock_advisor.cli import allocation as allocation_cli
from jstock_advisor.cli import allocation_detail as detail_cli
from jstock_advisor.cli import allocation_summary as summary_cli
from jstock_advisor.cli.main import app as main_app
from jstock_advisor.domain.business_calendar import BusinessCalendar
from jstock_advisor.infrastructure.aws.allocation_shadow_reader import (
    ReadFailure,
    ReadFailureKind,
)

_runner = CliRunner()
_DAY = "2026-10-14"
_REGION = "ap-northeast-1"
_TABLE = "jstock-audit_log"
_PROFILE_SENTINEL = "sentinel-profile-zz-958"
_ERROR_SENTINEL = "sentinel-error-text-zz-958"
_PATH_SENTINEL = "/sentinel/path/zz-958/config"
_FIXED_ERROR_TEXT = "記録の読取に失敗しました"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AWS の設定を一時の空 file へ向ける(利用者の実際の設定・資格情報に触れない)。"""
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    empty = tmp_path / "empty_aws_file"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))
    today = dt.date(2026, 10, 14)
    calendar = BusinessCalendar(frozenset(), frozenset())
    for module in (allocation_cli, detail_cli, summary_cli):
        monkeypatch.setattr(module, "_today_jst", lambda: today)
        monkeypatch.setattr(module, "_default_calendar", lambda: calendar)
    monkeypatch.setattr(allocation_cli, "load_labels_counting_dropped", lambda: ({}, 0))
    monkeypatch.setattr(detail_cli, "_load_labels_or_none", lambda: {})


def _dynamo_args(command: str, *extra: str) -> list[str]:
    common = ["--backend", "dynamodb", "--aws-region", _REGION, "--confirm-table", _TABLE]
    if command == "summary":
        return ["allocation", "summary", "--from", _DAY, "--to", _DAY, *common, *extra]
    return ["allocation", command, "--date", _DAY, *common, *extra]


_COMMANDS = ["today", "summary", "detail"]


def _assert_fixed_failure(result: Any, *sentinels: str) -> None:
    assert result.exit_code == 1, result.output
    # 例外が漏れていない(traceback になっていない): CliRunner では typer.Exit は SystemExit になる
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output
    assert _FIXED_ERROR_TEXT in result.output
    for sentinel in sentinels:
        assert sentinel not in result.output


# --- 受入条件 1: 存在しない profile ---------------------------------------------------------------


@pytest.mark.parametrize("command", _COMMANDS)
def test_a_nonexistent_profile_is_a_fixed_read_failure_not_a_traceback(command: str) -> None:
    """boto3 の実物が ProfileNotFound を送出する(AWS へは接続しない。設定 file は空)。"""
    result = _runner.invoke(main_app, _dynamo_args(command, "--aws-profile", _PROFILE_SENTINEL))

    _assert_fixed_failure(result, _PROFILE_SENTINEL)


# --- 受入条件 2: client の生成の他の例外 ---------------------------------------------------


class _BrokenSdkError(Exception):
    def __str__(self) -> str:
        return f"{_ERROR_SENTINEL} {_PATH_SENTINEL}"


@pytest.mark.parametrize("command", _COMMANDS)
@pytest.mark.parametrize(
    "error",
    [
        _BrokenSdkError(),
        RuntimeError(f"{_ERROR_SENTINEL} {_PATH_SENTINEL}"),
        OSError(_PATH_SENTINEL),
    ],
)
def test_any_error_while_building_the_client_is_the_same_fixed_failure(
    monkeypatch: pytest.MonkeyPatch, command: str, error: Exception
) -> None:
    def broken(*, region: str, profile: str | None) -> Any:
        raise error

    monkeypatch.setattr(allocation_cli, "build_client", broken)

    result = _runner.invoke(main_app, _dynamo_args(command, "--aws-profile", _PROFILE_SENTINEL))

    _assert_fixed_failure(result, _PROFILE_SENTINEL, _ERROR_SENTINEL, _PATH_SENTINEL)


@pytest.mark.parametrize("command", _COMMANDS)
def test_the_failure_carries_no_chained_exception_text(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """`from None`: 元の例外が `__cause__` として持ち越されず、traceback へ出ない。

    固定しているのは `__cause__ is None` と `__suppress_context__ is True`(元の例外の文面が
    traceback に出ない性質)。`__context__` 自体は Python の仕様で保持される(ここでは見ない)。
    """

    def broken(*, region: str, profile: str | None) -> Any:
        raise RuntimeError(_ERROR_SENTINEL)

    monkeypatch.setattr(allocation_cli, "build_client", broken)
    selection = allocation_cli.select_backend(
        allocation_cli.Backend.DYNAMODB, _REGION, _TABLE, None, None
    )

    with pytest.raises(ReadFailure) as caught:
        allocation_cli._build_source(selection)

    assert caught.value.kind is ReadFailureKind.ERROR
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert _ERROR_SENTINEL not in str(caught.value)


# --- 受入条件 3: 既存の読取失敗の文言・終了コードは変わらない -------------------------------

_EXISTING_TEXT = {
    ReadFailureKind.ACCESS_DENIED: "記録を読む権限がありません(AccessDenied)",
    ReadFailureKind.NOT_FOUND_TABLE: "対象の表が見つかりません",
    ReadFailureKind.CONNECTION: "接続に失敗しました",
    ReadFailureKind.ERROR: "記録の読取に失敗しました",
}


def test_the_existing_failure_texts_are_unchanged() -> None:
    assert allocation_cli._FAILURE_TEXT == _EXISTING_TEXT


@pytest.mark.parametrize("command", _COMMANDS)
@pytest.mark.parametrize("kind", list(ReadFailureKind))
def test_every_failure_kind_from_building_the_source_uses_its_fixed_text(
    monkeypatch: pytest.MonkeyPatch, command: str, kind: ReadFailureKind
) -> None:
    def build(selection: Any) -> Any:
        raise ReadFailure(kind)

    for module in (allocation_cli, detail_cli, summary_cli):
        monkeypatch.setattr(module, "_build_source", build)

    result = _runner.invoke(main_app, _dynamo_args(command))

    assert result.exit_code == 1
    assert f"読取に失敗しました: {_EXISTING_TEXT[kind]}" in result.output


# --- 受入条件 4: 範囲(引数の検査は client の生成より前のまま)-------------------------


@pytest.mark.parametrize("command", _COMMANDS)
def test_argument_errors_still_exit_with_2_before_any_client_is_built(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    built: list[Any] = []

    def spy(*, region: str, profile: str | None) -> Any:
        built.append(profile)
        raise AssertionError("client を作ってはいけない")

    monkeypatch.setattr(allocation_cli, "build_client", spy)
    args = _dynamo_args(command)
    # --confirm-table を落とす(= 起動前の拒否。終了コード 2)
    index = args.index("--confirm-table")
    del args[index : index + 2]

    result = _runner.invoke(main_app, args)

    assert result.exit_code == 2
    assert built == []
