"""`jstock replay`: 過去の判断を、現行(OLD)と新しい(NEW)で見比べる集計(Issue #880 R1b。F-33)。

**手元のファイルだけを読み、集計の表を出す**。AWS・ネットワーク・時計に触れない。最初の版(R1b)の
入力は手元のファイル(fixture・手元のコピー)で、実データ(AWS)を読む段階(R2)は別の承認。

* 集計(件数・割合・遷移行列)は行うが、★ 判断(OLD / NEW)の再計算は行わない ★
  (判断は入力のファイルで受ける)
* 値(窓・起点・最小セル数・税と手数料・一致率・症例の指定)は、すべて事前登録の本体から読む。
  コマンドの引数や環境変数から値を受け取らない(登録に無い値で計算しない)
* 事前登録の照合(引用した登録 ID・ハッシュ・使った値)が一致しないときは、結果の表(第 1〜8 節)を
  出さず、第 9 節の照合の結果だけを出して終了する(終了コード 3)
* 出力は画面の表(標準出力)と、任意の `--out` のファイル。★ `--out` は存在しない path のみ(既に
  在れば上書きせず失敗)★。JSON は出さない。注意書き 2 文(`NOTICES`。#992 で一元化された定数)を、
  画面とファイルの両方の末尾に付ける
* ★ `AWS_LAMBDA_FUNCTION_NAME` が設定された環境では拒否 ★(#923 と同じ型。どの値でも・
  コマンドの最初に)

終了コード: 0 = 出力した / 1 = ファイルの読取・書込の失敗(`--out` が既に在る場合を含む)/
2 = 指定の誤り(拒否を含む)/ 3 = 事前登録の照合が一致しない / 4 = 入力のファイルの形が契約に反する
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from jstock_advisor.domain.backtest.errors import ReplayInputError
from jstock_advisor.domain.backtest.preregistration_check import RegistrationCheck
from jstock_advisor.domain.notification.allocation_shadow_cli_renderer import NOTICES
from jstock_advisor.services import replay_input_loader as loader
from jstock_advisor.services import replay_report as report

app = typer.Typer(
    help="過去の判断を現行(OLD)と新しい(NEW)で見比べる集計。手元のファイルだけを読みます。"
)

_AWS_OVERRIDE_ENV_VAR = "AWS_LAMBDA_FUNCTION_NAME"
EXIT_FILE_FAILURE = 1
EXIT_PREREG_MISMATCH = 3
EXIT_INPUT_CONTRACT = 4


def _reject_in_lambda_environment(command: str) -> None:
    """AWS_LAMBDA_FUNCTION_NAME が設定された環境では、読取・構築の前に拒否する(Issue #880 R1b)。

    replay は手元のファイルだけを読む。Lambda を装う設定(この環境変数)のもとで実行する意図は
    無いため、どの非空の値でも拒否する(空文字は未設定と同じ)。**各コマンドの最初に呼ぶこと**。
    """
    if os.environ.get(_AWS_OVERRIDE_ENV_VAR):
        raise typer.BadParameter(
            f"環境変数 {_AWS_OVERRIDE_ENV_VAR} が設定されているため、replay {command} は拒否します"
            "(replay は手元のファイルだけを読みます。Lambda の環境では実行しません)。"
            "手元で実行するなら、この環境変数を外してください",
            param_hint=_AWS_OVERRIDE_ENV_VAR,
        )


def _notices_block() -> str:
    return "\n" + "\n".join(f"※ {notice}" for notice in NOTICES) + "\n"


def _read(path: Path, *, max_bytes: int) -> str:
    try:
        if path.stat().st_size > max_bytes:
            raise ReplayInputError("input file is too large")
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ReplayInputError("input file is not valid UTF-8") from None
    except OSError:
        typer.echo("入力のファイルを読めませんでした", err=True)
        raise typer.Exit(EXIT_FILE_FAILURE) from None


def _fail_contract(error: ReplayInputError) -> typer.Exit:
    typer.echo(f"入力のファイルの形が契約に反します: {error}", err=True)
    return typer.Exit(EXIT_INPUT_CONTRACT)


def _emit(text: str, out: Path | None) -> None:
    """標準出力へ出す。`--out` があれば排他作成(既存なら失敗)でファイルへも同じ内容を書く。"""
    body = text + _notices_block()
    if out is not None:
        try:
            with out.open("x", encoding="utf-8") as handle:  # 'x' = 既に在れば上書きせず失敗
                handle.write(body)
        except FileExistsError:
            typer.echo("--out のファイルが既に在ります(上書きしません)", err=True)
            raise typer.Exit(EXIT_FILE_FAILURE) from None
        except OSError:
            typer.echo("--out のファイルへ書けませんでした", err=True)
            raise typer.Exit(EXIT_FILE_FAILURE) from None
    typer.echo(body, nl=False)


@app.command("check-prereg")
def check_prereg(
    prereg: Path = typer.Option(..., "--prereg", help="事前登録の本体(JSON)"),
    cite: Path = typer.Option(..., "--cite", help="引用する登録 ID・ハッシュ・使った値(JSON)"),
) -> None:
    """事前登録の本体と、引用(登録 ID・ハッシュ・使った値)が一致するかだけを調べる。"""
    _reject_in_lambda_environment("check-prereg")
    try:
        body, _ = loader.load_prereg(_read(prereg, max_bytes=loader.PREREG_MAX_BYTES))
        citation = loader.load_citation(_read(cite, max_bytes=loader.PREREG_MAX_BYTES))
    except ReplayInputError as error:
        raise _fail_contract(error) from None
    text, check = report.check_only_report(body, citation)
    _emit(text, None)
    raise typer.Exit(0 if check is RegistrationCheck.MATCH else EXIT_PREREG_MISMATCH)


@app.command("run")
def run(
    prereg: Path = typer.Option(..., "--prereg", help="事前登録の本体(JSON)"),
    cite: Path = typer.Option(..., "--cite", help="引用する登録 ID・ハッシュ・使った値(JSON)"),
    observations: Path = typer.Option(..., "--observations", help="観測(JSON Lines)"),
    pairs: Path = typer.Option(..., "--pairs", help="OLD と NEW の判断の対(JSON Lines)"),
    prices: Path = typer.Option(..., "--prices", help="価格の列(JSON Lines)"),
    out: Path | None = typer.Option(
        None, "--out", help="表を書くファイル。存在しない path のみ(既に在れば失敗)"
    ),
) -> None:
    """入力のファイルから集計の表(第 1〜9 節)を出す。事前登録の照合が不一致なら表を出さない。"""
    _reject_in_lambda_environment("run")
    try:
        body, params = loader.load_prereg(_read(prereg, max_bytes=loader.PREREG_MAX_BYTES))
        citation = loader.load_citation(_read(cite, max_bytes=loader.PREREG_MAX_BYTES))
        check = report.registration_check(body, citation)
        if check is not RegistrationCheck.MATCH:
            # 登録に無い値・登録と違う値では計算しない: データのファイルは読まない
            _emit(report.mismatch_report(body, citation, check), out)
            raise typer.Exit(EXIT_PREREG_MISMATCH)
        observation_rows = loader.load_observations(
            _read(observations, max_bytes=params.max_input_bytes),
            held_out_refs=params.held_out_refs,
            max_lines=params.max_input_lines,
        )
        pair_rows = loader.load_pairs(
            _read(pairs, max_bytes=params.max_input_bytes), max_lines=params.max_input_lines
        )
        series = loader.load_prices(
            _read(prices, max_bytes=params.max_input_bytes), max_lines=params.max_input_lines
        )
        loader.check_references(observation_rows, pair_rows)
        text = report.build_report(
            body=body,
            citation=citation,
            params=params,
            observations=observation_rows,
            pairs=pair_rows,
            prices=series,
        )
    except ReplayInputError as error:
        raise _fail_contract(error) from None
    _emit(text, out)
