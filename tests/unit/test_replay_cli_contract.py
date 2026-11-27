"""`jstock replay`(Issue #880 R1b PR-1。`cli/replay.py`)の契約テスト。

固定するもの
  ・AWS_LAMBDA_FUNCTION_NAME が空でない値で設定されていたら、どのコマンドも終了コード 2 で拒否する
    (#923 と同じ型。ファイルの読取・boto3・ネットワークの前に。空文字は未設定と同じ)
  ・出力は画面と `--out` のファイルの両方の末尾に、注意書き 2 文(`NOTICES`)をそのまま付ける
  ・`--out` は存在しない path のみ(既に在れば終了コード 1 で、既存のファイルを変えない)
  ・事前登録の照合が不一致なら終了コード 3: 第 1〜8 節が無い ＋ 不一致の内容が在る(別々の主張)、
    データのファイルは読まない
  ・入力のファイルの形が契約に反したら終了コード 4
  ・boto3 を import せず、環境変数・引数から値を受け取らない
fixture は合成のみ。
"""

from __future__ import annotations

import ast
import re
import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jstock_advisor.cli import replay as cli_module
from jstock_advisor.cli.main import app as main_app
from jstock_advisor.domain.notification.allocation_shadow_cli_renderer import NOTICES
from tests.support import replay_fixtures as fx

_ENV = "AWS_LAMBDA_FUNCTION_NAME"
_runner = CliRunner()
_CLI_PATH = Path(__file__).resolve().parents[2] / "src" / "jstock_advisor" / "cli" / "replay.py"


@pytest.fixture
def forbidden_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """拒否が先に起きるなら 1 回も呼ばれない入口・AWS・ネットワーク。呼ばれたら記録して失敗する。"""
    calls: list[str] = []

    def make(name: str):  # type: ignore[no-untyped-def]
        def forbidden(*args: object, **kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"{name} は呼んではならない")

        return forbidden

    for target in ("boto3.client", "boto3.resource", "boto3.Session"):
        monkeypatch.setattr(target, make(target))
    monkeypatch.setattr(socket.socket, "connect", make("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", make("socket.create_connection"))
    monkeypatch.setattr(cli_module.loader, "load_prereg", make("load_prereg"))
    monkeypatch.setattr(cli_module.Path, "read_text", make("Path.read_text"))
    return calls


class _Files:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def prepare(
        self,
        *,
        cite_overrides: dict[str, Any] | None = None,
        prereg_overrides: dict[str, Any] | None = None,
        pair_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Path]:
        body = fx.prereg_body(**(prereg_overrides or {}))
        return {
            "prereg": self.write("prereg.json", fx.to_json(body)),
            "cite": self.write(
                "cite.json", fx.to_json(fx.citation_for(body, **(cite_overrides or {})))
            ),
            "observations": self.write("obs.jsonl", fx.to_lines(fx.observation_rows())),
            "pairs": self.write("pairs.jsonl", fx.to_lines(pair_rows or fx.pair_rows())),
            "prices": self.write("prices.jsonl", fx.to_lines(fx.price_rows())),
        }


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Files:
    monkeypatch.delenv(_ENV, raising=False)
    return _Files(tmp_path)


def _run_args(paths: dict[str, Path], out: Path | None = None) -> list[str]:
    args = ["run"]
    for key in ("prereg", "cite", "observations", "pairs", "prices"):
        args += [f"--{key}", str(paths[key])]
    if out is not None:
        args += ["--out", str(out)]
    return args


def _check_args(paths: dict[str, Path]) -> list[str]:
    return ["check-prereg", "--prereg", str(paths["prereg"]), "--cite", str(paths["cite"])]


def _notices() -> str:
    return "\n".join(f"※ {notice}" for notice in NOTICES) + "\n"


# =================================================================================================
# Lambda 環境の拒否(#923 と同じ型)
# =================================================================================================


@pytest.mark.parametrize("command", ["run", "check-prereg"])
def test_rejected_before_any_read_or_network_call(
    command: str, files: _Files, forbidden_calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = files.prepare()
    args = _run_args(paths) if command == "run" else _check_args(paths)
    monkeypatch.setenv(_ENV, "replay-cli")
    result = _runner.invoke(cli_module.app, args)
    assert result.exit_code == 2, result.output
    assert forbidden_calls == []  # ★ ファイルの読取も 0 回 ★
    assert _ENV in result.output and "拒否" in result.output and command in result.output


@pytest.mark.parametrize("value", ["x", "cli-target-aws-override", "0", "false", " "])
def test_any_non_empty_value_rejects(
    value: str, files: _Files, forbidden_calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = files.prepare()
    monkeypatch.setenv(_ENV, value)
    for args in (_run_args(paths), _check_args(paths)):
        assert _runner.invoke(cli_module.app, args).exit_code == 2, (value, args[0])
    assert forbidden_calls == []


def test_the_rejection_leaves_no_out_file(
    files: _Files, forbidden_calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = files.prepare()
    out = files.directory / "out.txt"
    monkeypatch.setenv(_ENV, "x")
    assert _runner.invoke(cli_module.app, _run_args(paths, out)).exit_code == 2
    assert not out.exists()


@pytest.mark.parametrize("command", ["run", "check-prereg"])
def test_unset_environment_variable_runs(
    command: str, files: _Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = files.prepare()
    monkeypatch.delenv(_ENV, raising=False)
    args = _run_args(paths) if command == "run" else _check_args(paths)
    result = _runner.invoke(cli_module.app, args)
    assert result.exit_code == 0, result.output
    assert "拒否" not in result.output


def test_an_empty_value_is_the_same_as_unset(
    files: _Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = files.prepare()
    monkeypatch.setenv(_ENV, "")
    assert _runner.invoke(cli_module.app, _run_args(paths)).exit_code == 0


# =================================================================================================
# 出力: 画面とファイルの両方の末尾に注意書き 2 文
# =================================================================================================


def test_stdout_ends_with_the_two_notices_verbatim(files: _Files) -> None:
    result = _runner.invoke(cli_module.app, _run_args(files.prepare()))
    assert result.exit_code == 0, result.output
    assert len(NOTICES) == 2
    assert result.output.endswith(_notices())
    assert "[9] " in result.output


def test_the_out_file_has_the_same_content_and_ends_with_the_notices(files: _Files) -> None:
    out = files.directory / "report.txt"
    result = _runner.invoke(cli_module.app, _run_args(files.prepare(), out))
    assert result.exit_code == 0, result.output
    written = out.read_text(encoding="utf-8")
    assert written.endswith(_notices())
    assert written == result.output  # 画面とファイルは同じ内容(JSON は出さない)
    assert written.startswith("[1] ")  # JSON ではなく表


def test_check_prereg_also_ends_with_the_notices(files: _Files) -> None:
    result = _runner.invoke(cli_module.app, _check_args(files.prepare()))
    assert result.exit_code == 0 and result.output.endswith(_notices())
    assert "照合の結果: MATCH" in result.output


def test_an_existing_out_file_is_never_overwritten(files: _Files) -> None:
    out = files.directory / "report.txt"
    out.write_text("keep me", encoding="utf-8")
    result = _runner.invoke(cli_module.app, _run_args(files.prepare(), out))
    assert result.exit_code == 1, result.output
    assert out.read_text(encoding="utf-8") == "keep me"  # ★ 上書きしない ★
    assert "[1] " not in result.output  # 書けなかったので画面にも表を出さない


def test_a_missing_directory_for_out_fails_with_exit_1(files: _Files) -> None:
    out = files.directory / "no_such_dir" / "report.txt"
    result = _runner.invoke(cli_module.app, _run_args(files.prepare(), out))
    assert result.exit_code == 1 and not out.exists()


def test_the_output_is_not_json(files: _Files) -> None:
    result = _runner.invoke(cli_module.app, _run_args(files.prepare()))
    help_text = _runner.invoke(cli_module.app, ["run", "--help"]).output
    assert "--json" not in help_text and "--format" not in help_text  # 出力形式の選択肢が無い
    assert not result.output.lstrip().startswith("{")


# =================================================================================================
# 事前登録の照合が不一致: 第 1〜8 節を出さない ＋ 不一致の内容を出す(別々)・データは読まない
# =================================================================================================

_MISMATCHES = [
    {"content_hash": "0" * 64},
    {"registration_id": "REG-OTHER"},
]


@pytest.mark.parametrize("cite_overrides", _MISMATCHES)
def test_a_mismatch_exits_3_and_prints_no_result_sections(
    cite_overrides: dict[str, Any], files: _Files
) -> None:
    result = _runner.invoke(cli_module.app, _run_args(files.prepare(cite_overrides=cite_overrides)))
    assert result.exit_code == 3, result.output
    for number in range(1, 9):
        assert f"[{number}] " not in result.output  # 結果の表が無い


@pytest.mark.parametrize(
    ("cite_overrides", "expected"),
    [(_MISMATCHES[0], "HASH_MISMATCH"), (_MISMATCHES[1], "ID_MISMATCH")],
)
def test_a_mismatch_still_shows_what_did_not_match_and_the_notices(
    cite_overrides: dict[str, Any], expected: str, files: _Files
) -> None:
    """上の「表が無い」とは別の主張: 不一致の内容(第 9 節の照合の結果)は出る。"""
    result = _runner.invoke(cli_module.app, _run_args(files.prepare(cite_overrides=cite_overrides)))
    assert "[9] " in result.output
    assert f"照合の結果: {expected}" in result.output
    assert result.output.endswith(_notices())


def test_a_mismatch_does_not_read_the_data_files(files: _Files) -> None:
    paths = files.prepare(cite_overrides={"content_hash": "0" * 64})
    for key in ("observations", "pairs", "prices"):
        paths[key] = files.directory / f"missing_{key}.jsonl"  # 存在しない
    result = _runner.invoke(cli_module.app, _run_args(paths))
    assert result.exit_code == 3, result.output  # 読んでいれば 1 か 4 になる


def test_a_mismatch_with_out_writes_the_mismatch_report_to_the_file(files: _Files) -> None:
    out = files.directory / "report.txt"
    paths = files.prepare(cite_overrides={"registration_id": "REG-OTHER"})
    result = _runner.invoke(cli_module.app, _run_args(paths, out))
    assert result.exit_code == 3
    written = out.read_text(encoding="utf-8")
    assert "照合の結果: ID_MISMATCH" in written and "[1] " not in written
    assert written.endswith(_notices())


def test_check_prereg_mismatch_exits_3(files: _Files) -> None:
    paths = files.prepare(cite_overrides={"content_hash": "0" * 64})
    result = _runner.invoke(cli_module.app, _check_args(paths))
    assert result.exit_code == 3 and "照合の結果: HASH_MISMATCH" in result.output


# =================================================================================================
# 入力の契約違反・ファイルの失敗
# =================================================================================================


def test_a_contract_violation_exits_4_without_echoing_values(files: _Files) -> None:
    rows = fx.pair_rows()
    rows[0]["old"]["action"] = "SECRET_VALUE_XYZ"
    result = _runner.invoke(cli_module.app, _run_args(files.prepare(pair_rows=rows)))
    assert result.exit_code == 4, result.output
    assert "SECRET_VALUE_XYZ" not in result.output  # 値を出さない(固定の文言だけ)
    assert "[1] " not in result.output


def test_an_unreadable_input_exits_1(files: _Files) -> None:
    paths = files.prepare()
    paths["prices"] = files.directory / "missing.jsonl"
    result = _runner.invoke(cli_module.app, _run_args(paths))
    assert result.exit_code == 1, result.output


def test_a_file_larger_than_the_registered_limit_exits_4(files: _Files) -> None:
    paths = files.prepare(prereg_overrides={"max_input_bytes": 10})
    result = _runner.invoke(cli_module.app, _run_args(paths))
    assert result.exit_code == 4, result.output


def test_invalid_utf8_exits_4(files: _Files) -> None:
    paths = files.prepare()
    paths["observations"].write_bytes(b"\xff\xfe\x00")
    assert _runner.invoke(cli_module.app, _run_args(paths)).exit_code == 4


# =================================================================================================
# 登録・構造
# =================================================================================================


def test_the_command_is_registered_in_the_main_app() -> None:
    result = _runner.invoke(main_app, ["replay", "--help"])
    assert result.exit_code == 0 and "run" in result.output and "check-prereg" in result.output


def test_the_commands_take_no_parameter_values_from_arguments_or_environment() -> None:
    """値(窓・起点・最小セル数・税と手数料・一致率・症例)は事前登録の本体だけから読む。"""
    result = _runner.invoke(cli_module.app, ["run", "--help"])
    for forbidden in ("--min-cell", "--window", "--origin", "--cost", "--agreement", "--held-out"):
        assert forbidden not in result.output
    tree = ast.parse(_CLI_PATH.read_text(encoding="utf-8"))
    env_reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in {"environ", "getenv"}
    ]
    assert len(env_reads) == 1  # 唯一の読取 = Lambda の拒否の判定(値は使わず、有無だけ)
    envvar_options = [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "envvar"]
    assert envvar_options == []


def test_the_cli_does_not_import_aws_or_a_clock() -> None:
    tree = ast.parse(_CLI_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    assert not {n for n in names if n.split(".")[0] in {"boto3", "botocore", "datetime", "time"}}
    assert not {n for n in names if n.startswith("jstock_advisor.repositories")}
    assert any(n.startswith("jstock_advisor.domain.backtest") for n in names)  # 走査は空振りでない
    assert re.search(r"def _reject_in_lambda_environment", _CLI_PATH.read_text(encoding="utf-8"))


def _first_statement_calls_the_guard(function: ast.FunctionDef) -> bool:
    body = [
        n
        for n in function.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    first = body[0]
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "_reject_in_lambda_environment"
    )


def test_every_command_calls_the_guard_first() -> None:
    """新しいコマンドを足して拒否の呼出が漏れると落ちる(#923 の回帰防止と同じ型)。"""
    tree = ast.parse(_CLI_PATH.read_text(encoding="utf-8"))
    commands = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "command"
            for d in n.decorator_list
        )
    ]
    assert sorted(c.name for c in commands) == ["check_prereg", "run"]
    assert [c.name for c in commands if not _first_statement_calls_the_guard(c)] == []
