"""`services/replay_*.py`(Issue #880 R1b PR-1)の import の許可リストの契約テスト。

固定するもの: replay の services は、標準ライブラリ(許可リスト)と `jstock_advisor.domain.backtest`
(R1a の純粋関数)と、replay 同士だけを import する。AWS(boto3)・repository・config・通知・NEW の
module・時計・環境変数・ファイルを import しない(= 本番の投資判断・保存先へ繋がらない)。

許可リストは「部分一致の禁止リスト」ではなく、**成員の判定**(許可に無ければ落ちる)。
空振りでないこと(実際に import が走査されている・本物の違反で落ちる)は別のテストで固定する。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SERVICES = Path(__file__).resolve().parents[2] / "src" / "jstock_advisor" / "services"
_FILES = sorted(_SERVICES.glob("replay_*.py"))
_ALLOWED_STDLIB = {
    "__future__",
    "collections",
    "collections.abc",
    "dataclasses",
    "decimal",
    "fractions",
    "json",
    "typing",
}
_BACKTEST = "jstock_advisor.domain.backtest"
_OWN_SERVICES = {"jstock_advisor.services.replay_input_loader"}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def _allowed(name: str) -> bool:
    return (
        name in _ALLOWED_STDLIB
        or name == _BACKTEST
        or name.startswith(_BACKTEST + ".")
        or name in _OWN_SERVICES
    )


def _violations(names: set[str]) -> list[str]:
    return sorted(n for n in names if not _allowed(n))


def test_the_replay_services_are_found() -> None:
    assert [p.name for p in _FILES] == ["replay_input_loader.py", "replay_report.py"]


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_only_the_allowlisted_modules_are_imported(path: Path) -> None:
    assert _violations(_imports(path)) == []


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_the_scan_is_not_vacuous(path: Path) -> None:
    names = _imports(path)
    assert "__future__" in names
    assert any(n.startswith(_BACKTEST) for n in names)  # R1a を使っている = 走査の対象に居る


@pytest.mark.parametrize(
    "intruder",
    [
        "boto3",
        "botocore.exceptions",
        "jstock_advisor.config",
        "jstock_advisor.repositories.base",
        "jstock_advisor.services.available_cash",
        "jstock_advisor.domain.exit_architecture.decision",  # NEW の module
        "jstock_advisor.domain.backtestx",  # 名前が似ているだけの別 package(前方一致で通さない)
        "os",
        "pathlib",
        "datetime",
        "time",
        "socket",
        "requests",
    ],
)
def test_a_real_violation_is_caught(intruder: str) -> None:
    """本物の違反(production の型・AWS・NEW・OS)を足した合成の import 集合で、確かに落ちる。"""
    path = _FILES[0]
    assert _violations(_imports(path) | {intruder}) == [intruder]


def test_the_report_may_import_only_the_loader_among_the_services() -> None:
    names = {
        n
        for n in _imports(_SERVICES / "replay_report.py")
        if n.startswith("jstock_advisor.services")
    }
    assert names == {"jstock_advisor.services.replay_input_loader"}
    loader_services = {
        n
        for n in _imports(_SERVICES / "replay_input_loader.py")
        if n.startswith("jstock_advisor.services")
    }
    assert loader_services == set()  # loader は他の services に依存しない(循環を作らない)
