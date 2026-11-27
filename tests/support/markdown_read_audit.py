"""Markdown を実際に読むテストが subset に載っていることを検査する(Issue #866)。

subset の定義 = ``tests/ci_markdown_subset.txt``。

## なぜ要るか

CI の test job は、Markdown だけを変える PR では全体 pytest を省略し、**Markdown を実際に読む
テスト(subset)だけ**を実行する(``scripts/ci_markdown_only.py``・``.github/workflows/ci.yml``)。
subset に「Markdown を読むテスト」が載っていないと、Markdown の変更でそのテストが壊れても
PR では見えず、main への push(全体 pytest)で初めて赤になる。main の赤は release・lock 解放の
証跡(MAIN_CI_PASS)を壊す。

## 何をするか(動的な検出)

全体 pytest の実行中に、各テストファイルが **tests/ の外の Markdown を開いた** こと
(``open`` の audit event)と、``git show`` / ``git cat-file`` で Markdown を読んだこと
(``subprocess.Popen`` の audit event)を、実行中のテストファイルに紐付けて記録する。
session の終わりに、記録されたテストファイルが subset に無ければ **session を失敗にする**
(終了コード 1・欠けている path と読んだ Markdown を表示)。

静的な検出(文字列定数に ``.md`` が現れる等)は、走査で間接的に Markdown を読むテスト
(``test_scan_for_pii``)を見つけられなかった(実測: 6 ファイル中 5 ファイル)。
そのため動的な検出を主とする。

## 検出できないこと(限界)

- **別プロセスの中**の Markdown 読み(``sys.addaudithook`` はプロセスごと)。tests が起こす
  子プロセスが Markdown を読む場合は検出できない。現時点では、子プロセスを使う各テストを
  個別に確認した結果(Issue #866 の記録)で、子が tests/ の外の Markdown を読むものは無い
- ディレクトリの列挙(``glob`` / ``rglob`` / ``iterdir`` / ``scandir``)や ``exists`` / ``stat``
  だけで Markdown の有無・数に依存するテスト(open が起きない)。静的な検査
  (``tests/unit/test_ci_markdown_subset.py``)が補う
- ``git show`` / ``git cat-file`` 以外の外部コマンドによる読み(``cat`` 等)
- ``tests/`` の下の Markdown(fixture)。tests/ を変更する PR は Markdown-only にならない

## collect-only の run では判定しない

``tests/unit/test_issue_277_cohort_markers.py`` は子の ``pytest --collect-only`` を起こす。子も
``tests/conftest.py`` からこの plugin を読み込むが、収集だけの run は subset の判定対象ではなく、
**子の終了コードを変えてはならない**ため、判定(session の失敗化)を行わない。

## 並列実行(pytest-xdist。Issue #970 P1)

xdist ではテストが worker プロセスで走り、記録は worker の ``_RECORDER`` に溜まる。controller
(終了コードを決めるプロセス)の記録は空のままで、worker の ``session.exitstatus`` の書換えは
controller に伝わらない。そのため、判定を各プロセスで行うと、並列では違反を**無音で**見逃す
(実測: subset 未掲載の Markdown を読むテストを 1 件置くと、直列 = 失敗・``-n 2`` = 成功)。

- **worker は判定しない**。自分の記録を ``config.workeroutput``(worker → controller の正規の
  受け渡し口)へ入れて終える(``serialize_reads``)
- **controller は ``pytest_testnodedown``(optionalhook)で各 worker の記録を併合**する
  (``merge_worker_reads``。純粋関数)。そのあと ``pytest_sessionfinish`` が、併合後の記録で
  従来と**同じ判定コード**(1 本)を実行する
- 直列の run では worker も併合も起きず、従来と完全に同じ。xdist が入っていない環境でも壊れない
  (``optionalhook=True``)
- 併合できない出力(形が崩れている・worker が出力を返さなかった)は、**違反として扱う**
  (黙って捨てない。検出力を落とす方向に倒さない)

## オーバーヘッド

audit hook は全 audit event で呼ばれる。実行中のテストが無いとき・``open`` と
``subprocess.Popen`` 以外の event は、先頭で即座に return する。
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest

SUBSET_RELPATH = "tests/ci_markdown_subset.txt"

_MARKDOWN_SUFFIXES = (".md", ".markdown")
# 仮想環境・VCS の内部・依存 package 内の Markdown は「リポジトリの文書」ではない
_IGNORED_PARTS = frozenset(
    {".venv", "venv", "site-packages", "node_modules", ".git", "__pycache__"}
)
_GIT_READ_SUBCOMMANDS = frozenset({"show", "cat-file"})


# --- 純粋な部分(unit test の対象) ---------------------------------------------------


def repo_markdown_relpath(path: str, root: str, cwd: str | None = None) -> str | None:
    """``path`` がリポジトリの文書としての Markdown なら repo 相対 path(/ 区切り)を返す。

    ``tests/`` の下(fixture)・仮想環境・repo の外は対象外(None)。
    """
    if not path.lower().endswith(_MARKDOWN_SUFFIXES):
        return None
    absolute = path if os.path.isabs(path) else os.path.join(cwd or os.getcwd(), path)
    absolute = os.path.abspath(absolute)
    root = os.path.abspath(root)
    norm_abs, norm_root = os.path.normcase(absolute), os.path.normcase(root)
    if not norm_abs.startswith(norm_root + os.sep):
        return None
    relative = absolute[len(root) + 1 :].replace(os.sep, "/")
    parts = relative.split("/")
    if _IGNORED_PARTS.intersection(parts) or parts[0] == "tests":
        return None
    return relative


def argv_from_popen_event(value: object) -> list[str] | None:
    """``subprocess.Popen`` の audit event の args を argv にする。

    POSIX では引数のリスト、Windows では連結済みのコマンドライン文字列で届く。
    """
    if isinstance(value, (list, tuple)):
        return [str(token) for token in value]
    if isinstance(value, str):
        try:
            tokens = shlex.split(value, posix=False)
        except ValueError:
            tokens = value.split()
        return [token.strip("\"'") for token in tokens]
    return None


def git_markdown_read_target(argv: Iterable[object]) -> str | None:
    """``git show`` / ``git cat-file`` の引数に Markdown の path があれば、その path を返す。"""
    tokens = [str(token) for token in argv]
    if not tokens or os.path.basename(tokens[0].replace("\\", "/")).lower() not in {
        "git",
        "git.exe",
    }:
        return None
    subcommand: str | None = None
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-c", "-C"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        subcommand = token
        index += 1
        break
    if subcommand not in _GIT_READ_SUBCOMMANDS:
        return None
    for token in tokens[index:]:
        # ``<rev>:<path>`` の形(policy_check の revision reader)も ``<path>`` の形も扱う
        candidate = token.split(":", 1)[-1]
        if candidate.lower().endswith(_MARKDOWN_SUFFIXES):
            return candidate
    return None


def parse_subset(text: str) -> set[str]:
    """``tests/ci_markdown_subset.txt`` を読む(1 行 1 path・# 行と空行は無視)。"""
    entries: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped)
    return entries


def find_unlisted_readers(reads: Mapping[str, set[str]], subset: set[str]) -> dict[str, list[str]]:
    """Markdown を読んだテストファイルのうち、subset に載っていないもの(読んだ Markdown つき)。"""
    return {
        test_file: sorted(targets)
        for test_file, targets in sorted(reads.items())
        if targets and test_file not in subset
    }


# --- 並列実行(xdist)での受け渡し。純粋な部分(unit test の対象)-----------------------

#: ``config.workeroutput`` の key(worker が記録を入れ、controller が読む)
WORKEROUTPUT_KEY = "markdown_read_audit_reads"
#: 『worker が異常終了して出力が無い』ことを表す目印(値の None とは別物。None は形の崩れ = 違反)
CRASHED_WORKER: Final = object()
#: 併合できない worker の出力を表す擬似のテストファイル名(subset に無い = 違反として報告される)
UNREADABLE_WORKER_OUTPUT = "(unreadable worker output: cannot verify Markdown reads)"


def serialize_reads(reads: Mapping[str, set[str]]) -> dict[str, list[str]]:
    """worker の記録を、``workeroutput`` に載せられる形(str → str のリスト)にする。"""
    return {
        test_file: sorted(targets) for test_file, targets in sorted(reads.items()) if targets
    }


def _is_well_formed_payload(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for test_file, targets in payload.items():
        if not isinstance(test_file, str):
            return False
        if isinstance(targets, (str, bytes)) or not isinstance(targets, (list, tuple, set)):
            return False
        if not all(isinstance(target, str) for target in targets):
            return False
    return True


def merge_worker_reads(
    base: Mapping[str, set[str]], payloads: Iterable[object]
) -> dict[str, set[str]]:
    """controller の記録 ``base`` に、各 worker の出力 ``payloads`` を併合した新しい記録を返す。

    同じテストファイルの読んだ Markdown は和集合にする。``CRASHED_WORKER``(目印の object)だけを
    『出力が無いことが分かっている』(worker が落ちた等。落ちたこと自体は xdist が失敗として報告する)
    として読み飛ばす。**``None`` を含め、形が崩れた出力は読み飛ばさず**、subset に載らない擬似の
    テストファイルとして記録する(= 違反として報告される。検出力を落とす方向に倒さない)。
    入力は変更しない。
    """
    merged: dict[str, set[str]] = {test_file: set(targets) for test_file, targets in base.items()}
    for payload in payloads:
        if payload is CRASHED_WORKER:
            continue
        if not _is_well_formed_payload(payload):
            merged.setdefault(UNREADABLE_WORKER_OUTPUT, set()).add("worker output")
            continue
        assert isinstance(payload, Mapping)
        for test_file, targets in payload.items():
            merged.setdefault(test_file, set()).update(targets)
    return merged


def _is_xdist_worker(config: pytest.Config) -> bool:
    """xdist の worker プロセスか(worker の config にだけ ``workerinput`` がある)。"""
    return hasattr(config, "workerinput")


# --- 実行時の記録 ----------------------------------------------------------------------


@dataclass
class _Recorder:
    root: str = ""
    current: str | None = None
    reads: dict[str, set[str]] = field(default_factory=dict)

    def record(self, target: str) -> None:
        if self.current is not None:
            self.reads.setdefault(self.current, set()).add(target)


_RECORDER = _Recorder()
_HOOK_INSTALLED = False


def _audit(event: str, args: tuple[Any, ...]) -> None:
    # 実行中のテストが無いとき・関係のない event は何もしない(オーバーヘッドの最小化)
    recorder = _RECORDER
    if recorder.current is None:
        return
    if event == "open":
        target = args[0] if args else None
        if isinstance(target, (str, os.PathLike)):
            text = os.fspath(target)
            if isinstance(text, str) and text.lower().endswith(_MARKDOWN_SUFFIXES):
                relative = repo_markdown_relpath(text, recorder.root)
                if relative is not None:
                    recorder.record(relative)
    elif event == "subprocess.Popen":
        argv = argv_from_popen_event(args[1]) if len(args) > 1 else None
        if argv is not None:
            relative = git_markdown_read_target(argv)
            if relative is not None:
                recorder.record(relative)


def _install_hook() -> None:
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True


def _relative_to_root(path: object, root: str) -> str | None:
    try:
        return Path(str(path)).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return None


def pytest_configure(config: pytest.Config) -> None:
    _RECORDER.root = str(config.rootpath)
    _RECORDER.reads = {}
    _RECORDER.current = None
    _install_hook()


def pytest_collectstart(collector: pytest.Collector) -> None:
    # 収集時(module の import)の Markdown 読みも、その module に紐付ける
    if isinstance(collector, pytest.Module):
        _RECORDER.current = _relative_to_root(collector.path, _RECORDER.root)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _RECORDER.current = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Any:
    _RECORDER.current = _relative_to_root(item.path, _RECORDER.root)
    try:
        return (yield)
    finally:
        _RECORDER.current = None


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    """controller: worker が終わったとき、その worker の記録を併合する(xdist の hook)。

    xdist が入っていない環境では呼ばれない(``optionalhook=True`` により、hook の仕様が無くても
    登録で壊れない)。
    """
    output = getattr(node, "workeroutput", None)
    payload: object
    if isinstance(output, Mapping) and WORKEROUTPUT_KEY in output:
        payload = output[WORKEROUTPUT_KEY]
    elif error is not None:
        payload = CRASHED_WORKER  # 異常終了。落ちたことは xdist が失敗として報告する
    else:
        payload = "missing"  # 正常終了したのに出力が無い = 形の崩れ(違反として扱う)
    _RECORDER.reads = merge_worker_reads(_RECORDER.reads, [payload])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if config.option.collectonly:
        return  # 収集だけの run は subset の判定対象ではなく、終了コードを変えない
    if _is_xdist_worker(config):
        # worker では判定しない: 記録を controller へ渡し、controller が併合後に判定する
        config.workeroutput[WORKEROUTPUT_KEY] = serialize_reads(_RECORDER.reads)  # type: ignore[attr-defined]
        return
    root = Path(config.rootpath)
    subset_path = root / SUBSET_RELPATH
    try:
        subset = parse_subset(subset_path.read_text(encoding="utf-8"))
    except OSError:
        if not _RECORDER.reads:
            return  # 定義が無く、Markdown を読んだテストも無い run(別の project)は対象外
        subset = set()
    violations = find_unlisted_readers(_RECORDER.reads, subset)
    if not violations:
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    lines = [
        "",
        "ERROR: Markdown を実際に読むテストが subset に載っていません (Issue #866)。",
        "  subset の定義: tests/ci_markdown_subset.txt",
        "  Markdown のみの変更では全体 pytest が省略され、このテストは PR で実行されません。",
        "  下のテストファイルを tests/ci_markdown_subset.txt に追加してください。",
    ]
    for test_file, targets in violations.items():
        lines.append(f"  {test_file}  <- {', '.join(targets)}")
    for line in lines:
        if reporter is not None:
            reporter.write_line(line, red=True)
        else:  # pragma: no cover - terminalreporter 無効 (-p no:terminal) の run
            print(line, file=sys.stderr)
    session.exitstatus = 1
