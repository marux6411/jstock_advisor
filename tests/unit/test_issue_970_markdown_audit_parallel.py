"""Markdown 読取監査(#866)を並列実行(pytest-xdist)でも必ず機能させる(Issue #970 P1)の契約テスト。

原因(TARO の特定。#970 …6258244932): xdist では記録が worker プロセスに溜まり、controller の記録は
空のまま。worker の ``session.exitstatus`` の書換えも controller に伝わらない。subset 未掲載の
Markdown を読むテストを 1 件置くと、直列 = 失敗・``-n 2`` = 成功(無音で見逃す)。

修正: worker は判定せず記録を ``config.workeroutput`` へ入れる / controller は
``pytest_testnodedown``(optionalhook)で併合してから従来どおり判定する(判定のコードは 1 本)。

★ このテストは xdist なしで検証する ★(xdist をインストールしない・CI / workflow / 依存を変えない)。
併合は純粋関数で、worker と controller の hook は擬似の config / session で呼ぶ。実機の xdist での
検証(直列 = 失敗・``-n 2`` = 失敗)は、試験用 branch + USER の承認の後に行う(本 PR の範囲外)。

固定するもの
  ・併合(純粋関数): 違反あり / なし / 複数 worker・和集合・入力を変更しない・形が崩れた出力は
    違反として扱う(黙って捨てない)
  ・worker は判定しない(終了コードを変えず、記録を workeroutput へ入れる)
  ・controller は併合後に判定する(違反 → 終了コード 1・なし → 変えない)
  ・直列(workerinput なし)の判定は従来と同一
  ・判定のコードは 1 本 / optionalhook / xdist を import しない
fixture は合成のみ。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support import markdown_read_audit as mra

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "support" / "markdown_read_audit.py"
_KEY = mra.WORKEROUTPUT_KEY

_A = "tests/unit/test_synthetic_a.py"
_B = "tests/unit/test_synthetic_b.py"
_C = "tests/unit/test_synthetic_c.py"


# --- 部品 -------------------------------------------------------------------------------------


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> mra._Recorder:
    """本物の記録を退避し、このテストの間だけ空の記録を使う(終了後に元へ戻る)。"""
    monkeypatch.setattr(mra._RECORDER, "reads", {})
    monkeypatch.setattr(mra._RECORDER, "current", None)
    return mra._RECORDER


def _config(root: Path, *, worker: bool, collectonly: bool = False) -> SimpleNamespace:
    config = SimpleNamespace(
        rootpath=root,
        option=SimpleNamespace(collectonly=collectonly),
        pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
    )
    if worker:
        config.workerinput = {"workerid": "gw0"}
        config.workeroutput = {}
    return config


def _project(tmp_path: Path, subset: list[str]) -> Path:
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "ci_markdown_subset.txt").write_text("\n".join(subset) + "\n", "utf-8")
    return tmp_path


def _finish(config: SimpleNamespace) -> SimpleNamespace:
    session = SimpleNamespace(config=config, exitstatus=0)
    mra.pytest_sessionfinish(session, 0)  # type: ignore[arg-type]
    return session


def _node(output: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(workeroutput=output if output is not None else {})


# =================================================================================================
# 併合(純粋関数)
# =================================================================================================


def test_serialize_keeps_only_readers_sorted_and_json_friendly() -> None:
    reads = {_B: {"docs/z.md", "docs/a.md"}, _A: set(), _C: {"docs/c.md"}}
    assert mra.serialize_reads(reads) == {_B: ["docs/a.md", "docs/z.md"], _C: ["docs/c.md"]}


def test_merge_with_no_workers_returns_an_equal_copy_of_the_base() -> None:
    base = {_A: {"docs/a.md"}}
    merged = mra.merge_worker_reads(base, [])
    assert merged == base and merged is not base and merged[_A] is not base[_A]


def test_merge_takes_the_violations_of_every_worker_that_the_controller_never_saw() -> None:
    payloads = [{_A: ["docs/a.md"]}, {_B: ["docs/b.md"]}]
    merged = mra.merge_worker_reads({}, payloads)
    assert merged == {_A: {"docs/a.md"}, _B: {"docs/b.md"}}
    assert mra.find_unlisted_readers(merged, {_A}) == {_B: ["docs/b.md"]}


def test_merge_is_a_union_when_workers_read_through_the_same_file() -> None:
    payloads = [{_A: ["docs/a.md"]}, {_A: ["docs/b.md", "docs/a.md"]}]
    merged = mra.merge_worker_reads({_A: {"docs/c.md"}}, payloads)
    assert merged == {_A: {"docs/a.md", "docs/b.md", "docs/c.md"}}


def test_merge_with_no_violation_stays_clean() -> None:
    merged = mra.merge_worker_reads({}, [{}, {_A: []}, mra.CRASHED_WORKER])
    assert mra.find_unlisted_readers(merged, set()) == {}


def test_merge_does_not_change_its_inputs() -> None:
    base = {_A: {"docs/a.md"}}
    payload = {_B: ["docs/b.md"]}
    mra.merge_worker_reads(base, [payload])
    assert base == {_A: {"docs/a.md"}} and payload == {_B: ["docs/b.md"]}


def test_a_crashed_worker_is_skipped_because_xdist_reports_the_crash() -> None:
    marker = mra.CRASHED_WORKER
    assert mra.merge_worker_reads({_A: {"docs/a.md"}}, [marker]) == {_A: {"docs/a.md"}}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "missing",
        5,
        ["x"],
        {1: ["docs/a.md"]},
        {_A: "docs/a.md"},
        {_A: [1]},
        {_A: None},
        {_A: b"x"},
    ],
)
def test_an_unreadable_worker_output_is_reported_as_a_violation_not_dropped(payload: Any) -> None:
    merged = mra.merge_worker_reads({}, [payload])
    violations = mra.find_unlisted_readers(merged, {_A})
    assert list(violations) == [mra.UNREADABLE_WORKER_OUTPUT]


def test_an_unreadable_output_does_not_hide_the_other_workers_violations() -> None:
    merged = mra.merge_worker_reads({}, ["junk", {_B: ["docs/b.md"]}])
    assert set(mra.find_unlisted_readers(merged, set())) == {mra.UNREADABLE_WORKER_OUTPUT, _B}


# =================================================================================================
# worker: 判定しない・記録を渡す
# =================================================================================================


def test_a_worker_does_not_judge_and_hands_its_reads_to_the_controller(
    tmp_path: Path, recorder: mra._Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [])
    recorder.reads = {_A: {"docs/a.md"}}  # subset に無い = 直列なら違反
    config = _config(root, worker=True)
    session = _finish(config)
    assert session.exitstatus == 0  # ★ worker は終了コードを変えない ★
    assert capsys.readouterr().err == ""  # 何も報告しない
    assert config.workeroutput == {_KEY: {_A: ["docs/a.md"]}}


def test_a_worker_with_no_reads_still_hands_over_an_empty_record(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    config = _config(_project(tmp_path, []), worker=True)
    _finish(config)
    assert config.workeroutput == {
        _KEY: {}
    }  # キーが在る = 『読んだものが無い』と『出力が無い』を区別できる


def test_a_collect_only_worker_hands_over_nothing_and_never_judges(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    recorder.reads = {_A: {"docs/a.md"}}
    config = _config(_project(tmp_path, []), worker=True, collectonly=True)
    assert _finish(config).exitstatus == 0 and config.workeroutput == {}


# =================================================================================================
# controller: 併合してから従来どおり判定する
# =================================================================================================


def test_the_controller_fails_the_session_for_a_violation_that_only_a_worker_saw(
    tmp_path: Path, recorder: mra._Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_A])
    mra.pytest_testnodedown(_node({_KEY: {_A: ["docs/a.md"]}}), None)
    mra.pytest_testnodedown(_node({_KEY: {_B: ["docs/b.md"]}}), None)
    session = _finish(_config(root, worker=False))
    assert session.exitstatus == 1  # ★ 並列でも違反を検出する ★
    err = capsys.readouterr().err
    assert _B in err and "docs/b.md" in err and _A not in err


def test_the_controller_keeps_the_exit_status_when_no_worker_read_an_unlisted_markdown(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    root = _project(tmp_path, [_A])
    mra.pytest_testnodedown(_node({_KEY: {_A: ["docs/a.md"]}}), None)
    mra.pytest_testnodedown(_node({_KEY: {}}), None)
    assert _finish(_config(root, worker=False)).exitstatus == 0


def test_the_controller_also_judges_its_own_reads_together_with_the_workers(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    recorder.reads = {_C: {"docs/c.md"}}
    mra.pytest_testnodedown(_node({_KEY: {_A: ["docs/a.md"]}}), None)
    assert recorder.reads == {_C: {"docs/c.md"}, _A: {"docs/a.md"}}
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 1


def test_a_worker_that_finished_normally_without_an_output_is_a_violation(
    tmp_path: Path, recorder: mra._Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    mra.pytest_testnodedown(_node(None), None)  # キーが無い = 監査が機能していない
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 1
    assert mra.UNREADABLE_WORKER_OUTPUT in capsys.readouterr().err


def test_a_none_value_with_a_normal_exit_is_a_violation_not_a_skip(
    tmp_path: Path, recorder: mra._Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    """値が None の出力は『落ちた』ではなく形の崩れ。正常終了(error なし)でも違反として報告する。"""
    mra.pytest_testnodedown(_node({_KEY: None}), None)
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 1
    assert mra.UNREADABLE_WORKER_OUTPUT in capsys.readouterr().err


def test_a_none_value_is_a_violation_even_when_an_error_is_present(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    """『落ちた』の判断は error で行い、出力の値には委ねない(None は error があっても形の崩れ)。"""
    mra.pytest_testnodedown(_node({_KEY: None}), RuntimeError("crash"))
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 1


def test_a_worker_that_crashed_is_skipped_here_because_xdist_fails_the_run(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    mra.pytest_testnodedown(_node(None), RuntimeError("crash"))
    assert recorder.reads == {}
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 0


def test_a_node_without_a_workeroutput_attribute_is_handled(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    mra.pytest_testnodedown(SimpleNamespace(), None)
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 1


def test_an_end_to_end_flow_from_two_worker_records_to_the_verdict(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    """worker のプロセスの代わりに、記録 → 直列化 → 併合 → 判定 を通しで確認する。"""
    root = _project(tmp_path, [_A])
    outputs = []
    for reads in ({_A: {"docs/a.md"}}, {_B: {"docs/b.md"}}):
        recorder.reads = dict(reads)
        worker_config = _config(root, worker=True)
        _finish(worker_config)
        outputs.append(worker_config.workeroutput)
    recorder.reads = {}  # controller は自分では何も読んでいない(並列の実際の状態)
    for output in outputs:
        mra.pytest_testnodedown(_node(output), None)
    assert _finish(_config(root, worker=False)).exitstatus == 1


# =================================================================================================
# 直列は従来と同一
# =================================================================================================


def test_a_serial_run_judges_its_own_reads_exactly_as_before(
    tmp_path: Path, recorder: mra._Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_A])
    recorder.reads = {_B: {"docs/b.md"}, _A: {"docs/a.md"}}
    config = _config(root, worker=False)
    assert _finish(config).exitstatus == 1
    assert not hasattr(config, "workeroutput")  # 直列では何も書き込まない
    err = capsys.readouterr().err
    assert "Issue #866" in err and f"{_B}  <- docs/b.md" in err


def test_a_serial_run_without_a_violation_keeps_the_exit_status(
    tmp_path: Path, recorder: mra._Recorder
) -> None:
    recorder.reads = {_A: {"docs/a.md"}}
    assert _finish(_config(_project(tmp_path, [_A]), worker=False)).exitstatus == 0


def test_a_serial_collect_only_run_never_fails(tmp_path: Path, recorder: mra._Recorder) -> None:
    recorder.reads = {_B: {"docs/b.md"}}
    config = _config(_project(tmp_path, []), worker=False, collectonly=True)
    assert _finish(config).exitstatus == 0


# =================================================================================================
# 構造(AST)
# =================================================================================================


def _tree() -> ast.Module:
    return ast.parse(_AUDIT_PATH.read_text(encoding="utf-8"))


def _functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def test_the_verdict_code_exists_once_only() -> None:
    """判定(subset との突合と終了コードの書換え)は 1 つの関数にだけ在る。"""
    setters = []
    judges = []
    for function in _functions(_tree()):
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "exitstatus" for t in node.targets
            ):
                setters.append(function.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "find_unlisted_readers"
            ):
                judges.append(function.name)
    assert setters == ["pytest_sessionfinish"]
    assert judges == ["pytest_sessionfinish"]


def test_the_merge_hook_is_an_optional_hook_so_a_missing_xdist_does_not_break_registration() -> (
    None
):
    marker = mra.pytest_testnodedown.pytest_impl  # type: ignore[attr-defined]
    assert marker["optionalhook"] is True


def test_the_plugin_does_not_import_xdist() -> None:
    names: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    assert not {n for n in names if n.split(".")[0] in {"xdist", "execnet"}}
    assert "pytest" in names  # 走査は空振りでない


def test_a_worker_is_recognized_only_by_the_workerinput_attribute(tmp_path: Path) -> None:
    assert mra._is_xdist_worker(_config(tmp_path, worker=True)) is True
    assert mra._is_xdist_worker(_config(tmp_path, worker=False)) is False
