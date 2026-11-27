"""Issue #798(受入条件 3): 運用手順書36.3の6(E1)の反例を、実行できる形で固定する。

E1は「buildの入力(`config/`・`src/`・`infra/layer/`)の作業ツリーのバイト列が、gitの内容と
同一であること」を、`git ls-files --eol`の作業ツリー側(`w/`)が`crlf`または`mixed`の行が
1件も無いこと、で確認する手作業のコマンドである。W10(2026-10-04)では、`git status`が
clean と見せる CRLF の作業ツリーから build され、ConfigLayer のバイト列が git の内容(LF)と
異なった。E1はその状態を検出するために足された検査で、これまで反例(CRLFの作業ツリー)を
実行する契約テストが無かった。

本テストは隔離repo(tmp_path)で次を固定する。ユーザーのrepo・AWS・Productionには触れない。

* 反例: `core.autocrlf=true`で作った作業ツリー(index=LF・作業ツリー=CRLF)を、E1が
  `config/`・`src/`・`infra/layer/`のそれぞれで検出する。`mixed`も検出する
* この状態を`git status --porcelain`は検出しない(= E1が必要な理由)
* 正例: LFまたは改行なし・binaryは検出しない。対象外のpath(`other/`)のCRLFも検出しない
* 是正: 運用手順書が選択肢とする「対象ファイルの再checkout」で、E1が空になる
* 手順書との連動: テストが実行するコマンドと判定は、運用手順書36.3の6の文面から読み出す
  (手順書とテストが食い違えば落ちる)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPERATIONS_MANUAL = _REPO_ROOT / "docs" / "operations_manual.md"
_GITATTRIBUTES = _REPO_ROOT / ".gitattributes"

_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git command is required")

_E1_ROOTS = ("config", "src", "infra/layer")
_CRLF = b"key: value\r\nother: 1\r\n"
_LF = b"key: value\nother: 1\n"
_MIXED = b"key: value\r\nother: 1\n"


def _git(cwd: Path, *args: str, autocrlf: str = "true") -> str:
    assert _GIT is not None
    result = subprocess.run(  # noqa: S603 - 固定のgit引数のみ(外部入力なし)
        [
            _GIT,
            "-c",
            f"core.autocrlf={autocrlf}",
            "-c",
            "core.safecrlf=false",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=synthetic",
            "-c",
            "user.email=synthetic@example.invalid",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def _e1_command_and_pattern() -> tuple[list[str], re.Pattern[str]]:
    """運用手順書36.3の6から、E1の列挙コマンドと検出パターンを読み出す。"""
    text = _OPERATIONS_MANUAL.read_text(encoding="utf-8")
    command_match = re.search(r"^\s*git ls-files --eol -- ([^|\n]+)$", text, re.MULTILINE)
    assert command_match is not None, "36.3の6のE1の列挙コマンドが見つからない"
    pattern_match = re.search(
        r"git ls-files --eol -- [^|\n]+\| grep -E '([^']+)'", text
    )
    assert pattern_match is not None, "36.3の6のE1の検出パターンが見つからない"
    pathspec = command_match.group(1).split()
    return ["ls-files", "--eol", "--", *pathspec], re.compile(pattern_match.group(1))


def _e1_hits(repo: Path) -> list[str]:
    command, pattern = _e1_command_and_pattern()
    out = _git(repo, *command)
    return [line for line in out.splitlines() if pattern.search(line)]


def _write(repo: Path, relative: str, payload: bytes) -> None:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _commit_all(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "synthetic")


def _new_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


def test_the_e1_command_in_the_manual_covers_exactly_the_three_build_input_roots() -> None:
    """手順書のE1の対象は`config`・`src`・`infra/layer`の3つ(buildの入力)。"""
    command, pattern = _e1_command_and_pattern()

    # 対象は集合で比べる(`-- a b` と `-- b a` は同じ集合を返す。順序の入れ替えで落とさない)。
    assert command[:3] == ["ls-files", "--eol", "--"]
    assert len(command[3:]) == len(set(command[3:]))
    assert set(command[3:]) == set(_E1_ROOTS)
    # パターンは作業ツリー側のcrlf / mixed を拾い、lf / none / -text は拾わない。
    assert pattern.search("i/lf    w/crlf  attr/ \tconfig/a.yaml")
    assert pattern.search("i/lf    w/mixed attr/ \tsrc/a.py")
    assert not pattern.search("i/lf    w/lf    attr/ \tconfig/a.yaml")
    assert not pattern.search("i/none  w/none  attr/ \tconfig/empty.yaml")
    assert not pattern.search("i/-text w/-text attr/ \tconfig/b.bin")


def test_a_crlf_worktree_that_git_status_calls_clean_is_detected_in_every_root(
    tmp_path: Path,
) -> None:
    """W10の状態(index=LF・作業ツリー=CRLF・`git status`はclean)を、E1は3つの対象で検出する。"""
    repo = _new_repo(tmp_path, "crlf")
    for root in _E1_ROOTS:
        _write(repo, f"{root}/sample.yaml", _CRLF)
    _commit_all(repo)

    # 前提(反例の成立): indexはLF・作業ツリーはCRLF・statusはclean。
    listing = _git(repo, "ls-files", "--eol", "--", *_E1_ROOTS)
    assert all(line.startswith("i/lf") and "w/crlf" in line for line in listing.splitlines())
    assert len(listing.splitlines()) == len(_E1_ROOTS)
    assert _git(repo, "status", "--porcelain") == ""

    hits = _e1_hits(repo)

    assert len(hits) == len(_E1_ROOTS)
    for root in _E1_ROOTS:
        assert any(line.endswith(f"{root}/sample.yaml") for line in hits), root


def test_a_mixed_line_ending_worktree_is_detected(tmp_path: Path) -> None:
    """`mixed`(CRLFとLFが同じファイルに混在)も検出する(`w/crlf`だけを見る検査は落ちる)。"""
    repo = _new_repo(tmp_path, "mixed")
    _write(repo, "config/mixed.yaml", _MIXED)
    _commit_all(repo)
    assert "w/mixed" in _git(repo, "ls-files", "--eol", "--", "config")

    hits = _e1_hits(repo)

    assert len(hits) == 1
    assert hits[0].endswith("config/mixed.yaml")


def test_lf_none_and_binary_worktrees_are_not_detected(tmp_path: Path) -> None:
    """LF・改行なし(空)・binaryの作業ツリーは検出しない(偽陽性で止めない)。"""
    repo = _new_repo(tmp_path, "clean")
    _write(repo, "config/lf.yaml", _LF)
    _write(repo, "src/empty.py", b"")
    _write(repo, "infra/layer/payload.bin", b"\x00\x01binary\r\nbytes\r\n")
    _commit_all(repo)
    listing = _git(repo, "ls-files", "--eol", "--", *_E1_ROOTS)
    assert "w/lf" in listing and "w/none" in listing and "w/-text" in listing

    assert _e1_hits(repo) == []


def test_crlf_outside_the_three_build_input_roots_is_not_detected(tmp_path: Path) -> None:
    """buildの入力でないpath(`other/`・`docs/`)のCRLFはE1の対象外(範囲を広げない)。"""
    repo = _new_repo(tmp_path, "outside")
    _write(repo, "other/sample.yaml", _CRLF)
    _write(repo, "docs/sample.md", _CRLF)
    _write(repo, "config/lf.yaml", _LF)
    _commit_all(repo)
    assert "w/crlf" in _git(repo, "ls-files", "--eol", "--", "other", "docs")

    assert _e1_hits(repo) == []


def test_re_checkout_with_the_real_gitattributes_clears_the_detection(tmp_path: Path) -> None:
    """手順書が選ぶ是正(対象ファイルの再checkout)で、E1が空になる。

    実際の`.gitattributes`(`eol=lf`)を持つrepoで、CRLFの作業ツリーを検出した後、
    対象ファイルを再checkoutするとLFに戻り、E1が空になる。
    """
    repo = _new_repo(tmp_path, "recheckout")
    shutil.copyfile(_GITATTRIBUTES, repo / ".gitattributes")
    for root in _E1_ROOTS:
        _write(repo, f"{root}/sample.yaml", _CRLF)
    _commit_all(repo)
    assert len(_e1_hits(repo)) == len(_E1_ROOTS)

    for root in _E1_ROOTS:
        (repo / root / "sample.yaml").unlink()
    _git(repo, "checkout", "--", *_E1_ROOTS)

    assert _e1_hits(repo) == []
    for root in _E1_ROOTS:
        assert (repo / root / "sample.yaml").read_bytes() == _LF, root
