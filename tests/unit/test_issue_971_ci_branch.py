"""#971 D1: branch への push の重複した全体 pytest を省く新 workflow(ci-branch.yml)の契約テスト。

USER 確定(#122 …6258191583)の項目を、workflow の構造と動作で固定する:

  1 新設の ci-branch.yml は 1 job(full pytest のみ)
  2 PR 作成前の branch では full pytest を実行する
  3 open PR がある branch では重複する full pytest を省略する
  4 PR の必須チェック 8 件は維持する(ci.yml の job・名前・PR の trigger は不変)
  5 GitHub API の確認に失敗したら full pytest を実行する(fail-safe)
  6 main の CI・Ruleset・strict・既存の回帰検出力を維持する(ci.yml の main の push は従来どおり)
  7 新 job 名は required checks と区別する
  8 workflow の構造と動作を契約テストで固定する(本ファイル)
"""

from __future__ import annotations

import ast
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import ci_branch_open_pr as script  # noqa: E402

_WORKFLOWS = _ROOT / ".github" / "workflows"
_BRANCH_YML = _WORKFLOWS / "ci-branch.yml"
_CI_YML = _WORKFLOWS / "ci.yml"

# Ruleset の required status checks の context 名(= ci.yml の job 名。
# test_ci_markdown_only.py も参照)
_REQUIRED = {
    "lint",
    "test",
    "typecheck",
    "secret-scan",
    "dependency-audit",
    "pii-scan",
    "layer-lock-drift",
    "pii-scan-commit-messages",
}


def _load(path: Path) -> dict[Any, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(config: dict[Any, Any]) -> dict[str, Any]:
    triggers = config.get("on", config.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _steps() -> list[dict[str, Any]]:
    steps = _load(_BRANCH_YML)["jobs"]["branch-full-pytest"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(needle: str) -> dict[str, Any]:
    found = [
        s for s in _steps() if needle in str(s.get("run", "")) or needle in str(s.get("uses", ""))
    ]
    assert len(found) == 1, (needle, len(found))
    return found[0]


# =================================================================================================
# 1・7 job は 1 つ・名前は required と別
# =================================================================================================


def test_the_new_workflow_has_exactly_one_job_and_its_name_is_not_a_required_check() -> None:
    jobs = _load(_BRANCH_YML)["jobs"]
    assert list(jobs) == ["branch-full-pytest"]
    assert set(jobs).isdisjoint(_REQUIRED)
    assert "name" not in jobs["branch-full-pytest"]  # name で required の名前を名乗らない
    assert "needs" not in jobs["branch-full-pytest"]


def test_the_workflow_name_is_not_ci() -> None:
    # workflow 名は concurrency の group に入る。ci.yml(name: CI)と別でなければ group が衝突する
    assert _load(_BRANCH_YML)["name"] == "CI branch"
    assert _load(_CI_YML)["name"] == "CI"


# =================================================================================================
# 起動と権限
# =================================================================================================


def test_it_runs_only_on_pushes_to_issue_branches() -> None:
    triggers = _triggers(_load(_BRANCH_YML))
    assert set(triggers) == {"push"}  # PR・main の push・手動は ci.yml
    assert triggers["push"] == {"branches": ["issue-**"]}


def test_permissions_include_contents_read_and_pull_requests_read_only() -> None:
    # permissions を書くと列挙しない scope は none。contents: read が無いと checkout できない
    assert _load(_BRANCH_YML)["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_the_concurrency_group_contains_the_workflow_name_so_it_never_cancels_a_pr_run() -> None:
    branch = _load(_BRANCH_YML)["concurrency"]
    ci = _load(_CI_YML)["concurrency"]
    assert "github.workflow" in branch["group"]  # #230: push が PR の実行を取り消さない理由
    assert "github.ref" in branch["group"]
    assert branch["cancel-in-progress"] is True  # 同じ branch の古い実行は取り消す
    # ci.yml の group とは必ず別(workflow 名が違う)
    assert "github.workflow" in ci["group"]
    assert _load(_BRANCH_YML)["name"] != _load(_CI_YML)["name"]


# =================================================================================================
# 3・5 open PR の確認と、省略の条件(fail-safe)
# =================================================================================================


def test_the_open_pr_check_runs_first_and_cannot_fail_the_job() -> None:
    steps = _steps()
    check = next(i for i, s in enumerate(steps) if "ci_branch_open_pr.py" in str(s.get("run", "")))
    heavy = [
        i
        for i, s in enumerate(steps)
        if "pip install" in str(s.get("run", "")) or "pytest" in str(s.get("run", ""))
    ]
    assert heavy and check < min(heavy)  # 判定は重い step より前
    step = steps[check]
    assert step.get("id") == "open-pr"
    assert step.get("continue-on-error") is True
    assert "if" not in step  # 常に実行する
    assert step["env"] == {"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}  # 渡す認証は 1 つだけ


def test_every_heavy_step_is_skipped_only_when_the_check_said_true() -> None:
    """`!= 'true'` の形: output が無い・unknown は省略にならない(fail-safe)。"""
    for needle in (
        "ci_markdown_only.py",
        "pip install",
        "python -m pytest tests -q",
        "ci_markdown_subset.txt",
    ):
        condition = str(_step(needle)["if"])
        assert "steps.open-pr.outputs.open_pr != 'true'" in condition, needle
        assert "== 'false'" not in condition  # unknown を『実行しない』側へ倒さない
        assert "open_pr == " not in condition.replace("!= 'true'", "")


def test_the_skip_notice_appears_only_for_an_open_pr() -> None:
    notice = _step("An open pull request exists")
    assert notice["if"] == "steps.open-pr.outputs.open_pr == 'true'"
    assert "pytest" not in str(notice["run"]).replace(
        "full pytest", ""
    )  # 通知のみ(テストを起動しない)


def test_the_full_pytest_step_is_identical_to_the_one_in_ci_yml() -> None:
    branch = _step("python -m pytest tests -q")
    ci_steps = _load(_CI_YML)["jobs"]["test"]["steps"]
    ci = next(s for s in ci_steps if "python -m pytest tests -q" in str(s.get("run", "")))
    assert shlex.split(branch["run"]) == shlex.split(
        ci["run"]
    )  # 同じ対象・同じ引数(回帰検出力を変えない)
    assert "steps.classify.outputs.markdown_only != 'true'" in branch["if"]


def test_the_markdown_only_subset_semantics_of_866_are_kept() -> None:
    subset = _step("ci_markdown_subset.txt")
    ci_steps = _load(_CI_YML)["jobs"]["test"]["steps"]
    ci_subset = next(s for s in ci_steps if "ci_markdown_subset.txt" in str(s.get("run", "")))
    assert subset["run"] == ci_subset["run"]
    assert "steps.classify.outputs.markdown_only == 'true'" in subset["if"]
    classify = _step("ci_markdown_only.py")
    assert classify["id"] == "classify"
    assert classify["continue-on-error"] is True  # 判定が失敗したら output なし → full(fail-safe)


def test_checkout_fetches_full_history_for_the_merge_base_comparison() -> None:
    checkout = next(s for s in _steps() if "actions/checkout" in str(s.get("uses", "")))
    assert checkout["with"] == {"fetch-depth": 0}


def test_no_other_secret_or_write_scope_is_used() -> None:
    text = _BRANCH_YML.read_text(encoding="utf-8")
    assert text.count("secrets.") == 1  # GITHUB_TOKEN のみ
    assert "write" not in text.replace("pull-requests: read", "").replace("contents: read", "")


# =================================================================================================
# 4・6 ci.yml(必須チェック・main の CI)は維持
# =================================================================================================


def test_ci_yml_keeps_every_job_and_the_pull_request_trigger() -> None:
    config = _load(_CI_YML)
    assert set(config["jobs"]) == _REQUIRED | {"catalog-coverage"}
    triggers = _triggers(config)
    assert set(triggers) == {"push", "pull_request", "workflow_dispatch"}
    assert (
        triggers["pull_request"] is None
    )  # base の絞りも paths も無い(どの PR でも必須チェックが出る)
    assert triggers["push"]["branches"] == ["main"]  # issue-** は ci-branch.yml へ
    assert "issue-**" not in json.dumps(triggers)


def test_ci_yml_concurrency_is_unchanged_main_is_never_cancelled() -> None:
    concurrency = _load(_CI_YML)["concurrency"]
    assert concurrency["group"] == "ci-${{ github.workflow }}-${{ github.head_ref || github.ref }}"
    assert concurrency["cancel-in-progress"] == "${{ github.ref != 'refs/heads/main' }}"


def test_the_two_workflows_never_define_the_same_job_name() -> None:
    assert set(_load(_CI_YML)["jobs"]).isdisjoint(set(_load(_BRANCH_YML)["jobs"]))


# =================================================================================================
# script の動作(fake の fetch。ネットワークを使わない)
# =================================================================================================

_ENV = {
    "GITHUB_REPOSITORY": "owner-a/repo-a",
    "GITHUB_REF_NAME": "issue-971-x",
    "GITHUB_TOKEN": "unit-test-value",
}


def _fetch_returning(payload: object) -> Any:
    calls: list[tuple[str, str]] = []

    def fetch(url: str, auth: str) -> object:
        calls.append((url, auth))
        return payload

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_an_open_pr_gives_true() -> None:
    fetch = _fetch_returning([{"number": 1}])
    assert script.decide(_ENV, fetch) == ("true", "OPEN_PR_EXISTS")
    url, auth = fetch.calls[0]
    assert url.startswith("https://api.github.com/repos/owner-a/repo-a/pulls?")
    assert "state=open" in url and "head=owner-a%3Aissue-971-x" in url and "per_page=1" in url
    assert auth == "unit-test-value"


def test_no_open_pr_gives_false() -> None:
    assert script.decide(_ENV, _fetch_returning([])) == ("false", "NO_OPEN_PR")


@pytest.mark.parametrize("payload", [{"message": "Not Found"}, "oops", None, 3, {"items": [1]}])
def test_an_unexpected_response_shape_gives_unknown(payload: object) -> None:
    value, reason = script.decide(_ENV, _fetch_returning(payload))
    assert value == "unknown" and reason.startswith("CHECK_FAILED:")


@pytest.mark.parametrize(
    "error", [OSError("boom"), TimeoutError(), ValueError("x"), RuntimeError("y")]
)
def test_any_failure_gives_unknown_with_only_the_exception_type(error: Exception) -> None:
    def fetch(url: str, auth: str) -> object:
        raise error

    value, reason = script.decide(_ENV, fetch)
    assert value == "unknown"
    assert reason == f"CHECK_FAILED:{type(error).__name__}"
    assert "boom" not in reason  # 例外の文面を出さない


@pytest.mark.parametrize("missing", ["GITHUB_REPOSITORY", "GITHUB_REF_NAME", "GITHUB_TOKEN"])
def test_a_missing_input_gives_unknown_and_never_calls_the_api(missing: str) -> None:
    fetch = _fetch_returning([{"number": 1}])
    env = {k: v for k, v in _ENV.items() if k != missing}
    assert script.decide(env, fetch) == ("unknown", "MISSING_INPUT")
    assert fetch.calls == []


def test_a_repository_without_a_slash_is_unknown() -> None:
    assert (
        script.decide({**_ENV, "GITHUB_REPOSITORY": "noslash"}, _fetch_returning([]))[0]
        == "unknown"
    )


def test_the_api_url_comes_from_the_environment_when_given() -> None:
    fetch = _fetch_returning([])
    script.decide({**_ENV, "GITHUB_API_URL": "https://ghes.example/api/v3/"}, fetch)
    assert fetch.calls[0][0].startswith("https://ghes.example/api/v3/repos/owner-a/repo-a/pulls?")


def test_main_writes_exactly_one_output_line_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "output.txt"
    assert script.main({**_ENV, "GITHUB_OUTPUT": str(out)}, _fetch_returning([{"n": 1}])) == 0
    assert out.read_text(encoding="utf-8") == "open_pr=true\n"
    printed = capsys.readouterr().out
    assert "unit-test-value" not in printed  # 認証の値を出さない


def test_main_returns_zero_even_when_everything_fails(tmp_path: Path) -> None:
    def fetch(url: str, auth: str) -> object:
        raise OSError("down")

    out = tmp_path / "output.txt"
    assert script.main({**_ENV, "GITHUB_OUTPUT": str(out)}, fetch) == 0
    assert out.read_text(encoding="utf-8") == "open_pr=unknown\n"


def test_main_survives_an_unwritable_output_path(tmp_path: Path) -> None:
    bad = tmp_path / "no-such-dir" / "output.txt"
    assert script.main({**_ENV, "GITHUB_OUTPUT": str(bad)}, _fetch_returning([])) == 0


def test_the_script_uses_only_the_standard_library() -> None:
    tree = ast.parse((_ROOT / "scripts" / "ci_branch_open_pr.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names), sorted(imported - set(sys.stdlib_module_names))


def test_the_script_never_prints_the_credential_value() -> None:
    source = (_ROOT / "scripts" / "ci_branch_open_pr.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "auth" not in names  # print の引数に認証の値を渡さない


def test_every_run_command_is_valid_shell_text_not_truncated_by_a_yaml_comment() -> None:
    """` #` は YAML の plain scalar の中で『コメントの開始』になり、残りが黙って切り捨てられる。

    実機検証で、通知の echo が `(... Issue #971)` の `#971)` 以降を切り捨てられ、閉じない引用符に
    なって step が exit 2 で失敗した(Issue #971 の試験用 branch)。各 run が引用符の閉じた文として
    読めることと、元の text に ` #` を含む plain scalar が無いことを固定する。
    """
    text = _BRANCH_YML.read_text(encoding="utf-8")
    for step in _steps():
        run = step.get("run")
        if run is not None:
            assert shlex.split(str(run))  # 引用符が閉じている(ValueError にならない)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("run:") and not stripped.startswith("run: |"):
            assert " #" not in stripped.split("run:", 1)[1], line  # コメントとして切り捨てられる
