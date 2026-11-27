"""branch に open の PR があるかを調べる(Issue #971 D1。ci-branch.yml の最初の判定 step)。

`ci-branch.yml` は `issue-**` への push だけで起動し、**PR がまだ無い branch でだけ全体の pytest を
実行する**(Issue #230: PR を作る前にも全体回帰を見られる)。open の PR がある branch では、PR の実行
(`ci.yml`。必須チェックの出どころ)が全体の pytest を行うため、push 側の重複した実行を省く。

## 出力(`GITHUB_OUTPUT` の `open_pr`)

    true     open の PR がある → workflow は重い step を省く
    false    open の PR が無い → workflow は全体の pytest を実行する
    unknown  確認できなかった(認証・宛先・応答・通信の失敗)→ ★ 全体の pytest を実行する ★

workflow 側は `steps.<id>.outputs.open_pr != 'true'` で重い step を実行する。
したがって `unknown` も、
この script が起動しなかった・output が書けなかった場合も、省略ではなく実行になる(fail-safe)。

## 性質

* 標準ライブラリのみ(pip install の前に実行する。`scripts/ci_markdown_only.py` と同じ)。
* **常に exit 0**。判定の失敗で workflow を赤くしない(この workflow の job は必須チェックではない
  〔必須は `ci.yml` の PR の実行だけ〕ため、GitHub API の障害で merge は止まらない)。
* 認証情報・応答の本文は出力しない(例外の型名と固定の語だけを出す)。
* `GITHUB_OUTPUT` へ追記するのは `open_pr=...` の 1 行だけ。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping

_API_VERSION = "2022-11-28"
_DEFAULT_API_URL = "https://api.github.com"
_TIMEOUT_SECONDS = 15

#: (url, 認証の値) → 応答 JSON。テストでは fake に差し替える
Fetch = Callable[[str, str], object]


def _http_fetch(url: str, auth: str) -> object:
    request = urllib.request.Request(  # noqa: S310 - 宛先は https の GitHub API(環境変数で決まる)
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {auth}",
            "X-GitHub-Api-Version": _API_VERSION,
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def pulls_url(api_url: str, repository: str, branch: str) -> str:
    """その branch を head とする open の PR を引く URL(同じ repository 内の branch)。"""
    owner = repository.split("/", 1)[0]
    query = urllib.parse.urlencode({"state": "open", "head": f"{owner}:{branch}", "per_page": "1"})
    return f"{api_url.rstrip('/')}/repos/{repository}/pulls?{query}"


def has_open_pr(url: str, auth: str, fetch: Fetch = _http_fetch) -> bool:
    """open の PR が 1 件以上あるか。応答が配列でなければ例外(= 呼び出し側で unknown)。"""
    data = fetch(url, auth)
    if not isinstance(data, list):
        raise ValueError("unexpected response shape")
    return len(data) > 0


def decide(env: Mapping[str, str], fetch: Fetch = _http_fetch) -> tuple[str, str]:
    """(open_pr の値, 理由の固定語)。例外を外へ出さない(失敗は unknown)。"""
    repository = env.get("GITHUB_REPOSITORY", "")
    branch = env.get("GITHUB_REF_NAME", "")
    auth = env.get("GITHUB_TOKEN", "")
    if not repository or "/" not in repository or not branch or not auth:
        return "unknown", "MISSING_INPUT"
    api_url = env.get("GITHUB_API_URL", "") or _DEFAULT_API_URL
    try:
        found = has_open_pr(pulls_url(api_url, repository, branch), auth, fetch)
    except Exception as exc:  # noqa: BLE001 - どの失敗も unknown(全体の pytest を実行する側)
        return "unknown", f"CHECK_FAILED:{type(exc).__name__}"
    return ("true", "OPEN_PR_EXISTS") if found else ("false", "NO_OPEN_PR")


def main(env: Mapping[str, str] | None = None, fetch: Fetch = _http_fetch) -> int:
    values = dict(os.environ) if env is None else dict(env)
    value, reason = decide(values, fetch)
    print(f"OPEN_PR={value} reason={reason}")
    output_path = values.get("GITHUB_OUTPUT")
    if output_path:
        try:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(f"open_pr={value}\n")
        except OSError as exc:
            print(f"OPEN_PR: cannot write GITHUB_OUTPUT ({type(exc).__name__})")
    return 0  # 常に 0: 判定の失敗で workflow を赤くしない


if __name__ == "__main__":
    sys.exit(main())
