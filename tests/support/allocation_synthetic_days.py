"""資金配分 Shadow の合成記録の builder(Issue #934 BT-3。tests のみ)。

`jstock allocation summary` を、日ごとに内容の違う営業日分の記録で通しで試すための部品。
**実在の銘柄・保有・価格・owner を含まない**(owner は `owner-a` 等の架空の識別子から、テスト用の
鍵で作った符号。銘柄コードは `0001` 起点の連番、金額・価格・数量は例示の値)。乱数を使わない
(同じ呼出しは同じ記録になる)。

この module は記録の『形』を作るだけで、期待値(どの日が数える日か等)は持たない。期待値は
呼ぶ側の契約テストが構成から手で導いて固定する(実装の再計算を期待値にしない)。
10/27〜28 の実データの読取の前に『期待される記録の形』の見本としても使える。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from jstock_advisor.domain.entities.identifier_ref import KeyMaterial, keyed_owner_ref
from jstock_advisor.domain.notification.allocation_shadow_cli_view import (
    day_skip_key,
    index_key,
    items_key,
)

BATCH: Final = "buy-candidates-synthetic"
#: テスト用の鍵(秘密ではない。本番の鍵とは無関係)
KEY: Final = KeyMaterial(key_version=1, key=b"s" * 32)

#: 秘密(出力に出てはならない値)の見本。記録へ仕込み、出力に現れないことを確かめる
CANARY_STOCK_CODE: Final = "CANARYCODE"
CANARY_CASH: Final = "98765432"
CANARY_PRICE: Final = "7654321"

V1_MODEL: Final = "raer-synthetic-v1"
V2_MODEL: Final = "raer-synthetic-v2"


def ref(owner: str) -> str:
    """owner の識別子(架空)→ 鍵つきの符号(記録の `owner_ref`)。同じ owner は常に同じ符号。"""
    return str(keyed_owner_ref(owner, KEY))


def digest(owner: str) -> str:
    """符号の末尾(表示に使われる部分)。出力にこの全文が出ないことを確かめるのに使う。"""
    return ref(owner).rsplit(":", 1)[-1]


def business_days(start: str, end: str, holidays: Sequence[str] = ()) -> list[str]:
    """start〜end(両端を含む)の平日のうち、holidays を除いた日付(営業日の見本。順序は日付順)。"""
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    days: list[str] = []
    day = first
    while day <= last:
        if day.weekday() < 5 and day.isoformat() not in holidays:
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return days


# --- 候補・owner の記録 ------------------------------------------------------------------------


def candidate(
    seq: int,
    *,
    raer: str = "AVAILABLE",
    decision: str = "ALLOCATED",
    reason: str = "ALLOCATED",
    buy_action: str = "BUY",
    stock_code: str | None = None,
    price: str = "1000",
    amount: str = "1000000",
) -> dict[str, Any]:
    """候補 1 件。raer = AVAILABLE / PROVISIONAL / UNAVAILABLE。"""
    allocated = decision == "ALLOCATED"
    raer_out: dict[str, Any] = {"state": raer}
    if raer != "UNAVAILABLE":
        raer_out.update({"value": f"{30 + seq}.0", "confidence": "MEDIUM"})
    return {
        "seq": seq,
        "stock_code": stock_code or f"{seq:04d}",
        "attrs": {"buy_action": buy_action},
        "price": {"current_price": price},
        "raer_out": raer_out,
        "allocation": {
            "rank": seq,
            "decision": decision,
            "reason_code": reason,
            "shares": 1000 if allocated else 0,
            "amount": amount if allocated else "0",
            "cap_limited_shares": 2000,
            "cash_shortfall": "0",
        },
    }


def owner_record(
    business_date: str,
    owner: str,
    *,
    outcome: str = "COMPUTED",
    skip_reason: str | None = None,
    status: str = "ALLOCATED_SOME",
    unavailable_reason: str | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    model_version: str = V1_MODEL,
    config_hash: str = "cfg-synthetic-1",
    available_cash: str = "10000000",
    reconciled_on: str | None = None,
    extra_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """owner 1 件(1 日分)の項目つきの記録。reconciled_on = 買付余力の最終照合日(鮮度の見本)。"""
    cands = [candidate(1)] if candidates is None else [deepcopy(dict(c)) for c in candidates]
    computed = outcome == "COMPUTED"
    run: dict[str, Any] = {
        "outcome": outcome,
        "status": status if computed else None,
        "unavailable_reason": unavailable_reason,
        "skip_reason": skip_reason,
        "policy_missing": [],
        "elapsed_ms": 5,
        "read_states": {},
    }
    run.update(dict(extra_run or {}))
    return {
        "schema_version": 1,
        "record_kind": "allocation_shadow_items",
        "business_date": business_date,
        "batch_id": BATCH,
        "owner_ref": ref(owner),
        "model": {
            "raer_model_version": model_version,
            "policy_id": "p-synthetic",
            "config_hash": config_hash,
            "contract_version": "c1",
            "shadow_mode": "SHADOW",
        },
        "run": run,
        "counts": {"incomplete_reasons": [], "exclusions": {}, "candidates_truncated": 0},
        "cash": {
            "available_cash": available_cash,
            "last_reconciled_at": f"{reconciled_on or business_date}T09:05:00+09:00",
            "updated_at": None,
        },
        "portfolio": {"portfolio_value": "50000000"},
        "candidates": cands if computed else [],
    }


def index_record(
    business_date: str,
    owners: Sequence[str],
    *,
    attempted: int | None = None,
    unrecorded: int = 0,
    reasons: Mapping[str, int] | None = None,
    schema_version: int = 1,
) -> dict[str, Any]:
    """索引。attempted の既定 = 載せた owner 数 + unrecorded(書込側と同じ整合)。"""
    refs = [ref(o) for o in owners]
    return {
        "schema_version": schema_version,
        "record_kind": "allocation_shadow_index",
        "business_date": business_date,
        "batch_id": BATCH,
        "shadow_mode": "SHADOW",
        "key_version": 1,
        "owner_refs": refs,
        "owner_refs_truncated": False,
        "attempted_owner_count": len(refs) + unrecorded if attempted is None else attempted,
        "recorded_owner_count": len(refs),
        "unrecorded_owner_count": unrecorded,
        "unrecorded_reasons": dict(reasons or {}),
    }


def day_skip_record(business_date: str, *, reason: str = "KEY_UNAVAILABLE") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_kind": "allocation_shadow_day_skip",
        "business_date": business_date,
        "batch_id": BATCH,
        "shadow_mode": "SHADOW",
        "reason": reason,
        "key_unavailable_reason": "ACCESS_DENIED",
        "unrecorded_owner_count": 2,
    }


# --- 1 日分(索引・skip・owner 記録)を読取元の形(audit_id → 記録)にする -----------------------


def day_records(
    business_date: str,
    owners: Mapping[str, dict[str, Any]],
    *,
    index: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """owners = {owner の識別子: owner 記録}。index を省けば owners 全員を載せた整合した索引。"""
    records: dict[str, dict[str, Any]] = {
        index_key(business_date): index
        if index is not None
        else index_record(business_date, list(owners))
    }
    for owner, record in owners.items():
        records[items_key(business_date, ref(owner))] = record
    return records


def skip_only_records(business_date: str) -> dict[str, dict[str, Any]]:
    """索引なし + 日単位の skip だけがある日。"""
    return {day_skip_key(business_date): day_skip_record(business_date)}
