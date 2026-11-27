"""Issue #278 (#63 F-1 O-d): json_store の id 欄が取れない要素で落ちるとき、原因をログに残す。

背景: `JsonCollectionStore._decode` は、要素を `model_validate` する前に id 欄(主キー)を取り出す。
要素が dict でない(文字列・数値・配列・null)、または id 欄が無い場合は、failure policy へ届く前に
`KeyError` / `TypeError` で落ちる。**落ちること自体は意図した設計**(id を鍵にできない要素は名指しで
復旧できないため救済しない)。問題は、落ちたときに「どの collection の何が悪いか」が
ログから分からないことだった。

固定するもの(O-d):
  ・例外の型・メッセージ・送出の条件は従来どおり(握りつぶさない・包み直さない)
  ・WARNING を 1 回だけ出す。出すのは collection 名・『id 欄が取れなかった』という事実・
    要素の型名のみ
  ・★ 要素の中身(値・key・id)はログに出さない ★(sentinel の漏れ走査)
  ・正常な要素・id が取れる要素のデコード失敗(failure policy の対象)では、この WARNING を出さない
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from jstock_advisor.infrastructure.local_repository import json_store as json_store_module
from jstock_advisor.infrastructure.local_repository.json_store import JsonCollectionStore
from jstock_advisor.infrastructure.record_failure_policy import RecordFailurePolicy

_SENTINEL = "sentinel-element-zz-278"
_LOGGER = json_store_module.logger.name
_MESSAGE_HEAD = "json_store: id 欄を取れない要素があります"


class _Item(BaseModel):
    item_id: str
    amount: int


def _store(
    tmp_path: Path, policy: RecordFailurePolicy = RecordFailurePolicy.STRICT
) -> JsonCollectionStore[_Item]:
    return JsonCollectionStore(_Item, "items.json", "item_id", tmp_path, failure_policy=policy)


def _write(tmp_path: Path, rows: list[object]) -> None:
    (tmp_path / "items.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == _LOGGER and r.getMessage().startswith(_MESSAGE_HEAD)
    ]


_NOT_A_DICT: list[object] = [_SENTINEL, 12345, [_SENTINEL], None, 3.5, True]


# --- 要素が dict でない ------------------------------------------


@pytest.mark.parametrize("element", _NOT_A_DICT)
@pytest.mark.parametrize("policy", list(RecordFailurePolicy))
def test_a_non_dict_element_still_raises_the_same_error_and_logs_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, element: object, policy: RecordFailurePolicy
) -> None:
    _write(tmp_path, [{"item_id": "good-1", "amount": 1}, element])
    store = _store(tmp_path, policy)

    with caplog.at_level(logging.WARNING, logger=_LOGGER), pytest.raises(TypeError) as caught:
        store._read_all()

    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "collection=items.json" in message
    assert "reason=element_not_a_dict" in message
    assert f"element_type={type(element).__name__}" in message
    assert records[0].levelno == logging.WARNING
    # 例外はそのまま(包み直さない): __cause__ も context も付けない
    assert caught.value.__cause__ is None
    assert _SENTINEL not in message


# --- id 欄が無い dict ------------------------------------------


@pytest.mark.parametrize("policy", list(RecordFailurePolicy))
def test_a_dict_without_the_id_field_still_raises_key_error_and_logs_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, policy: RecordFailurePolicy
) -> None:
    _write(tmp_path, [{"amount": 1, "memo": _SENTINEL}])
    store = _store(tmp_path, policy)

    with caplog.at_level(logging.WARNING, logger=_LOGGER), pytest.raises(KeyError) as caught:
        store._read_all()

    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "reason=id_field_missing" in message
    assert "element_type=dict" in message
    assert str(caught.value) == "'item_id'"  # 例外の文面は従来どおり


# --- ★ 値を出さない ----------------------------------------------


@pytest.mark.parametrize(
    "element",
    [_SENTINEL, [_SENTINEL], {"amount": 1, _SENTINEL: _SENTINEL}, {"other": _SENTINEL}],
)
def test_no_part_of_the_element_reaches_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, element: object
) -> None:
    _write(tmp_path, [element])

    with caplog.at_level(logging.DEBUG), pytest.raises((KeyError, TypeError)):
        _store(tmp_path)._read_all()

    everything = "\n".join(r.getMessage() + str(r.args) + str(r.exc_info) for r in caplog.records)
    assert _SENTINEL not in everything
    assert all(_SENTINEL not in str(value) for r in caplog.records for value in (r.args or ()))


# --- 影響を受けないもの ---------------------------------------------


def test_normal_elements_do_not_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(tmp_path, [{"item_id": "a", "amount": 1}, {"item_id": "b", "amount": 2}])

    with caplog.at_level(logging.DEBUG):
        loaded = _store(tmp_path)._read_all()

    assert sorted(loaded) == ["a", "b"]
    assert _warnings(caplog) == []


def test_a_decode_failure_with_a_readable_id_goes_to_the_failure_policy_not_this_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path, [{"item_id": "a", "amount": "not-a-number"}])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        loaded = _store(tmp_path, RecordFailurePolicy.LENIENT)._read_all()

    assert loaded == {}
    assert (
        _warnings(caplog) == []
    )  # id は取れた = 既存の failure policy の担当(この WARNING の対象外)


def test_an_empty_or_missing_file_does_not_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        assert _store(tmp_path)._read_all() == {}
        _write(tmp_path, [])
        assert _store(tmp_path)._read_all() == {}

    assert _warnings(caplog) == []
