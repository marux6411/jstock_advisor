"""replay の読取 facade(Issue #880 R1b PR-2。F-33。★ dormant ★)。

DynamoDB の client を **引数で受け**、読取の操作だけを通し、取得した項目の識別子を鍵つきの符号へ
置き換えて返す。★ どこからも import されない(CLI にも handler にも配線しない)★。AWS の client の
生成・boto3 / botocore の import・環境変数・時計は持たない。実データの読取(R2)は USER の別の承認で、
client の生成と配線はその段(R2-prep)が足す。設計: #880 rev3(…6214296197)・rev4・rev6(…6259587894)・
rev7(…6259775154)・サブちゃんの設計レビュー合格(…6259884051)。

## 守ること(契約テストで固定)

* **許可する操作は引数で受ける(既定なし)**。空・語彙外(読取の 4 語の部分集合でないもの)は構築時に
  拒否する。書込の操作の名前はこの module の語彙に存在しない(肯定リストに閉じる)
* **上限は全て必須の引数(既定なし)**: 件数・ページ数・時間・見積り × 倍率。
  超えたら固定の種別で停止する
* **部分結果を返さない**: 上限・失敗・形の崩れ・未処理(UnprocessedKeys)・ページ送りの異常は
  `ReplayReadFailure` を送出する(`None` や読めた分だけを返さない = 『0 件』と『読めなかった』の区別)
* **読む属性を宣言する**(projection は必須・空は拒否)。置換の対象は読む属性に含まれること
* **識別子は取得の直後に鍵つきの符号へ置き換える**。置換前の値は戻り値にも例外にも残さない。
  鍵は `bytes` で受ける(取得経路・鍵の名前は持たない。R2-prep が決める)
* 例外の文言に入れてよいのは、読取の 4 語の語彙とコードの識別子(属性名・種別名)まで。表名・キー・
  識別子・属性の値・例外の文面・ARN・鍵は入れない

## 限界(明記)

* proxy は『許可されない操作を誤って呼ぶ事故を防ぐ層』であり、実装の内部属性(`_client`)への
  直接アクセスは防げない。実際の境界は IAM の読取専用 role
* `max_seconds` は **総予算**で、各リクエストの『前』に経過時間と比べる。1 回の呼び出しが長引くと
  上限を超えて戻る(実効の上限 = `max_seconds` + 1 回の最悪時間)。1 回の最悪時間の桁は
  #833 の実測から 3.7 MB ÷ 2 MB/秒 ≒ 1.9 秒(平均 0.34 秒の約 5.5 倍)。read timeout をこれより短く
  すると、遅い日に毎回 timeout する。client の timeout・再試行の上限は読取計画が宣言する(R2-prep)
* `max_seconds` は変換(1 件ずつのモデル化)の時間を含まない。#833 の手元の CPU 84 秒(44%)の
  大部分はこの facade の外にある。R2-prep の総予算は、この facade の予算だけで立てない
* 鍵のメモリからの消去は保証しない(`bytes` は不変で零化できない)。本当の境界は『鍵を持つ role を
  限る』こと(`identifier_ref.KeyMaterial` と同じ限界)

## R2-prep への設計条件

* `ReplayReadFailure` を広く捕まえて続行してはならない。捕まえる場合は種別を人へ伝えて停止する
  (`except` の後に `continue` / `pass` を置かない)
* client は明示の timeout と再試行の上限をつけて作る。読取計画は総予算と 1 回の最悪時間の
  両方を宣言する
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, NoReturn

#: 読取の操作の語彙(肯定リスト)。許可する操作はこの集合の部分集合でなければならない
READ_OPERATIONS: Final[frozenset[str]] = frozenset({"get_item", "batch_get_item", "query", "scan"})

#: read_pages が受ける操作(ページ送りのあるもの)
_PAGED_OPERATIONS: Final[frozenset[str]] = frozenset({"scan", "query"})
#: request_extras に渡してよいキー(これ以外は拒否 = 読む範囲を広げる引数を通さない)
_ALLOWED_EXTRAS: Final[frozenset[str]] = frozenset(
    {
        "KeyConditionExpression",
        "FilterExpression",
        "ExpressionAttributeValues",
        "IndexName",
        "ConsistentRead",
        "Limit",
    }
)
_ACCESS_DENIED_CODES: Final[frozenset[str]] = frozenset(
    {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}
)
_CONNECTION_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "EndpointConnectionError",
        "ConnectionClosedError",
        "ConnectionError",
        "TimeoutError",
        "SSLError",
    }
)
_BATCH_GET_ITEM_LIMIT: Final = 100  # BatchGetItem が 1 回で受けるキーの数の仕様上の上限
_REF_PREFIX: Final = "rp1"
_DIGEST_HEX_LENGTH: Final = 24
MIN_KEY_BYTES: Final = 32
MIN_KEY_VERSION: Final = 1
MAX_KEY_VERSION: Final = 999


class ReplayReadConfigError(ValueError):
    """構成の誤り(呼び出し側の誤り)。メッセージは固定で、値・表名・鍵を含めない。"""


class ReplayReadViolationError(RuntimeError):
    """許可されていない操作を呼ぼうとした。要求された操作名(コードの語彙)だけを含む。"""


class ReplayReadFailureKind(StrEnum):
    ITEM_LIMIT = "ITEM_LIMIT"
    PAGE_LIMIT = "PAGE_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    ESTIMATE_EXCEEDED = "ESTIMATE_EXCEEDED"
    ACCESS_DENIED = "ACCESS_DENIED"  # 権限なし。『0 件』と区別する・迂回しない
    MALFORMED = "MALFORMED"
    INCOMPLETE = "INCOMPLETE"
    CONNECTION = "CONNECTION"
    ERROR = "ERROR"


class ReplayReadFailure(Exception):  # noqa: N818 - 固定の種別だけを持つ
    """読取の失敗。種別だけを持つ(値・表名・キー・識別子・例外の文面・ARN を持たない)。

    ★ 広く捕まえて続行してはならない(`except` の後に `continue` / `pass` を置かない)。捕まえる場合は
    種別を人へ伝えて停止する ★(R2-prep の設計条件)。
    """

    def __init__(self, kind: ReplayReadFailureKind) -> None:
        super().__init__(kind.value)
        self.kind = kind


def _config_error(problem: str) -> ReplayReadConfigError:
    return ReplayReadConfigError(problem)


def _failure(kind: ReplayReadFailureKind) -> ReplayReadFailure:
    return ReplayReadFailure(kind)


# --- proxy ------------------------------------------------------------------------------------


class ReplayReadClient:
    """許可した読取の操作だけを通す client の proxy。それ以外は例外。

    許可する操作は引数で受ける(既定なし)。空・読取の 4 語の部分集合でないものは構築時に拒否する。
    client の subclass ではない。`_client` への直接アクセスは防げない
    (実際の境界は IAM の読取専用 role)。
    """

    __slots__ = ("_allowed", "_client")

    def __init__(self, client: Any, *, allowed_operations: frozenset[str]) -> None:
        if not isinstance(allowed_operations, frozenset) or not allowed_operations:
            raise _config_error("allowed_operations must be a non-empty frozenset")
        if not all(isinstance(name, str) for name in allowed_operations) or not (
            allowed_operations <= READ_OPERATIONS
        ):
            raise _config_error("allowed_operations must be a subset of the read operations")
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_allowed", allowed_operations)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ は通常の探索で見つからなかったときだけ呼ばれる。許可した操作だけを通す。
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._allowed:
            raise ReplayReadViolationError(f"operation not allowed: {name}")
        return getattr(self._client, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ReplayReadClient is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ReplayReadClient is immutable")

    def __dir__(self) -> list[str]:
        return sorted(self._allowed)

    def __repr__(self) -> str:
        return f"ReplayReadClient(allowed_operations={sorted(self._allowed)})"


# --- 上限 -------------------------------------------------------------------------------------


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _config_error(f"{name} must be a positive int")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _config_error(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise _config_error(f"{name} must be a positive finite number")
    return value


@dataclass(frozen=True)
class ReadLimits:
    """読取の上限。★ 全て必須の引数(既定なし)★。値は読取計画が宣言する(コードに置かない)。

    ページ数・件数は『ちょうど = 通す / 超過 = 停止』。見積りの超過は
    `estimated_items × estimate_multiple` を超えたとき。
    """

    max_items: int
    max_pages: int
    max_seconds: float
    estimated_items: int
    estimate_multiple: float

    def __post_init__(self) -> None:
        _positive_int(self.max_items, "max_items")
        _positive_int(self.max_pages, "max_pages")
        _positive_number(self.max_seconds, "max_seconds")
        _positive_int(self.estimated_items, "estimated_items")
        _positive_number(self.estimate_multiple, "estimate_multiple")


# --- 読む属性・置換 ----------------------------------------------------------------------------


def _is_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and value.replace("_", "").isalnum()
    )


@dataclass(frozen=True)
class ReadSpec:
    """何を読むか。表名・読む属性(必須・空は拒否)・置換する属性(読む属性の部分集合)と用途ラベル。"""

    table_name: str
    projection: tuple[str, ...]
    mask_fields: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise _config_error("table_name must be a non-empty string")
        projection = self.projection
        if not isinstance(projection, tuple) or not projection:
            raise _config_error("projection must be a non-empty tuple")
        if not all(isinstance(name, str) and name for name in projection):
            raise _config_error("projection must contain non-empty strings")
        if len(set(projection)) != len(projection):
            raise _config_error("projection must not contain duplicates")
        if not isinstance(self.mask_fields, Mapping):
            raise _config_error("mask_fields must be a mapping")
        if not set(self.mask_fields) <= set(projection):
            raise _config_error("mask_fields must be a subset of projection")
        if not all(_is_label(kind) for kind in self.mask_fields.values()):
            raise _config_error("mask kind must be an ascii identifier")


class IdentifierMasker:
    """識別子の置換に使う鍵 + 鍵の版。★ 鍵の bytes は repr・str・pickle・copy・例外に出ない ★

    鍵を取り出す accessor は無い(`mask_identifier` だけが `_hmac` を使う)。
    `identifier_ref.KeyMaterial`
    と同じ防御(`__slots__`・不変・repr は版だけ・比較は `hmac.compare_digest`・hash は鍵を含めない・
    pickle / copy の拒否)を持つ。owner の符号とは接頭辞・用途ラベルが違い、相互に流用できない。

    ★ 保護の範囲 ★: 通常の表示・直列化・複製・例外の経路から鍵が出ないことまで。
    **内部属性への直接アクセス**(`object.__getattribute__`・`_hmac` の呼び出し・デバッガ・
    gc の走査)までは防止できない。鍵の本当の境界は『鍵を持つ role を限る』こと。
    メモリからの消去は保証しない。
    """

    __slots__ = ("_key", "_key_version")
    _key: bytes
    _key_version: int

    def __init__(self, *, key_version: int, key: bytes) -> None:
        if isinstance(key_version, bool) or not isinstance(key_version, int):
            raise _config_error("key_version must be an int")
        if not MIN_KEY_VERSION <= key_version <= MAX_KEY_VERSION:
            raise _config_error("key_version is out of range")
        if not isinstance(key, bytes):
            raise _config_error("key must be bytes")
        if len(key) < MIN_KEY_BYTES:
            raise _config_error("key is too short")
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_key_version", key_version)

    @property
    def key_version(self) -> int:
        return self._key_version

    def _hmac(self, message: bytes) -> str:
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("IdentifierMasker is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("IdentifierMasker is immutable")

    def __repr__(self) -> str:
        return f"IdentifierMasker(key_version={self._key_version})"

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdentifierMasker):
            return NotImplemented
        return self._key_version == other._key_version and hmac.compare_digest(
            self._key, other._key
        )

    def __hash__(self) -> int:
        return hash(("IdentifierMasker", self._key_version))

    def __reduce__(self) -> NoReturn:
        _refuse("pickled")

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        _refuse("pickled or copied")

    def __copy__(self) -> NoReturn:
        _refuse("copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        _refuse("copied")

    def __getstate__(self) -> NoReturn:
        _refuse("serialized")


def _refuse(what: str) -> NoReturn:
    raise TypeError(f"IdentifierMasker cannot be {what}")


def mask_identifier(masker: IdentifierMasker, kind: str, value: str) -> str:
    """識別子 → `rp1.{鍵の版}:` + HMAC-SHA256(鍵, 用途ラベル + 0x00 + 値)の先頭 24 桁。

    同じ鍵・版・kind・値なら同じ符号(突き合わせに必要)。kind・鍵・版が違えば別の符号。空・空白だけの
    値は拒否する(全ての空の入力が同じ符号になり区別できなくなるため)。
    """
    if not isinstance(masker, IdentifierMasker):
        raise _config_error("masker must be an IdentifierMasker")
    if not _is_label(kind):
        raise _config_error("mask kind must be an ascii identifier")
    if not isinstance(value, str) or not value.strip():
        raise _config_error("identifier must be a non-blank string")
    message = f"replay:{kind}".encode() + b"\x00" + value.encode("utf-8")
    digest = masker._hmac(message)[:_DIGEST_HEX_LENGTH]
    return f"{_REF_PREFIX}.{masker.key_version}:{digest}"


# --- 失敗の分類 -------------------------------------------------------------------------------


def classify_exception(exc: BaseException) -> ReplayReadFailureKind:
    """例外 → 失敗の種別。botocore を import せず、`response["Error"]["Code"]` を型を確かめて読む。

    形が崩れていても(response が無い・Mapping でない・"Error" が無い・"Code" が文字列でない)
    別の例外を出さず、ERROR に倒れる。例外の文面は読まない・捨てる。
    """
    try:
        response = getattr(exc, "response", None)
        if isinstance(response, Mapping):
            error = response.get("Error")
            if isinstance(error, Mapping):
                code = error.get("Code")
                if isinstance(code, str) and code in _ACCESS_DENIED_CODES:
                    return ReplayReadFailureKind.ACCESS_DENIED
        if type(exc).__name__ in _CONNECTION_TYPE_NAMES:
            return ReplayReadFailureKind.CONNECTION
    except Exception:  # noqa: BLE001 - 分類の途中の例外は ERROR に倒す(文面は読まない)
        return ReplayReadFailureKind.ERROR
    return ReplayReadFailureKind.ERROR


# --- reader -----------------------------------------------------------------------------------


class ReplaySourceReader:
    """読取の本体。client(`ReplayReadClient`)・鍵(`IdentifierMasker`)・経過時間の関数を引数で受ける。"""

    def __init__(
        self,
        client: ReplayReadClient,
        masker: IdentifierMasker,
        *,
        elapsed_seconds: Callable[[], float],
    ) -> None:
        if not isinstance(client, ReplayReadClient):
            raise _config_error("client must be a ReplayReadClient")
        if not isinstance(masker, IdentifierMasker):
            raise _config_error("masker must be an IdentifierMasker")
        if not callable(elapsed_seconds):
            raise _config_error("elapsed_seconds must be callable")
        self._client = client
        self._masker = masker
        self._elapsed_seconds = elapsed_seconds

    # -- 共通 --

    def _check_time(self, limits: ReadLimits) -> None:
        elapsed: object = None
        failed = False
        try:
            elapsed = self._elapsed_seconds()
        except Exception:  # noqa: BLE001 - 時計の関数の失敗は固定の種別へ(文面・原因は持たない)
            failed = True
        if failed:  # except の外で送出する(__context__ に元の例外を残さない)
            raise _failure(ReplayReadFailureKind.ERROR)
        if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
            raise _failure(ReplayReadFailureKind.ERROR)
        if elapsed > limits.max_seconds:
            raise _failure(ReplayReadFailureKind.TIME_LIMIT)

    def _call(self, operation: str, request: dict[str, Any]) -> Mapping[str, Any]:
        method = getattr(
            self._client, operation
        )  # 許可されない操作はここで ReplayReadViolationError
        kind: ReplayReadFailureKind | None = None
        response: object = None
        try:
            response = method(**request)
        except ReplayReadFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - 分類して固定の種別へ(文面・原因は持たない)
            kind = classify_exception(exc)
        if kind is not None:  # except の外で送出する(__context__ に元の例外を残さない)
            raise _failure(kind)
        if not isinstance(response, Mapping):
            raise _failure(ReplayReadFailureKind.MALFORMED)
        return response

    @staticmethod
    def _projection_request(spec: ReadSpec) -> dict[str, Any]:
        names = {f"#p{index}": name for index, name in enumerate(spec.projection)}
        return {
            "ProjectionExpression": ", ".join(names),
            "ExpressionAttributeNames": names,
        }

    def _mask_items(
        self,
        spec: ReadSpec,
        raw_items: object,
        seen: dict[str, str],
    ) -> list[dict[str, Any]]:
        if isinstance(raw_items, str | bytes) or not isinstance(raw_items, Sequence):
            raise _failure(ReplayReadFailureKind.MALFORMED)
        allowed = set(spec.projection)
        masked: list[dict[str, Any]] = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise _failure(ReplayReadFailureKind.MALFORMED)
            if not set(raw) <= allowed:
                raise _failure(ReplayReadFailureKind.MALFORMED)  # 宣言していない属性は返さない
            item = dict(raw)
            for attribute, kind in spec.mask_fields.items():
                typed = raw.get(attribute)
                if not isinstance(typed, Mapping) or set(typed) != {"S"}:
                    raise _failure(ReplayReadFailureKind.MALFORMED)
                text = typed["S"]
                if not isinstance(text, str) or not text.strip():
                    raise _failure(ReplayReadFailureKind.MALFORMED)
                ref = mask_identifier(self._masker, kind, text)
                previous = seen.setdefault(f"{kind}\x00{ref}", text)
                if previous != text:
                    raise _failure(ReplayReadFailureKind.MALFORMED)  # 衝突を黙って併合しない
                item[attribute] = {"S": ref}
            masked.append(item)
        return masked

    @staticmethod
    def _check_counts(count: int, limits: ReadLimits) -> None:
        if count > limits.max_items:
            raise _failure(ReplayReadFailureKind.ITEM_LIMIT)
        if count > limits.estimated_items * limits.estimate_multiple:
            raise _failure(ReplayReadFailureKind.ESTIMATE_EXCEEDED)

    # -- scan / query --

    def read_pages(
        self,
        operation: str,
        spec: ReadSpec,
        limits: ReadLimits,
        *,
        request_extras: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """scan / query を `LastEvaluatedKey` が尽きるまで読む。超過は例外(部分結果なし)。"""
        if operation not in _PAGED_OPERATIONS:
            raise _config_error("operation must be scan or query")
        if not isinstance(spec, ReadSpec) or not isinstance(limits, ReadLimits):
            raise _config_error("spec and limits are required")
        extras = dict(request_extras or {})
        if not set(extras) <= _ALLOWED_EXTRAS:
            raise _config_error("request_extras contains a key that is not allowed")
        base: dict[str, Any] = {
            "TableName": spec.table_name,
            **self._projection_request(spec),
            **extras,
        }
        items: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        start_key: Mapping[str, Any] | None = None
        pages = 0
        while True:
            self._check_time(limits)
            if pages >= limits.max_pages:
                raise _failure(ReplayReadFailureKind.PAGE_LIMIT)
            request = dict(base)
            if start_key is not None:
                request["ExclusiveStartKey"] = start_key
            response = self._call(operation, request)
            pages += 1
            items.extend(self._mask_items(spec, response.get("Items", []), seen))
            self._check_counts(len(items), limits)
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                return items
            if not isinstance(last_key, Mapping):
                raise _failure(ReplayReadFailureKind.MALFORMED)
            start_key = last_key

    # -- batch_get_item --

    def batch_get(
        self,
        spec: ReadSpec,
        keys: Sequence[Mapping[str, Any]],
        limits: ReadLimits,
    ) -> list[dict[str, Any]]:
        """キーの列を 100 件ずつ batch_get_item で読む。UnprocessedKeys が残ったら INCOMPLETE。"""
        if not isinstance(spec, ReadSpec) or not isinstance(limits, ReadLimits):
            raise _config_error("spec and limits are required")
        if isinstance(keys, str | bytes) or not isinstance(keys, Sequence):
            raise _config_error("keys must be a sequence")
        items: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        for requests, offset in enumerate(range(0, len(keys), _BATCH_GET_ITEM_LIMIT)):
            self._check_time(limits)
            if requests >= limits.max_pages:
                raise _failure(ReplayReadFailureKind.PAGE_LIMIT)
            chunk = [dict(key) for key in keys[offset : offset + _BATCH_GET_ITEM_LIMIT]]
            request = {
                "RequestItems": {spec.table_name: {"Keys": chunk, **self._projection_request(spec)}}
            }
            response = self._call("batch_get_item", request)
            if response.get("UnprocessedKeys"):
                raise _failure(ReplayReadFailureKind.INCOMPLETE)
            responses = response.get("Responses")
            if not isinstance(responses, Mapping):
                raise _failure(ReplayReadFailureKind.MALFORMED)
            items.extend(self._mask_items(spec, responses.get(spec.table_name, []), seen))
            self._check_counts(len(items), limits)
        return items
