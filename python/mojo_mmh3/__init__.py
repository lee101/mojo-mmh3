"""Python API for the Mojo MurmurHash3 kernels."""

from __future__ import annotations

import operator

from ._lib import buffer_array, digest128 as _digest128, hash32 as _hash32

__version__ = "0.1.0"

_U32_MAX = (1 << 32) - 1


def _seed(value: object) -> int:
    try:
        seed = operator.index(value)
    except TypeError:
        raise TypeError("seed must be an integer") from None
    if not 0 <= seed <= _U32_MAX:
        raise ValueError("seed is out of range")
    return seed


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _legacy_key(key: object) -> object:
    if not isinstance(key, (bytes, str)):
        raise TypeError("argument 1 must be read-only bytes-like object, not mutable")
    return key


def hash(key, seed=0, signed=True) -> int:
    """Return the MurmurHash3 x86 32-bit hash of bytes or a UTF-8 string."""
    value = _hash32(_legacy_key(key), _seed(seed), allow_str=True)
    return _signed(value, 32) if signed else value


def hash_from_buffer(key, seed=0, signed=True) -> int:
    """Return the MurmurHash3 x86 32-bit hash of a contiguous buffer."""
    value = _hash32(key, _seed(seed), allow_str=True)
    return _signed(value, 32) if signed else value


def hash_bytes(key, seed=0, x64arch=True) -> bytes:
    """Return a 16-byte MurmurHash3 digest."""
    return _digest128(
        _legacy_key(key), _seed(seed), x64arch=bool(x64arch), allow_str=True
    )


def hash128(key, seed=0, x64arch=True, signed=False) -> int:
    """Return a MurmurHash3 128-bit integer."""
    value = int.from_bytes(hash_bytes(key, seed, x64arch), "little")
    return _signed(value, 128) if signed else value


def hash64(key, seed=0, x64arch=True, signed=True) -> tuple[int, int]:
    """Return the 128-bit hash as two 64-bit integers."""
    digest = hash_bytes(key, seed, x64arch)
    values = (
        int.from_bytes(digest[:8], "little"),
        int.from_bytes(digest[8:], "little"),
    )
    if signed:
        return _signed(values[0], 64), _signed(values[1], 64)
    return values


def mmh3_32_digest(key, seed=0, /) -> bytes:
    value = _hash32(key, _seed(seed), allow_str=False)
    return value.to_bytes(4, "little")


def mmh3_32_sintdigest(key, seed=0, /) -> int:
    return _signed(_hash32(key, _seed(seed), allow_str=False), 32)


def mmh3_32_uintdigest(key, seed=0, /) -> int:
    return _hash32(key, _seed(seed), allow_str=False)


def mmh3_x64_128_digest(key, seed=0, /) -> bytes:
    return _digest128(key, _seed(seed), x64arch=True, allow_str=False)


def mmh3_x64_128_sintdigest(key, seed=0, /) -> int:
    return _signed(int.from_bytes(mmh3_x64_128_digest(key, seed), "little"), 128)


def mmh3_x64_128_uintdigest(key, seed=0, /) -> int:
    return int.from_bytes(mmh3_x64_128_digest(key, seed), "little")


def mmh3_x64_128_stupledigest(key, seed=0, /) -> tuple[int, int]:
    digest = mmh3_x64_128_digest(key, seed)
    return (
        _signed(int.from_bytes(digest[:8], "little"), 64),
        _signed(int.from_bytes(digest[8:], "little"), 64),
    )


def mmh3_x64_128_utupledigest(key, seed=0, /) -> tuple[int, int]:
    digest = mmh3_x64_128_digest(key, seed)
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")


def mmh3_x86_128_digest(key, seed=0, /) -> bytes:
    return _digest128(key, _seed(seed), x64arch=False, allow_str=False)


def mmh3_x86_128_sintdigest(key, seed=0, /) -> int:
    return _signed(int.from_bytes(mmh3_x86_128_digest(key, seed), "little"), 128)


def mmh3_x86_128_uintdigest(key, seed=0, /) -> int:
    return int.from_bytes(mmh3_x86_128_digest(key, seed), "little")


def mmh3_x86_128_stupledigest(key, seed=0, /) -> tuple[int, int]:
    digest = mmh3_x86_128_digest(key, seed)
    return (
        _signed(int.from_bytes(digest[:8], "little"), 64),
        _signed(int.from_bytes(digest[8:], "little"), 64),
    )


def mmh3_x86_128_utupledigest(key, seed=0, /) -> tuple[int, int]:
    digest = mmh3_x86_128_digest(key, seed)
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")


class _Hasher:
    _digest_function = staticmethod(mmh3_32_digest)

    def __init__(self, data=None, seed=0):
        self._seed = _seed(seed)
        self._data = bytearray()
        if data is not None:
            self.update(data)

    def update(self, object, /) -> None:
        self._data.extend(buffer_array(object, allow_str=False))

    def digest(self) -> bytes:
        return self._digest_function(self._data, self._seed)

    def copy(self):
        duplicate = object.__new__(type(self))
        duplicate._seed = self._seed
        duplicate._data = self._data.copy()
        return duplicate


class mmh3_32(_Hasher):
    """Incremental MurmurHash3 x86 32-bit hasher."""

    name = "mmh3_32"
    digest_size = 4
    block_size = 12
    _digest_function = staticmethod(mmh3_32_digest)

    def sintdigest(self) -> int:
        return _signed(self.uintdigest(), 32)

    def uintdigest(self) -> int:
        return int.from_bytes(self.digest(), "little")


class _Hasher128(_Hasher):
    digest_size = 16
    block_size = 32

    def sintdigest(self) -> int:
        return _signed(self.uintdigest(), 128)

    def uintdigest(self) -> int:
        return int.from_bytes(self.digest(), "little")

    def stupledigest(self) -> tuple[int, int]:
        a, b = self.utupledigest()
        return _signed(a, 64), _signed(b, 64)

    def utupledigest(self) -> tuple[int, int]:
        digest = self.digest()
        return (
            int.from_bytes(digest[:8], "little"),
            int.from_bytes(digest[8:], "little"),
        )


class mmh3_x64_128(_Hasher128):
    """Incremental MurmurHash3 x64 128-bit hasher."""

    name = "mmh3_x64_128"
    _digest_function = staticmethod(mmh3_x64_128_digest)


class mmh3_x86_128(_Hasher128):
    """Incremental MurmurHash3 x86 128-bit hasher."""

    name = "mmh3_x86_128"
    _digest_function = staticmethod(mmh3_x86_128_digest)


__all__ = [
    "hash",
    "hash128",
    "hash64",
    "hash_bytes",
    "hash_from_buffer",
    "mmh3_32",
    "mmh3_32_digest",
    "mmh3_32_sintdigest",
    "mmh3_32_uintdigest",
    "mmh3_x64_128",
    "mmh3_x64_128_digest",
    "mmh3_x64_128_sintdigest",
    "mmh3_x64_128_stupledigest",
    "mmh3_x64_128_uintdigest",
    "mmh3_x64_128_utupledigest",
    "mmh3_x86_128",
    "mmh3_x86_128_digest",
    "mmh3_x86_128_sintdigest",
    "mmh3_x86_128_stupledigest",
    "mmh3_x86_128_uintdigest",
    "mmh3_x86_128_utupledigest",
]
