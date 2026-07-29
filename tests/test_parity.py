"""Numerical and behavioural parity with mmh3 5.2.1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import mmh3
import numpy as np
import pytest

import mojo_mmh3 as mojo
from mojo_mmh3 import _lib


SEEDS = [0, 1, 42, 0x9747B28C, 0xFFFFFFFF]
LENGTHS = list(range(18)) + [31, 32, 33, 63, 64, 65, 255, 4096]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (b"", 0),
        (b"foo", -156908512),
        (b"hello", 613153351),
        (b"The quick brown fox jumps over the lazy dog", 776992547),
    ],
)
def test_published_x86_32_vectors(key, expected):
    assert mojo.hash(key) == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_x86_32_random_parity(seed):
    rng = np.random.default_rng(seed)
    for length in LENGTHS:
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
        assert mojo.mmh3_32_digest(data, seed) == mmh3.mmh3_32_digest(data, seed)
        assert mojo.mmh3_32_sintdigest(data, seed) == mmh3.mmh3_32_sintdigest(
            data, seed
        )
        assert mojo.mmh3_32_uintdigest(data, seed) == mmh3.mmh3_32_uintdigest(
            data, seed
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("architecture", ["x64", "x86"])
def test_128_random_parity(seed, architecture):
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    prefix = f"mmh3_{architecture}_128"
    for length in LENGTHS:
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
        for suffix in [
            "digest",
            "sintdigest",
            "uintdigest",
            "stupledigest",
            "utupledigest",
        ]:
            ours = getattr(mojo, f"{prefix}_{suffix}")
            upstream = getattr(mmh3, f"{prefix}_{suffix}")
            assert ours(data, seed) == upstream(data, seed)


@pytest.mark.parametrize("key", [b"", b"foo", b"a\x00b", "foo", "κόσμε"])
@pytest.mark.parametrize("seed", [0, 42, 0xFFFFFFFF])
def test_legacy_hash_parity(key, seed):
    assert mojo.hash(key, seed) == mmh3.hash(key, seed)
    assert mojo.hash(key, seed, signed=False) == mmh3.hash(
        key, seed, signed=False
    )
    assert mojo.hash_from_buffer(key, seed) == mmh3.hash_from_buffer(key, seed)


@pytest.mark.parametrize("x64arch", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_legacy_128_parity(x64arch, signed):
    for key in [b"", b"foo", b"\x00" * 17, "murmur"]:
        for seed in [0, 42, 0xFFFFFFFF]:
            assert mojo.hash64(
                key, seed, x64arch=x64arch, signed=signed
            ) == mmh3.hash64(
                key, seed, x64arch=x64arch, signed=signed
            )
            assert mojo.hash128(
                key, seed, x64arch=x64arch, signed=signed
            ) == mmh3.hash128(
                key, seed, x64arch=x64arch, signed=signed
            )
            assert mojo.hash_bytes(key, seed, x64arch=x64arch) == mmh3.hash_bytes(
                key, seed, x64arch=x64arch
            )


@pytest.mark.parametrize(
    "value",
    [
        b"buffer data",
        bytearray(b"buffer data"),
        memoryview(b"buffer data"),
        np.arange(32, dtype=np.int16),
    ],
)
def test_buffer_protocol_parity(value):
    seed = 123
    assert mojo.hash_from_buffer(value, seed) == mmh3.hash_from_buffer(value, seed)
    assert mojo.mmh3_32_digest(value, seed) == mmh3.mmh3_32_digest(value, seed)
    assert mojo.mmh3_x64_128_digest(value, seed) == mmh3.mmh3_x64_128_digest(
        value, seed
    )
    assert mojo.mmh3_x86_128_digest(value, seed) == mmh3.mmh3_x86_128_digest(
        value, seed
    )


@pytest.mark.parametrize(
    "value",
    [
        np.array(7, dtype=np.uint8),
        np.arange(12, dtype=np.float64),
        memoryview(b""),
    ],
)
def test_buffer_lengths_are_measured_in_bytes(value):
    assert mojo.mmh3_32_digest(value) == mmh3.mmh3_32_digest(value)
    assert mojo.mmh3_x64_128_digest(value) == mmh3.mmh3_x64_128_digest(value)
    assert mojo.mmh3_x86_128_digest(value) == mmh3.mmh3_x86_128_digest(value)


@pytest.mark.parametrize(
    "value",
    [
        memoryview(b"abcdef")[::2],
        np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2],
        np.asfortranarray(np.arange(12, dtype=np.uint8).reshape(3, 4)),
    ],
)
def test_non_c_contiguous_buffers_are_rejected(value):
    with pytest.raises(BufferError):
        mojo.mmh3_32_digest(value)


def test_raw_ffi_rejects_invalid_addresses_and_lengths():
    lib = _lib.library()
    result = _lib._ffi.new("uint64_t[2]")
    result_address = int(_lib._ffi.cast("uintptr_t", result))
    assert lib.mojo_mmh3_x86_32(0, 1, 0, result_address) != 0
    assert lib.mojo_mmh3_x64_128(0, -1, 0, result_address) != 0
    assert lib.mojo_mmh3_x86_128(result_address, 1, 0, 0) != 0


@pytest.mark.parametrize(
    ("ours_class", "upstream_class"),
    [
        (mojo.mmh3_32, mmh3.mmh3_32),
        (mojo.mmh3_x64_128, mmh3.mmh3_x64_128),
        (mojo.mmh3_x86_128, mmh3.mmh3_x86_128),
    ],
)
def test_incremental_hashers(ours_class, upstream_class):
    chunks = [b"a", b"bcdef", memoryview(b"ghijk"), bytearray(b"lmnop")]
    ours = ours_class(seed=42)
    upstream = upstream_class(seed=42)
    for chunk in chunks:
        assert ours.update(chunk) is None
        assert upstream.update(chunk) is None
        assert ours.digest() == upstream.digest()
    assert ours.name == upstream.name
    assert ours.digest_size == upstream.digest_size
    assert ours.block_size == upstream.block_size
    for method in ["digest", "sintdigest", "uintdigest"]:
        assert getattr(ours, method)() == getattr(upstream, method)()
    if ours.digest_size == 16:
        assert ours.stupledigest() == upstream.stupledigest()
        assert ours.utupledigest() == upstream.utupledigest()


@pytest.mark.parametrize(
    ("ours_class", "upstream_class"),
    [
        (mojo.mmh3_32, mmh3.mmh3_32),
        (mojo.mmh3_x64_128, mmh3.mmh3_x64_128),
        (mojo.mmh3_x86_128, mmh3.mmh3_x86_128),
    ],
)
def test_hasher_initial_data_and_copy(ours_class, upstream_class):
    ours = ours_class(b"prefix", 7)
    upstream = upstream_class(b"prefix", 7)
    ours_copy = ours.copy()
    upstream_copy = upstream.copy()
    ours.update(b"-left")
    upstream.update(b"-left")
    ours_copy.update(b"-right")
    upstream_copy.update(b"-right")
    assert ours.digest() == upstream.digest()
    assert ours_copy.digest() == upstream_copy.digest()


@pytest.mark.parametrize("seed", [-1, 1 << 32])
@pytest.mark.parametrize(
    "function",
    [
        mojo.hash,
        mojo.hash128,
        mojo.hash64,
        mojo.hash_bytes,
        mojo.mmh3_32_digest,
        mojo.mmh3_x64_128_digest,
        mojo.mmh3_x86_128_digest,
    ],
)
def test_seed_range_validation(function, seed):
    with pytest.raises(ValueError):
        function(b"data", seed)


@pytest.mark.parametrize("function", [mojo.hash, mojo.hash128, mojo.hash64])
def test_legacy_functions_reject_mutable_keys(function):
    with pytest.raises(TypeError):
        function(bytearray(b"data"))


@pytest.mark.parametrize(
    "function",
    [
        mojo.mmh3_32_digest,
        mojo.mmh3_x64_128_digest,
        mojo.mmh3_x86_128_digest,
    ],
)
def test_modern_buffer_functions_reject_strings(function):
    with pytest.raises(TypeError):
        function("not a buffer")


def test_large_input_parity():
    data = os.urandom(2_000_003)
    assert mojo.mmh3_32_digest(data, 99) == mmh3.mmh3_32_digest(data, 99)
    assert mojo.mmh3_x64_128_digest(data, 99) == mmh3.mmh3_x64_128_digest(
        data, 99
    )
    assert mojo.mmh3_x86_128_digest(data, 99) == mmh3.mmh3_x86_128_digest(
        data, 99
    )


@pytest.mark.parametrize("length", range(12, 37))
def test_x86_32_unaligned_simd_and_scalar_tails(length):
    backing = bytearray(range(length + 1))
    data = memoryview(backing)[1:]
    assert mojo.mmh3_32_digest(data, 42) == mmh3.mmh3_32_digest(data, 42)


def test_parallel_128_digest_calls_use_independent_results():
    inputs = [os.urandom(1024 + index) for index in range(32)]

    def digest(case):
        index, data = case
        if index % 2:
            return mojo.mmh3_x64_128_digest(data, index)
        return mojo.mmh3_x86_128_digest(data, index)

    def upstream_digest(case):
        index, data = case
        if index % 2:
            return mmh3.mmh3_x64_128_digest(data, index)
        return mmh3.mmh3_x86_128_digest(data, index)

    cases = list(enumerate(inputs))
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(digest, cases)) == list(
            executor.map(upstream_digest, cases)
        )
