"""CFFI bridge to the compiled Mojo kernels."""

from __future__ import annotations

import os
import struct
import subprocess
import threading
from pathlib import Path

from cffi import FFI

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("MOJO_MMH3_LIB", ROOT / "dist" / "libmojo-mmh3.so"))

_ffi = FFI()
_ffi.cdef(
    """
    typedef unsigned int uint32_t;
    typedef unsigned long long uint64_t;
    typedef long long int64_t;

    int64_t mojo_mmh3_x86_32(int64_t, int64_t, int64_t, int64_t);
    int64_t mojo_mmh3_x64_128(int64_t, int64_t, int64_t, int64_t);
    int64_t mojo_mmh3_x86_128(int64_t, int64_t, int64_t, int64_t);
    """
)
_PACK_U64 = struct.Struct("<QQ")
_PACK_U32 = struct.Struct("<IIII")
_library_lock = threading.Lock()
_thread_results = threading.local()


class BuildError(RuntimeError):
    """Raised when the shared library cannot be built."""


def build(force: bool = False) -> Path:
    source = ROOT / "src" / "mmh3.mojo"
    if (
        not force
        and LIB_PATH.exists()
        and (not source.exists() or LIB_PATH.stat().st_mtime >= source.stat().st_mtime)
    ):
        return LIB_PATH
    script = ROOT / "build" / "build.sh"
    if not script.exists():
        raise BuildError(
            f"{LIB_PATH} is missing and the build script is unavailable; "
            "set MOJO_MMH3_LIB to a compiled library"
        )
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not LIB_PATH.exists():
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB_PATH


_library = None


def library():
    global _library
    if _library is None:
        with _library_lock:
            if _library is None:
                _library = _ffi.dlopen(str(build()))
    return _library


def buffer_array(value: object, *, allow_str: bool) -> memoryview:
    if isinstance(value, str):
        if not allow_str:
            raise TypeError("object supporting the buffer API required")
        value = value.encode("utf-8")
    try:
        view = memoryview(value)
    except TypeError:
        raise TypeError("object supporting the buffer API required") from None
    if not view.c_contiguous:
        raise BufferError("memoryview: underlying buffer is not C-contiguous")
    return view.cast("B")


def _buffer(value: object, *, allow_str: bool):
    view = buffer_array(value, allow_str=allow_str)
    pointer = _ffi.from_buffer(view)
    address = int(_ffi.cast("uintptr_t", pointer))
    return view, pointer, address


def _result_buffer():
    try:
        return _thread_results.result128
    except AttributeError:
        result = _ffi.new("uint64_t[2]")
        cached = result, int(_ffi.cast("uintptr_t", result))
        _thread_results.result128 = cached
        return cached


def _result32_buffer():
    try:
        return _thread_results.result32
    except AttributeError:
        result = _ffi.new("uint32_t[1]")
        cached = result, int(_ffi.cast("uintptr_t", result))
        _thread_results.result32 = cached
        return cached


def _check_status(status: int) -> None:
    if status:
        raise RuntimeError("Mojo MurmurHash3 kernel rejected its buffer arguments")


def hash32(value: object, seed: int, *, allow_str: bool) -> int:
    view, pointer, address = _buffer(value, allow_str=allow_str)
    result, result_address = _result32_buffer()
    status = library().mojo_mmh3_x86_32(
        address, len(view), seed, result_address
    )
    _check_status(status)
    return result[0]


def digest128(value: object, seed: int, *, x64arch: bool, allow_str: bool) -> bytes:
    view, pointer, address = _buffer(value, allow_str=allow_str)
    result, result_address = _result_buffer()
    if x64arch:
        status = library().mojo_mmh3_x64_128(
            address, len(view), seed, result_address
        )
        _check_status(status)
        return _PACK_U64.pack(result[0], result[1])
    result32 = _ffi.cast("uint32_t *", result)
    status = library().mojo_mmh3_x86_128(
        address, len(view), seed, result_address
    )
    _check_status(status)
    return _PACK_U32.pack(result32[0], result32[1], result32[2], result32[3])
