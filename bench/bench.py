"""Benchmark mojo-mmh3 against upstream mmh3 on identical buffers."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time

import mmh3
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_mmh3 as mojo  # noqa: E402


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def measure(function, iterations: int, repeats: int = 7) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return statistics.median(samples)


def format_time(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} us"
    return f"{ns / 1_000_000:.2f} ms"


def main() -> None:
    rng = np.random.default_rng(2026)
    small = rng.integers(0, 256, 64, dtype=np.uint8)
    large = rng.integers(0, 256, 8 * 1024 * 1024, dtype=np.uint8)
    cases = [
        (
            "x86_32, 64 B",
            lambda: mojo.mmh3_32_uintdigest(small),
            lambda: mmh3.mmh3_32_uintdigest(small),
            20_000,
        ),
        (
            "x86_32, 8 MiB",
            lambda: mojo.mmh3_32_uintdigest(large),
            lambda: mmh3.mmh3_32_uintdigest(large),
            12,
        ),
        (
            "x64_128, 64 B",
            lambda: mojo.mmh3_x64_128_digest(small),
            lambda: mmh3.mmh3_x64_128_digest(small),
            20_000,
        ),
        (
            "x64_128, 8 MiB",
            lambda: mojo.mmh3_x64_128_digest(large),
            lambda: mmh3.mmh3_x64_128_digest(large),
            12,
        ),
        (
            "x86_128, 64 B",
            lambda: mojo.mmh3_x86_128_digest(small),
            lambda: mmh3.mmh3_x86_128_digest(small),
            20_000,
        ),
        (
            "x86_128, 8 MiB",
            lambda: mojo.mmh3_x86_128_digest(large),
            lambda: mmh3.mmh3_x86_128_digest(large),
            12,
        ),
    ]

    print(f"Machine: {cpu_name()}")
    print(f"System: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    print()
    print("| workload | mojo-mmh3 | mmh3 5.2.1 | relative |")
    print("|---|---:|---:|---:|")
    for name, ours, upstream, iterations in cases:
        if ours() != upstream():
            raise AssertionError(f"parity failure in benchmark case {name}")
        mojo_ns = measure(ours, iterations)
        upstream_ns = measure(upstream, iterations)
        if mojo_ns <= upstream_ns:
            comparison = f"{upstream_ns / mojo_ns:.2f}x faster"
        else:
            comparison = f"{mojo_ns / upstream_ns:.2f}x slower"
        print(
            f"| {name} | {format_time(mojo_ns)} | {format_time(upstream_ns)} "
            f"| {comparison} |"
        )


if __name__ == "__main__":
    main()
