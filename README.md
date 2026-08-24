# mojo-mmh3

`mojo-mmh3` is a standalone Mojo implementation of MurmurHash3 with a Python
API modeled on [`mmh3`](https://pypi.org/project/mmh3/). The hash loops run in
one compiled Mojo shared library; a small Python layer provides the familiar
return types, signedness controls, buffer handling, and hashlib-style objects.

MurmurHash3 is a fast non-cryptographic hash. It must not be used for passwords,
signatures, message authentication, or other cryptographic purposes.

## Coverage

The port implements all algorithms exposed by `mmh3 5.2.1`:

- MurmurHash3 x86 32-bit
- MurmurHash3 x64 128-bit
- MurmurHash3 x86 128-bit

The covered Python surface includes `hash`, `hash_from_buffer`, `hash64`,
`hash128`, `hash_bytes`, all 32-bit and 128-bit `digest`, `sintdigest`,
`uintdigest`, `stupledigest`, and `utupledigest` functions, plus `mmh3_32`,
`mmh3_x64_128`, and `mmh3_x86_128`.

The module is named `mojo_mmh3`, allowing it to coexist with upstream `mmh3`
for parity testing. The incremental classes have matching results, attributes,
copy behavior, and update semantics, but currently retain their input chunks in
Python and hash the accumulated buffer on each digest call. They are therefore
not a constant-memory replacement for upstream's incremental state objects.
This repository does not yet publish prebuilt wheels.

## Install

The checked-in Pixi environment pins the tested Mojo nightly and installs
upstream `mmh3 5.2.1` for tests and benchmarks:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-mmh3.so`.

## Usage

This example runs from the repository after the install and build above:

```bash
pixi run python - <<'PY'
import mojo_mmh3 as mmh3

print(mmh3.hash("foo"))
print(mmh3.hash128("foo"))

hasher = mmh3.mmh3_x64_128(b"hello ")
hasher.update(b"world")
print(hasher.digest().hex())
PY
```

It prints:

```text
-156908512
168394135621993849475852668931176482145
0e617feb46603f53b163eb607d4697ab
```

## Correctness

`pixi run test` compares every public function and hasher class against the
real `mmh3 5.2.1` extension. It exercises published vectors, randomized data,
every tail length around the 4-byte and 16-byte block boundaries, UTF-8
strings, multiple buffer-protocol types, extreme seeds, signed and unsigned
forms, incremental updates, copies, and a multi-megabyte input.

```bash
pixi run build && pixi run test
```

## Benchmarks

Run benchmarks only through `pixi run bench`; the task takes a machine-wide
lock. Times below are the median per call from a real run on an Intel Xeon
E5-2697 v4 at 2.30 GHz, Linux x86_64, Python 3.13.14:

| workload | mojo-mmh3 | mmh3 5.2.1 | relative |
|---|---:|---:|---:|
| x86_32, 64 B | 2.60 us | 193 ns | 13.43x slower |
| x86_32, 8 MiB | 3.78 ms | 3.89 ms | 1.03x faster |
| x64_128, 64 B | 2.86 us | 184 ns | 15.56x slower |
| x64_128, 8 MiB | 1.68 ms | 1.57 ms | 1.07x slower |
| x86_128, 64 B | 3.38 us | 203 ns | 16.63x slower |
| x86_128, 8 MiB | 3.04 ms | 2.42 ms | 1.25x slower |

Upstream is a mature native C extension. Tiny-input results are dominated by
the Python-to-CFFI call and result conversion; the compute-bound large-buffer
cases more closely measure the Mojo kernels.

## How it works

`src/mmh3.mojo` contains the three canonical MurmurHash3 variants in one
compilation unit. The x86_32 loop loads and mixes four words with SIMD before
applying the hash-state recurrence in order. The x64_128 loop loads two blocks
at a time with an unaligned SIMD load. Scalar block and byte-tail loops handle
every remainder. The kernels use Mojo's bit-rotate primitive so the compiler
emits native rotates instead of duplicated shift/multiply sequences. Unsigned
32-bit and 64-bit arithmetic provides the algorithm's required wraparound
behavior.

Python obtains a C-contiguous byte view without copying, keeps that view and its
CFFI pointer alive for the entire native call, and passes its address and byte
length across the C ABI as `Int` values. Each thread reuses its caller-owned
output storage across calls, avoiding a CFFI allocation per digest. The
exported Mojo functions reject negative lengths and null addresses before
reconstructing `UnsafePointer[..., AnyOrigin[mut=True]]` values. Empty
inputs use the non-null output allocation as a never-dereferenced dummy input
because Mojo pointers are non-nullable. Mojo performs no allocation and returns
a status code; Python raises if a kernel rejects its arguments, then constructs
upstream's bytes, integer, or tuple representation.

The kernels are not internally parallelized: every block updates state needed
by the next block, so splitting a single digest across threads changes the
algorithm and no correct size threshold can make that recurrence independent.
No GPU path is provided because these kernels are below roughly two arithmetic
operations per byte moved and remain serial across blocks; device transfer and
launch overhead cannot be amortized.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The project is MIT licensed.
