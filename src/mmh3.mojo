"""MurmurHash3 x86_32, x64_128, and x86_128 kernels."""

from std.bit import rotate_bits_left
from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]


@always_inline
def rotl32[r: Int](x: UInt32) -> UInt32:
    return rotate_bits_left[r](x)


@always_inline
def rotl64[r: Int](x: UInt64) -> UInt64:
    return rotate_bits_left[r](x)


@always_inline
def load32(data: BPtr, offset: Int) -> UInt32:
    return (
        UInt32(data[offset])
        | (UInt32(data[offset + 1]) << 8)
        | (UInt32(data[offset + 2]) << 16)
        | (UInt32(data[offset + 3]) << 24)
    )


@always_inline
def load64(data: BPtr, offset: Int) -> UInt64:
    return UInt64(load32(data, offset)) | (
        UInt64(load32(data, offset + 4)) << 32
    )


@always_inline
def fmix32(value: UInt32) -> UInt32:
    var h = value
    h = h ^ (h >> 16)
    h = h * UInt32(0x85EBCA6B)
    h = h ^ (h >> 13)
    h = h * UInt32(0xC2B2AE35)
    h = h ^ (h >> 16)
    return h


@always_inline
def fmix64(value: UInt64) -> UInt64:
    var k = value
    k = k ^ (k >> 33)
    k = k * UInt64(0xFF51AFD7ED558CCD)
    k = k ^ (k >> 33)
    k = k * UInt64(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> 33)
    return k


def x86_32(data: BPtr, n: Int, seed: UInt32) -> UInt32:
    var h1 = seed
    comptime c1 = UInt32(0xCC9E2D51)
    comptime c2 = UInt32(0x1B873593)
    comptime W = simd_width_of[DType.float64]()
    comptime vc1 = SIMD[DType.uint32, W](c1)
    comptime vc2 = SIMD[DType.uint32, W](c2)

    var i = 0
    while i + W * 4 <= n:
        var keys = (data + i).bitcast[UInt32]().load[width=W, alignment=1]()
        keys = rotate_bits_left[15](keys * vc1) * vc2
        comptime for lane in range(W):
            h1 = h1 ^ keys[lane]
            h1 = rotl32[13](h1)
            h1 = h1 * UInt32(5) + UInt32(0xE6546B64)
        i += W * 4

    while i + 4 <= n:
        var k1 = load32(data, i)
        k1 = k1 * c1
        k1 = rotl32[15](k1)
        k1 = k1 * c2
        h1 = h1 ^ k1
        h1 = rotl32[13](h1)
        h1 = h1 * UInt32(5) + UInt32(0xE6546B64)
        i += 4

    var k1 = UInt32(0)
    var rem = n - i
    if rem >= 3:
        k1 = k1 ^ (UInt32(data[i + 2]) << 16)
    if rem >= 2:
        k1 = k1 ^ (UInt32(data[i + 1]) << 8)
    if rem >= 1:
        k1 = k1 ^ UInt32(data[i])
        k1 = k1 * c1
        k1 = rotl32[15](k1)
        k1 = k1 * c2
        h1 = h1 ^ k1

    h1 = h1 ^ UInt32(n)
    return fmix32(h1)


def x64_128(data: BPtr, n: Int, seed: UInt32, result: U64Ptr):
    var h1 = UInt64(seed)
    var h2 = UInt64(seed)
    comptime c1 = UInt64(0x87C37B91114253D5)
    comptime c2 = UInt64(0x4CF5AD432745937F)

    var i = 0
    while i + 16 <= n:
        var k1 = load64(data, i)
        var k2 = load64(data, i + 8)

        k1 = k1 * c1
        k1 = rotl64[31](k1)
        k1 = k1 * c2
        h1 = h1 ^ k1
        h1 = rotl64[27](h1)
        h1 = h1 + h2
        h1 = h1 * UInt64(5) + UInt64(0x52DCE729)

        k2 = k2 * c2
        k2 = rotl64[33](k2)
        k2 = k2 * c1
        h2 = h2 ^ k2
        h2 = rotl64[31](h2)
        h2 = h2 + h1
        h2 = h2 * UInt64(5) + UInt64(0x38495AB5)
        i += 16

    var k1 = UInt64(0)
    var k2 = UInt64(0)
    var rem = n - i
    if rem >= 15:
        k2 = k2 ^ (UInt64(data[i + 14]) << 48)
    if rem >= 14:
        k2 = k2 ^ (UInt64(data[i + 13]) << 40)
    if rem >= 13:
        k2 = k2 ^ (UInt64(data[i + 12]) << 32)
    if rem >= 12:
        k2 = k2 ^ (UInt64(data[i + 11]) << 24)
    if rem >= 11:
        k2 = k2 ^ (UInt64(data[i + 10]) << 16)
    if rem >= 10:
        k2 = k2 ^ (UInt64(data[i + 9]) << 8)
    if rem >= 9:
        k2 = k2 ^ UInt64(data[i + 8])
        k2 = k2 * c2
        k2 = rotl64[33](k2)
        k2 = k2 * c1
        h2 = h2 ^ k2
    if rem >= 8:
        k1 = k1 ^ (UInt64(data[i + 7]) << 56)
    if rem >= 7:
        k1 = k1 ^ (UInt64(data[i + 6]) << 48)
    if rem >= 6:
        k1 = k1 ^ (UInt64(data[i + 5]) << 40)
    if rem >= 5:
        k1 = k1 ^ (UInt64(data[i + 4]) << 32)
    if rem >= 4:
        k1 = k1 ^ (UInt64(data[i + 3]) << 24)
    if rem >= 3:
        k1 = k1 ^ (UInt64(data[i + 2]) << 16)
    if rem >= 2:
        k1 = k1 ^ (UInt64(data[i + 1]) << 8)
    if rem >= 1:
        k1 = k1 ^ UInt64(data[i])
        k1 = k1 * c1
        k1 = rotl64[31](k1)
        k1 = k1 * c2
        h1 = h1 ^ k1

    h1 = h1 ^ UInt64(n)
    h2 = h2 ^ UInt64(n)
    h1 = h1 + h2
    h2 = h2 + h1
    h1 = fmix64(h1)
    h2 = fmix64(h2)
    h1 = h1 + h2
    h2 = h2 + h1
    result[0] = h1
    result[1] = h2


def x86_128(data: BPtr, n: Int, seed: UInt32, result: U32Ptr):
    var h1 = seed
    var h2 = seed
    var h3 = seed
    var h4 = seed
    comptime c1 = UInt32(0x239B961B)
    comptime c2 = UInt32(0xAB0E9789)
    comptime c3 = UInt32(0x38B34AE5)
    comptime c4 = UInt32(0xA1E38B93)

    var i = 0
    while i + 16 <= n:
        var k1 = load32(data, i)
        var k2 = load32(data, i + 4)
        var k3 = load32(data, i + 8)
        var k4 = load32(data, i + 12)

        k1 = rotl32[15](k1 * c1) * c2
        h1 = h1 ^ k1
        h1 = (rotl32[19](h1) + h2) * UInt32(5) + UInt32(0x561CCD1B)

        k2 = rotl32[16](k2 * c2) * c3
        h2 = h2 ^ k2
        h2 = (rotl32[17](h2) + h3) * UInt32(5) + UInt32(0x0BCAA747)

        k3 = rotl32[17](k3 * c3) * c4
        h3 = h3 ^ k3
        h3 = (rotl32[15](h3) + h4) * UInt32(5) + UInt32(0x96CD1C35)

        k4 = rotl32[18](k4 * c4) * c1
        h4 = h4 ^ k4
        h4 = (rotl32[13](h4) + h1) * UInt32(5) + UInt32(0x32AC3B17)
        i += 16

    var k1 = UInt32(0)
    var k2 = UInt32(0)
    var k3 = UInt32(0)
    var k4 = UInt32(0)
    var rem = n - i
    if rem >= 15:
        k4 = k4 ^ (UInt32(data[i + 14]) << 16)
    if rem >= 14:
        k4 = k4 ^ (UInt32(data[i + 13]) << 8)
    if rem >= 13:
        k4 = k4 ^ UInt32(data[i + 12])
        k4 = rotl32[18](k4 * c4) * c1
        h4 = h4 ^ k4
    if rem >= 12:
        k3 = k3 ^ (UInt32(data[i + 11]) << 24)
    if rem >= 11:
        k3 = k3 ^ (UInt32(data[i + 10]) << 16)
    if rem >= 10:
        k3 = k3 ^ (UInt32(data[i + 9]) << 8)
    if rem >= 9:
        k3 = k3 ^ UInt32(data[i + 8])
        k3 = rotl32[17](k3 * c3) * c4
        h3 = h3 ^ k3
    if rem >= 8:
        k2 = k2 ^ (UInt32(data[i + 7]) << 24)
    if rem >= 7:
        k2 = k2 ^ (UInt32(data[i + 6]) << 16)
    if rem >= 6:
        k2 = k2 ^ (UInt32(data[i + 5]) << 8)
    if rem >= 5:
        k2 = k2 ^ UInt32(data[i + 4])
        k2 = rotl32[16](k2 * c2) * c3
        h2 = h2 ^ k2
    if rem >= 4:
        k1 = k1 ^ (UInt32(data[i + 3]) << 24)
    if rem >= 3:
        k1 = k1 ^ (UInt32(data[i + 2]) << 16)
    if rem >= 2:
        k1 = k1 ^ (UInt32(data[i + 1]) << 8)
    if rem >= 1:
        k1 = k1 ^ UInt32(data[i])
        k1 = rotl32[15](k1 * c1) * c2
        h1 = h1 ^ k1

    h1 = h1 ^ UInt32(n)
    h2 = h2 ^ UInt32(n)
    h3 = h3 ^ UInt32(n)
    h4 = h4 ^ UInt32(n)
    h1 = h1 + h2 + h3 + h4
    h2 = h2 + h1
    h3 = h3 + h1
    h4 = h4 + h1
    h1 = fmix32(h1)
    h2 = fmix32(h2)
    h3 = fmix32(h3)
    h4 = fmix32(h4)
    h1 = h1 + h2 + h3 + h4
    h2 = h2 + h1
    h3 = h3 + h1
    h4 = h4 + h1
    result[0] = h1
    result[1] = h2
    result[2] = h3
    result[3] = h4


@export("mojo_mmh3_x86_32")
def mojo_mmh3_x86_32(
    data_addr: Int, n: Int, seed: Int, result_addr: Int
) abi("C") -> Int:
    if n < 0 or result_addr == 0 or (n > 0 and data_addr == 0):
        return 1
    var result = U32Ptr(unsafe_from_address=result_addr)
    if n == 0:
        # UnsafePointer is non-nullable. The result allocation is a safe dummy
        # input for an empty hash; x86_32 never dereferences it when n is zero.
        result[0] = x86_32(result.bitcast[UInt8](), n, UInt32(seed))
    else:
        result[0] = x86_32(
            BPtr(unsafe_from_address=data_addr), n, UInt32(seed)
        )
    return 0


@export("mojo_mmh3_x64_128")
def mojo_mmh3_x64_128(
    data_addr: Int, n: Int, seed: Int, result_addr: Int
) abi("C") -> Int:
    if n < 0 or result_addr == 0 or (n > 0 and data_addr == 0):
        return 1
    var result = U64Ptr(unsafe_from_address=result_addr)
    if n == 0:
        x64_128(result.bitcast[UInt8](), n, UInt32(seed), result)
    else:
        x64_128(
            BPtr(unsafe_from_address=data_addr), n, UInt32(seed), result
        )
    return 0


@export("mojo_mmh3_x86_128")
def mojo_mmh3_x86_128(
    data_addr: Int, n: Int, seed: Int, result_addr: Int
) abi("C") -> Int:
    if n < 0 or result_addr == 0 or (n > 0 and data_addr == 0):
        return 1
    var result = U32Ptr(unsafe_from_address=result_addr)
    if n == 0:
        x86_128(result.bitcast[UInt8](), n, UInt32(seed), result)
    else:
        x86_128(
            BPtr(unsafe_from_address=data_addr), n, UInt32(seed), result
        )
    return 0
