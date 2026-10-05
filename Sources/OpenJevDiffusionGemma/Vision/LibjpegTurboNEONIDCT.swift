// A Swift translation of libjpeg-turbo 3.1.4.1's accurate integer inverse DCT for Arm Neon,
// `jsimd_idct_islow_neon` in simd/arm/jidctint-neon.c, which the libjpeg-turbo Pillow 12.3.0
// bundles runs on Apple silicon. Translated to Swift lane by lane, keeping its 16-bit
// dequantization and sums, its 32-bit products and its saturating output, by the OpenJevSwift
// contributors, 2026. This is an altered version of the original source.
//
// jidctint-neon.c:
//   Copyright (C) 2020, Arm Limited.  All Rights Reserved.
//   Copyright (C) 2020, 2024, D. R. Commander.  All Rights Reserved.
//
//   This software is provided 'as-is', without any express or implied
//   warranty.  In no event will the authors be held liable for any damages
//   arising from the use of this software.
//
//   Permission is granted to anyone to use this software for any purpose,
//   including commercial applications, and to alter it and redistribute it
//   freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//      claim that you wrote the original software. If you use this software
//      in a product, an acknowledgment in the product documentation would be
//      appreciated but is not required.
//   2. Altered source versions must be plainly marked as such, and must not be
//      misrepresented as being the original software.
//   3. This notice may not be removed or altered from any source distribution.
//
// It implements the algorithm of jidctint.c (`jpeg_idct_islow`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1998, Thomas G. Lane.
//   Modification developed 2002-2018 by Guido Vollbeding.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2015, 2020, 2022, 2026, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// The README.ijg and libjpeg-turbo's LICENSE.md, which covers the zlib-licensed SIMD code, are in
// ThirdPartyLicenses beside this file.

extension LibjpegTurboDecoder {
    /// `jsimd_idct_islow_neon`: dequantizes and inverse-transforms one block of coefficients in
    /// natural order, writing 8 rows of 8 samples `stride` apart.
    ///
    /// The Neon code is the C algorithm (jidctint.c) on 16-bit lanes. It agrees with the C code
    /// wherever no intermediate leaves 16 bits, which is every block of a well-formed JPEG; on
    /// corrupt data it differs, and this follows the Neon code, because that is what Pillow runs
    /// on the Macs upstream runs on:
    ///
    /// - The multipliers are the quantization values as 16-bit integers (`ISLOW_MULT_TYPE` is
    ///   `short` in a SIMD build), so a 16-bit table entry above 32,767 is negative, and each
    ///   dequantized coefficient keeps the low 16 bits of its product (`vmul_s16`).
    /// - The sums the C code forms before multiplying (`z2 + z3`, `tmp0 + tmp2`, ...) wrap at 16
    ///   bits, the products and accumulations at 32 bits.
    /// - The first pass keeps the low 16 bits of its rounded result (`vrshrn_n_s32`), the second
    ///   takes the high 16 bits of each 32-bit sum (`vaddhn_s32`) and then rounds and saturates
    ///   to a signed byte (`vqrshrn_n_s16`), where the C code wraps through its range-limit table.
    ///
    /// The Neon code's special cases (a half block whose rows 4 to 7 are zero, or whose AC
    /// coefficients are zero, or that is zero entirely) compute the same lanes as its general
    /// case, so this has only the general case and two shortcuts that give the same results: a
    /// column with no AC coefficient in the first pass, and a row with nothing past its first
    /// entry in the second.
    static func inverseDCT(
        _ coefficients: UnsafePointer<Int16>, _ multipliers: UnsafePointer<Int16>,
        _ workspace: UnsafeMutablePointer<Int16>, _ output: UnsafeMutablePointer<UInt8>,
        stride: Int
    ) {
        // The constants of jidctint-neon.c, scaled by 2^13.
        let fix0298: Int32 = 2446
        let fix0390: Int32 = 3196
        let fix0541: Int32 = 4433
        let fix0765: Int32 = 6270
        let fix0899: Int32 = 7373
        let fix1175: Int32 = 9633
        let fix1501: Int32 = 12299
        let fix1847: Int32 = 15137
        let fix1961: Int32 = 16069
        let fix2053: Int32 = 16819
        let fix2562: Int32 = 20995
        let fix3072: Int32 = 25172

        /// A value kept to its low 16 bits, as a 16-bit lane holds it.
        @inline(__always) func s16(_ x: Int32) -> Int32 { Int32(Int16(truncatingIfNeeded: x)) }

        // Pass 1: columns. Lane `column` of each of the 8 rows, dequantized to 16 bits.
        for column in 0..<8 {
            @inline(__always) func dequantized(_ row: Int) -> Int32 {
                s16(Int32(coefficients[row * 8 + column]) &* Int32(multipliers[row * 8 + column]))
            }
            var ac: Int16 = 0
            for row in 1..<8 { ac |= coefficients[row * 8 + column] }
            if ac == 0 {
                // vshl_n_s16(dq0, PASS1_BITS), what every path gives a column with no AC term.
                let value = Int16(truncatingIfNeeded: dequantized(0) &* 4)
                for row in 0..<8 { workspace[row * 8 + column] = value }
                continue
            }
            // Even part.
            var z2 = dequantized(2)
            var z3 = dequantized(6)
            let tmp2 = z2 &* fix0541 &+ z3 &* (fix0541 - fix1847)
            let tmp3 = z2 &* (fix0541 + fix0765) &+ z3 &* fix0541
            z2 = dequantized(0)
            z3 = dequantized(4)
            let tmp0 = s16(z2 &+ z3) << 13
            let tmp1 = s16(z2 &- z3) << 13
            let tmp10 = tmp0 &+ tmp3
            let tmp13 = tmp0 &- tmp3
            let tmp11 = tmp1 &+ tmp2
            let tmp12 = tmp1 &- tmp2
            // Odd part.
            let t0 = dequantized(7)
            let t1 = dequantized(5)
            let t2 = dequantized(3)
            let t3 = dequantized(1)
            let z3s = s16(t0 &+ t2)
            let z4s = s16(t1 &+ t3)
            let z3o = z3s &* (fix1175 - fix1961) &+ z4s &* fix1175
            let z4o = z3s &* fix1175 &+ z4s &* (fix1175 - fix0390)
            let o0 = t0 &* (fix0298 - fix0899) &- t3 &* fix0899 &+ z3o
            let o1 = t1 &* (fix2053 - fix2562) &- t2 &* fix2562 &+ z4o
            let o2 = t2 &* (fix3072 - fix2562) &- t1 &* fix2562 &+ z3o
            let o3 = t3 &* (fix1501 - fix0899) &- t0 &* fix0899 &+ z4o
            /// vrshrn_n_s32(x, CONST_BITS - PASS1_BITS): rounded, then the low 16 bits.
            @inline(__always) func narrow(_ x: Int32) -> Int16 {
                Int16(truncatingIfNeeded: (Int(x) + 1024) >> 11)
            }
            workspace[0 * 8 + column] = narrow(tmp10 &+ o3)
            workspace[1 * 8 + column] = narrow(tmp11 &+ o2)
            workspace[2 * 8 + column] = narrow(tmp12 &+ o1)
            workspace[3 * 8 + column] = narrow(tmp13 &+ o0)
            workspace[4 * 8 + column] = narrow(tmp13 &- o0)
            workspace[5 * 8 + column] = narrow(tmp12 &- o1)
            workspace[6 * 8 + column] = narrow(tmp11 &- o2)
            workspace[7 * 8 + column] = narrow(tmp10 &- o3)
        }

        /// vaddhn_s32 (or vsubhn_s32) then vqrshrn_n_s16(x, DESCALE_P2 - 16) and the shift to
        /// unsigned: the high half of the 32-bit sum, rounded, saturated to a signed byte, plus 128.
        @inline(__always) func sample(_ sum: Int32) -> UInt8 {
            let high = Int(sum >> 16)
            let value = (high + 2) >> 2
            return UInt8(
                truncatingIfNeeded: (value < -128 ? -128 : value > 127 ? 127 : value) + 128)
        }

        // Pass 2: rows.
        for row in 0..<8 {
            let w = workspace + row * 8
            let out = output + row * stride
            if w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7] == 0 {
                let value = sample(Int32(w[0]) << 13)
                for x in 0..<8 { out[x] = value }
                continue
            }
            // Even part.
            var z2 = Int32(w[2])
            var z3 = Int32(w[6])
            let tmp2 = z2 &* fix0541 &+ z3 &* (fix0541 - fix1847)
            let tmp3 = z2 &* (fix0541 + fix0765) &+ z3 &* fix0541
            z2 = Int32(w[0])
            z3 = Int32(w[4])
            let tmp0 = s16(z2 &+ z3) << 13
            let tmp1 = s16(z2 &- z3) << 13
            let tmp10 = tmp0 &+ tmp3
            let tmp13 = tmp0 &- tmp3
            let tmp11 = tmp1 &+ tmp2
            let tmp12 = tmp1 &- tmp2
            // Odd part.
            let t0 = Int32(w[7])
            let t1 = Int32(w[5])
            let t2 = Int32(w[3])
            let t3 = Int32(w[1])
            let z3s = s16(t0 &+ t2)
            let z4s = s16(t1 &+ t3)
            let z3o = z3s &* (fix1175 - fix1961) &+ z4s &* fix1175
            let z4o = z3s &* fix1175 &+ z4s &* (fix1175 - fix0390)
            let o0 = t0 &* (fix0298 - fix0899) &- t3 &* fix0899 &+ z3o
            let o1 = t1 &* (fix2053 - fix2562) &- t2 &* fix2562 &+ z4o
            let o2 = t2 &* (fix3072 - fix2562) &- t1 &* fix2562 &+ z3o
            let o3 = t3 &* (fix1501 - fix0899) &- t0 &* fix0899 &+ z4o
            out[0] = sample(tmp10 &+ o3)
            out[1] = sample(tmp11 &+ o2)
            out[2] = sample(tmp12 &+ o1)
            out[3] = sample(tmp13 &+ o0)
            out[4] = sample(tmp13 &- o0)
            out[5] = sample(tmp12 &- o1)
            out[6] = sample(tmp11 &- o2)
            out[7] = sample(tmp10 &- o3)
        }
    }
}
