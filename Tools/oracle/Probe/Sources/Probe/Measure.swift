// Memory counters and tensor digests, measured the same way as the Python oracle.

import CryptoKit
import Darwin
import Foundation
import MLX

/// Resident size, physical footprint and its lifetime peak from proc_pid_rusage
/// (RUSAGE_INFO_V4), plus MLX's allocator counters. The oracle reads the same fields.
func processMemory() -> [String: Int] {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
        }
    }
    var out: [String: Int] = [
        "mlx_active_bytes": Memory.activeMemory,
        "mlx_cache_bytes": Memory.cacheMemory,
        "mlx_peak_bytes": Memory.peakMemory,
    ]
    if status == 0 {
        out["resident_bytes"] = Int(info.ri_resident_size)
        out["phys_footprint_bytes"] = Int(info.ri_phys_footprint)
        out["lifetime_max_phys_footprint_bytes"] = Int(info.ri_lifetime_max_phys_footprint)
    }
    return out
}

/// SHA-256 of the raw bfloat16 bytes in C order (little endian, as numpy writes them), and
/// float64 sums in C order. The oracle's `tensor_digest` computes the same fields.
func tensorDigest(_ array: MLXArray) -> TensorDigest {
    precondition(array.dtype == .bfloat16, "the cache digests are defined for bfloat16")
    let bits = array.view(dtype: .uint16).asArray(UInt16.self)
    let digest = bits.withUnsafeBufferPointer { SHA256.hash(data: UnsafeRawBufferPointer($0)) }
    let values = array.asType(.float32).asArray(Float.self)
    var sum = 0.0
    var squares = 0.0
    var largest = 0.0
    for value in values {
        let x = Double(value)
        sum += x
        squares += x * x
        largest = max(largest, abs(x))
    }
    return TensorDigest(
        dtype: "bfloat16", shape: array.shape,
        sha256: digest.map { String(format: "%02x", $0) }.joined(),
        sum: sum, sumOfSquares: squares, maxAbs: largest)
}

/// Monotonic seconds.
func now() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1e9
}
