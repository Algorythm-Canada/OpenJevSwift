import Darwin
import Foundation
import os

/// Wall-clock time in seconds from a monotonic clock.
public func now() -> Double {
    Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9
}

/// Median, 95th percentile (nearest rank) and range of a set of durations, in milliseconds.
public struct LatencyStats: Codable, Sendable {
    public let count: Int
    public let medianMs: Double
    public let p95Ms: Double
    public let meanMs: Double
    public let minMs: Double
    public let maxMs: Double

    public init(seconds: [Double]) {
        let ms = seconds.map { $0 * 1000 }.sorted()
        count = ms.count
        guard !ms.isEmpty else {
            medianMs = 0
            p95Ms = 0
            meanMs = 0
            minMs = 0
            maxMs = 0
            return
        }
        let mid = ms.count / 2
        medianMs = ms.count % 2 == 1 ? ms[mid] : (ms[mid - 1] + ms[mid]) / 2
        p95Ms = ms[max(0, Int((0.95 * Double(ms.count)).rounded(.up)) - 1)]
        meanMs = ms.reduce(0, +) / Double(ms.count)
        minMs = ms[0]
        maxMs = ms[ms.count - 1]
    }
}

/// The process's physical footprint, the figure iOS compares with its memory limit, and the
/// largest it has been since the process started.
public func physicalFootprint() -> (current: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { return (0, 0) }
    return (info.phys_footprint, UInt64(max(0, info.ledger_phys_footprint_peak)))
}

/// Samples the physical footprint on a background thread and keeps the largest value, so a
/// configuration's peak can be told apart from the process's lifetime peak.
public final class FootprintSampler: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: (peak: UInt64(0), running: false))
    private var thread: Thread?

    public init() {}

    public func start(interval: TimeInterval = 0.005) {
        lock.withLock { $0 = (physicalFootprint().current, true) }
        let thread = Thread { [self] in
            while lock.withLock({ $0.running }) {
                let current = physicalFootprint().current
                lock.withLock { $0.peak = max($0.peak, current) }
                Thread.sleep(forTimeInterval: interval)
            }
        }
        thread.qualityOfService = .utility
        self.thread = thread
        thread.start()
    }

    /// Stops sampling and returns the largest footprint seen, in bytes.
    public func stop() -> UInt64 {
        let current = physicalFootprint().current
        return lock.withLock {
            $0.running = false
            $0.peak = max($0.peak, current)
            return $0.peak
        }
    }
}

/// What the harness ran on.
public struct DeviceInfo: Codable, Sendable {
    public let machine: String
    public let model: String
    public let chip: String
    public let operatingSystem: String
    public let processorCount: Int
    public let physicalMemoryBytes: UInt64

    public static func current() -> DeviceInfo {
        DeviceInfo(
            machine: sysctlString("hw.machine") ?? "unknown",
            model: sysctlString("hw.model") ?? "unknown",
            chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            processorCount: ProcessInfo.processInfo.processorCount,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(
            decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

public func thermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
