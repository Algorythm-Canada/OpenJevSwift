import CoreML
import Darwin
import EncoderHarness
import Foundation

/// What each loaded function of a multifunction package costs on this Mac (D-042): it loads the
/// functions one at a time and keeps them all, runs each at its full shape twice, and records the
/// time of each call and the memory after it; then it releases them all and loads the first one
/// again. One configuration per process, so the figures belong to it.
///
///     encoder-capacity --package laya-m18-fp16 [--units cpuAndGPU] [--functions b1_s128,...]
///                      [--output result.json]
///
/// Without `--functions` it loads every function, the batch-1 ones first. The resident memory is
/// the physical footprint plus the resident pages of the files Core ML maps each loaded GPU
/// function's weights from (`payload-*.bin` in the temporary folder), which the footprint does
/// not count. Packages come from `OPENJEV_ENCODER_MODELS` or ~/Library/Caches/OpenJevSwift/encoders.
@main
struct EncoderCapacityCommand {
    struct Step: Codable {
        var function: String
        /// The first call: Core ML's load, then the first prediction.
        var firstCallSeconds: Double
        /// `MLModel(contentsOf:configuration:)` alone, part of the first call.
        var loadSeconds: Double
        var secondCallSeconds: Double
        var functionsLoaded: Int
        var footprintMB: Double
        var weightCopiesMB: Double
        var weightCopyRegions: Int
        var residentMB: Double
    }

    struct Result: Codable {
        var package: String
        var units: String
        var device: DeviceInfo
        var weightFileMB: Double?
        var baselineFootprintMB: Double
        var steps: [Step]
        /// The largest footprint sampled from the first call to the last, every 5 ms.
        var peakFootprintMB: Double
        /// The peak footprint plus the copies held at the end, which only grow while the
        /// functions load: an upper bound on the largest resident memory.
        var peakResidentMB: Double
        var afterReleaseFootprintMB: Double
        var afterReleaseWeightCopiesMB: Double
        /// The first function loaded again after every function was released: its first call.
        var reloadFunction: String
        var reloadFirstCallSeconds: Double
        var reloadLoadSeconds: Double
    }

    static func main() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        var options: [String: String] = [:]
        var arguments = CommandLine.arguments.dropFirst()
        while let flag = arguments.popFirst() {
            guard flag.hasPrefix("--"), let value = arguments.popFirst() else {
                fail("expected --flag value pairs, got \(flag)")
            }
            options[String(flag.dropFirst(2))] = value
        }
        guard let name = options["package"], let spec = PackageSpec.named(name),
            spec.kind == .multifunction
        else {
            let names = PackageSpec.all.filter { $0.kind == .multifunction }.map(\.name)
            fail("--package must be one of \(names.joined(separator: ", "))")
        }
        guard let units = ComputeUnitsName(rawValue: options["units"] ?? "cpuAndGPU") else {
            fail("--units must be one of \(ComputeUnitsName.allCases.map(\.rawValue))")
        }
        let every = spec.batches.flatMap { batch in
            spec.lengths.map { spec.functionName(batch: batch, length: $0) }
        }
        var functions = every
        if let list = options["functions"] {
            functions = list.split(separator: ",").map(String.init)
            guard !functions.isEmpty, Set(functions).count == functions.count,
                functions.allSatisfy({ shape(of: $0, in: spec) != nil })
            else {
                fail(
                    "--functions must name distinct functions of \(spec.name): "
                        + every.joined(separator: ", "))
            }
        }
        do {
            let result = try await measure(spec: spec, units: units, functions: functions)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(result)
            if let output = options["output"] {
                try data.write(to: URL(fileURLWithPath: output))
                print("wrote \(output)")
            } else {
                print(String(decoding: data, as: UTF8.self))
            }
        } catch {
            fail("\(spec.name) \(units.rawValue): \(error)")
        }
    }

    static func measure(spec: PackageSpec, units: ComputeUnitsName, functions: [String])
        async throws -> Result
    {
        let environment = ProcessInfo.processInfo.environment
        let models =
            environment["OPENJEV_ENCODER_MODELS"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Caches/OpenJevSwift/encoders")
        let package = models.appendingPathComponent("\(spec.name).mlpackage")
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw HarnessError.missing(package.path)
        }
        // The compile the backends keep beside the package, else a new one.
        let kept = models.appendingPathComponent("\(spec.name).mlmodelc")
        let compiled =
            FileManager.default.fileExists(atPath: kept.path)
            ? kept : try await MLModel.compileModel(at: package)
        let weightFile = compiled.appendingPathComponent("weights/weight.bin")
        let weightBytes = (try? FileManager.default.attributesOfItem(atPath: weightFile.path))?[
            .size] as? Int
        let baseline = physicalFootprint().current
        print("\(spec.name) \(units.rawValue): baseline \(megabytes(baseline)) MB")

        var model: EncoderModel? = try EncoderModel(
            spec: spec, compiled: compiled, units: units.units, padId: 50_283,
            capacity: functions.count)
        let sampler = FootprintSampler()
        sampler.start()
        var steps: [Step] = []
        for function in functions {
            guard let (batch, length) = shape(of: function, in: spec), let loaded = model else {
                continue
            }
            let input = rows(batch: batch, length: length, planes: spec.planes)
            var started = now()
            _ = try loaded.run(input)
            let first = now() - started
            started = now()
            _ = try loaded.run(input)
            let second = now() - started
            let footprint = physicalFootprint().current
            let copies = weightCopies()
            let step = Step(
                function: function, firstCallSeconds: first,
                loadSeconds: loaded.functionLoadSeconds[function] ?? 0, secondCallSeconds: second,
                functionsLoaded: steps.count + 1, footprintMB: megabytes(footprint),
                weightCopiesMB: megabytes(copies.bytes), weightCopyRegions: copies.regions,
                residentMB: megabytes(footprint + copies.bytes))
            steps.append(step)
            print(
                "  \(function): first call \(seconds(first)) s (load \(seconds(step.loadSeconds)) s), "
                    + "second \(seconds(second)) s; \(step.functionsLoaded) loaded: footprint "
                    + "\(step.footprintMB) MB + weight copies \(step.weightCopiesMB) MB = "
                    + "\(step.residentMB) MB")
        }
        let peak = sampler.stop()
        let held = weightCopies().bytes
        model = nil
        try await Task.sleep(for: .milliseconds(500))
        let releasedFootprint = physicalFootprint().current
        let releasedCopies = weightCopies().bytes
        print(
            "  released: footprint \(megabytes(releasedFootprint)) MB + weight copies "
                + "\(megabytes(releasedCopies)) MB")

        let first = functions[0]
        guard let (batch, length) = shape(of: first, in: spec) else {
            throw HarnessError.shape(first)
        }
        let again = try EncoderModel(
            spec: spec, compiled: compiled, units: units.units, padId: 50_283, capacity: 1)
        let started = now()
        _ = try again.run(rows(batch: batch, length: length, planes: spec.planes))
        let reload = now() - started
        print(
            "  \(first) loaded again: first call \(seconds(reload)) s (load "
                + "\(seconds(again.functionLoadSeconds[first] ?? 0)) s)")
        return Result(
            package: spec.name, units: units.rawValue, device: .current(),
            weightFileMB: weightBytes.map { megabytes(UInt64($0)) },
            baselineFootprintMB: megabytes(baseline), steps: steps,
            peakFootprintMB: megabytes(peak), peakResidentMB: megabytes(peak + held),
            afterReleaseFootprintMB: megabytes(releasedFootprint),
            afterReleaseWeightCopiesMB: megabytes(releasedCopies), reloadFunction: first,
            reloadFirstCallSeconds: reload,
            reloadLoadSeconds: again.functionLoadSeconds[first] ?? 0)
    }

    /// `b{batch}_s{length}` as a shape of the package, or `nil`.
    static func shape(of function: String, in spec: PackageSpec) -> (Int, Int)? {
        let parts = function.dropFirst().split(separator: "_s")
        guard function.hasPrefix("b"), parts.count == 2, let batch = Int(parts[0]),
            let length = Int(parts[1]), spec.batches.contains(batch), spec.lengths.contains(length)
        else { return nil }
        return (batch, length)
    }

    /// `batch` rows of `length` tokens: [CLS], spread ids, [SEP]; the mask; Laya's question type 0.
    static func rows(batch: Int, length: Int, planes: Int) -> [[[Int32]]] {
        (0..<batch).map { row in
            var ids = (0..<length).map { Int32(100 + ($0 * 7919 + row * 131) % 50_000) }
            ids[0] = 50_281
            ids[length - 1] = 50_282
            let rest = (1..<planes).map { [Int32](repeating: $0 == 1 ? 1 : 0, count: length) }
            return [ids] + rest
        }
    }

    /// The resident bytes of the regions mapped from Core ML's `payload-*.bin` files, and how
    /// many regions there are: what vmmap lists as `mapped file` with those paths.
    static func weightCopies() -> (bytes: UInt64, regions: Int) {
        var address: UInt64 = 0
        var bytes: UInt64 = 0
        var regions = 0
        let pageSize = UInt64(getpagesize())
        let size = Int32(MemoryLayout<proc_regionwithpathinfo>.size)
        while true {
            // The region that holds `address`, or the next one, with its file's path if it has one.
            var info = proc_regionwithpathinfo()
            guard proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, address, &info, size) == size else {
                break
            }
            let region = info.prp_prinfo
            let path = withUnsafeBytes(of: info.prp_vip.vip_path) { raw in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            if path.contains("/payload-") {
                bytes += UInt64(region.pri_pages_resident) * pageSize
                regions += 1
            }
            address = region.pri_address + region.pri_size
        }
        return (bytes, regions)
    }

    static func megabytes(_ bytes: UInt64) -> Double {
        (Double(bytes) / 1_048_576 * 10).rounded() / 10
    }

    static func seconds(_ value: Double) -> String { String(format: "%.3f", value) }

    static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("encoder-capacity: \(message)\n".utf8))
        exit(1)
    }
}
