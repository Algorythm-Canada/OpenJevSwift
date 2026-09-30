import EncoderHarness
import Foundation

/// Measures one converted package with one set of compute units on this Mac and writes the result
/// as JSON. Tools/encoders/run_macos.sh runs it once per configuration, each in its own process,
/// so that the footprint figures belong to one configuration and a Core ML crash loses only that
/// configuration.
///
///     encoder-harness --package verdict-m18-fp16 --units all --output result.json [--root <repo>]
@main
struct EncoderHarnessCommand {
    static func main() async {
        // Line-buffered, so that the log of a run that crashes inside Core ML keeps its last lines.
        setvbuf(stdout, nil, _IOLBF, 0)
        var options: [String: String] = [:]
        var arguments = CommandLine.arguments.dropFirst()
        while let flag = arguments.popFirst() {
            guard flag.hasPrefix("--"), let value = arguments.popFirst() else {
                fail("expected --flag value pairs, got \(flag)")
            }
            options[String(flag.dropFirst(2))] = value
        }
        guard let name = options["package"], let spec = PackageSpec.named(name) else {
            fail("--package must be one of \(PackageSpec.all.map(\.name).joined(separator: ", "))")
        }
        guard let unitsName = options["units"], let units = ComputeUnitsName(rawValue: unitsName)
        else {
            fail(
                "--units must be one of \(ComputeUnitsName.allCases.map(\.rawValue).joined(separator: ", "))"
            )
        }
        let root = URL(fileURLWithPath: options["root"] ?? FileManager.default.currentDirectoryPath)
        guard
            let locations = HarnessLocations.repository(
                root: root, environment: ProcessInfo.processInfo.environment)
        else {
            fail(
                "no Fixtures/encoders under \(root.path); run from the repository root or pass --root"
            )
        }
        let passes = Int(options["passes16"] ?? "3") ?? 3
        do {
            let result = try await runBenchmark(
                spec: spec, units: units, locations: locations, passes16: passes)
            let data = try encodeResult(result)
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

    static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("encoder-harness: \(message)\n".utf8))
        exit(1)
    }
}
