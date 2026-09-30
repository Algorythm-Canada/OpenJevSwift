import EncoderHarness
import SwiftUI
import UIKit

/// Measures the configurations named in the launch arguments, one after another, then exits:
///
///     --configs verdict-m18-fp16:all,laya-m18-fp16:cpuAndNeuralEngine [--passes16 3] [--cooldown 30]
///
/// `--cooldown` is the pause in seconds between configurations, so the phone can shed heat.
/// `--settle` is the longest wait, in seconds, for the thermal state to return to nominal before
/// each configuration (0, the default, does not wait); the app prints HARNESS_SETTLE with the wait
/// and the state it reached.
///
/// Each result is printed as one line starting with HARNESS_RESULT, which
/// `xcrun devicectl device process launch --console` passes back to the Mac, and is written to
/// Documents/results/<package>-<units>.json. HARNESS_START, HARNESS_OK and HARNESS_ERROR lines
/// name the configuration they belong to, so that Tools/encoders/run_ios.sh can tell which one was
/// running if Core ML ends the process. The screen stays on while the app runs, so the phone does
/// not lock between configurations.
@main
struct EncoderHarnessApp: App {
    var body: some Scene {
        WindowGroup { RunView() }
    }
}

struct RunView: View {
    @State private var status = "starting"

    var body: some View {
        Text(status)
            .font(.system(.body, design: .default))
            .padding()
            .task { await run() }
    }

    @MainActor
    func run() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        UIApplication.shared.isIdleTimerDisabled = true
        var options: [String: String] = [:]
        var arguments = ProcessInfo.processInfo.arguments.dropFirst()
        while let flag = arguments.popFirst() {
            if flag.hasPrefix("--"), let value = arguments.popFirst() {
                options[String(flag.dropFirst(2))] = value
            }
        }
        // An app package copies its resources to the root of the app bundle.
        guard let staged = Bundle.main.url(forResource: "Staged", withExtension: nil),
            let locations = HarnessLocations.staged(in: staged)
        else {
            return finish(
                "HARNESS_ERROR nothing staged; run Tools/encoders/stage_harness.sh", code: 2)
        }
        let passes = Int(options["passes16"] ?? "3") ?? 3
        let cooldown = Double(options["cooldown"] ?? "30") ?? 30
        let settle = Double(options["settle"] ?? "0") ?? 0
        let configs = (options["configs"] ?? "").split(separator: ",").map {
            $0.split(separator: ":")
        }
        guard !configs.isEmpty else {
            return finish("HARNESS_ERROR pass --configs package:units[,package:units]", code: 2)
        }
        let results = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("results")
        try? FileManager.default.createDirectory(at: results, withIntermediateDirectories: true)
        for (index, config) in configs.enumerated() {
            if index > 0 && cooldown > 0 {
                status = "cooling down for \(Int(cooldown)) s"
                try? await Task.sleep(for: .seconds(cooldown))
            }
            if settle > 0 {
                let started = now()
                while ProcessInfo.processInfo.thermalState != .nominal && now() - started < settle {
                    status = "waiting for the phone to cool (\(thermalStateName()))"
                    try? await Task.sleep(for: .seconds(10))
                }
                print("HARNESS_SETTLE \(Int(now() - started)) s \(thermalStateName())")
            }
            guard config.count == 2, let spec = PackageSpec.named(String(config[0])),
                let units = ComputeUnitsName(rawValue: String(config[1]))
            else {
                print("HARNESS_ERROR unknown configuration \(config.joined(separator: ":"))")
                continue
            }
            status = "measuring \(spec.name) with \(units.rawValue)"
            print("HARNESS_START \(spec.name) \(units.rawValue)")
            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try await runBenchmark(
                        spec: spec, units: units, locations: locations, passes16: passes)
                }.value
                let pretty = try encodeResult(result)
                try pretty.write(
                    to: results.appendingPathComponent("\(spec.name)-\(units.rawValue).json"))
                let line = try JSONEncoder().encode(result)
                print("HARNESS_RESULT " + String(decoding: line, as: UTF8.self))
                print("HARNESS_OK \(spec.name) \(units.rawValue)")
            } catch {
                print("HARNESS_ERROR \(spec.name) \(units.rawValue): \(error)")
            }
        }
        finish("HARNESS_DONE", code: 0)
    }

    @MainActor
    func finish(_ message: String, code: Int32) {
        status = message
        print(message)
        fflush(stdout)
        exit(code)
    }
}
