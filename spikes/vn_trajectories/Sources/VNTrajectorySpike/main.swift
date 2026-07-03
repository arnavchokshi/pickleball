import Foundation
import Vision

// W0-BALL-SPIKE rung-1 harness.
//
// Usage:
//   vn_trajectory_spike <input.mp4> <output.json> \
//     [--trajectory-length N] [--min-radius R] [--max-radius R] \
//     [--max-frames N] [--verbose]
//
// On any failure (unsupported OS, unreadable video, Vision error), this
// writes a BlockerReport JSON to <output.json> and exits non-zero instead of
// fabricating numbers.

struct CLIOptions {
    var inputPath: String
    var outputPath: String
    var trajectoryLength: Int = 10
    var minRadius: Float = 0.001
    var maxRadius: Float = 0.05
    var maxFrames: Int? = nil
    var verbose: Bool = false
}

func usageAndExit() -> Never {
    FileHandle.standardError.write(
        """
        Usage: vn_trajectory_spike <input.mp4> <output.json> \
        [--trajectory-length N] [--min-radius R] [--max-radius R] [--max-frames N] [--verbose]

        """.data(using: .utf8)!
    )
    exit(64)
}

func parseArgs() -> CLIOptions {
    var args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else { usageAndExit() }
    var opts = CLIOptions(inputPath: args.removeFirst(), outputPath: args.removeFirst())
    var i = 0
    while i < args.count {
        let arg = args[i]
        func nextValue() -> String {
            i += 1
            guard i < args.count else { usageAndExit() }
            return args[i]
        }
        switch arg {
        case "--trajectory-length":
            guard let v = Int(nextValue()) else { usageAndExit() }
            opts.trajectoryLength = v
        case "--min-radius":
            guard let v = Float(nextValue()) else { usageAndExit() }
            opts.minRadius = v
        case "--max-radius":
            guard let v = Float(nextValue()) else { usageAndExit() }
            opts.maxRadius = v
        case "--max-frames":
            guard let v = Int(nextValue()) else { usageAndExit() }
            opts.maxFrames = v
        case "--verbose":
            opts.verbose = true
        default:
            FileHandle.standardError.write("Unknown argument: \(arg)\n".data(using: .utf8)!)
            usageAndExit()
        }
        i += 1
    }
    return opts
}

func writeBlockerAndExit(reason: String, detail: String, inputPath: String, outputPath: String) -> Never {
    let report = BlockerReport(
        schemaVersion: 1,
        artifactType: "vn_trajectories_spike_blocker",
        status: "BLOCKED",
        blockedReason: reason,
        detail: detail,
        sourceVideo: inputPath,
        osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
        notes: [
            "This is a precise blocker report, not a fabricated result. See BALL_TRACKING_PIPELINE.md for the killed-candidate policy.",
        ]
    )
    do {
        try JSONWriter.write(report, to: outputPath)
    } catch {
        FileHandle.standardError.write("also failed to write blocker report: \(error)\n".data(using: .utf8)!)
    }
    FileHandle.standardError.write("BLOCKED: \(reason): \(detail)\n".data(using: .utf8)!)
    exit(1)
}

let opts = parseArgs()

guard #available(macOS 11.0, *) else {
    writeBlockerAndExit(
        reason: "os_version_unsupported",
        detail: "VNDetectTrajectoriesRequest requires macOS 11.0+; running on \(ProcessInfo.processInfo.operatingSystemVersionString)",
        inputPath: opts.inputPath,
        outputPath: opts.outputPath
    )
}

guard FileManager.default.fileExists(atPath: opts.inputPath) else {
    writeBlockerAndExit(
        reason: "input_not_found",
        detail: "no file at \(opts.inputPath)",
        inputPath: opts.inputPath,
        outputPath: opts.outputPath
    )
}

let harness = VideoTrajectoryHarness(
    inputPath: opts.inputPath,
    trajectoryLength: opts.trajectoryLength,
    objectMinimumNormalizedRadius: opts.minRadius,
    objectMaximumNormalizedRadius: opts.maxRadius,
    maxFrames: opts.maxFrames,
    verbose: opts.verbose
)

do {
    let output = try harness.run()
    try JSONWriter.write(output, to: opts.outputPath)
    FileHandle.standardError.write(
        "OK: \(output.run.framesFed) frames fed, \(output.run.emissionCount) trajectory emissions, \(String(format: "%.2f", output.run.wallClockSeconds))s wall-clock -> \(opts.outputPath)\n".data(using: .utf8)!
    )
} catch {
    writeBlockerAndExit(
        reason: "harness_run_failed",
        detail: "\(error)",
        inputPath: opts.inputPath,
        outputPath: opts.outputPath
    )
}
