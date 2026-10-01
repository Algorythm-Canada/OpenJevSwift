// Written by Tools/encoders/manifest.py from verdict-m18-fp16.mlpackage and the checkpoint's files
// at its pinned revision. Do not edit it by hand: convert the packages, run the script and publish
// the files it prints the commands for (D-033).

import Foundation

extension EncoderPackageManifest {
    /// `verdict-1.4` on Core ML: the `verdict-m18-fp16` package (float16, one function per shape,
    /// iOS 18 and macOS 15) from the GitHub release `verdict-m18-fp16-v1` of
    /// Algorythm-Canada/openjev-models, and the tokenizer and calibrator of
    /// heman10x/rlcd-modernbert-151m at `8af2496`.
    public static let verdict = EncoderPackageManifest(
        model: "verdict-1.4",
        package: "verdict-m18-fp16",
        minimumOS: MinimumOS(iOS: 18, macOS: 15),
        checkpoint: Checkpoint(
            repository: "heman10x/rlcd-modernbert-151m",
            revision: "8af2496eb63c7fa66d7d234e1f62629380030eb4"),
        packageDownloadsEnabled: true,
        packageFiles: [
            File(
                path: "Data/com.apple.CoreML/model.mlmodel",
                url: verdictRelease("verdict-m18-fp16-v1", "Data--com.apple.CoreML--model.mlmodel"),
                bytes: 1_865_826,
                sha256: "c2ae5f4bed7b7d53e41ca078cc7ccffadeac48123d1cd5363a06f98200fdb354"),
            File(
                path: "Data/com.apple.CoreML/weights/weight.bin",
                url: verdictRelease(
                    "verdict-m18-fp16-v1", "Data--com.apple.CoreML--weights--weight.bin"),
                bytes: 303_919_808,
                sha256: "27f4f3af023abb44088646815c9ca3fab407d6f15bf2f1b223adf8f28ca18efd"),
            File(
                path: "Manifest.json",
                url: verdictRelease("verdict-m18-fp16-v1", "Manifest.json"),
                bytes: 617,
                sha256: "160328826c96b0e14a558c73f7b7e4138574d020647a6b2f8637a0064431fb00"),
        ],
        tokenizerFiles: [
            File(
                path: "tokenizer.json",
                url: verdictCheckpoint("tokenizer.json"),
                bytes: 3_583_596,
                sha256: "8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723"),
            File(
                path: "tokenizer_config.json",
                url: verdictCheckpoint("tokenizer_config.json"),
                bytes: 380,
                sha256: "fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016"),
        ],
        calibrator: File(
            path: "calibrator.json",
            url: verdictCheckpoint("calibrator.json"),
            bytes: 1259,
            sha256: "af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de"))
}

/// A package file's URL: an asset of the package's GitHub release (D-033).
private func verdictRelease(_ tag: String, _ asset: String) -> URL {
    URL(
        string: "https://github.com/Algorythm-Canada/openjev-models/releases/download/"
            + tag + "/" + asset)!
}

/// A checkpoint file's URL: Hugging Face, at the pinned revision.
private func verdictCheckpoint(_ path: String) -> URL {
    URL(
        string: "https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/"
            + "8af2496eb63c7fa66d7d234e1f62629380030eb4/" + path)!
}
