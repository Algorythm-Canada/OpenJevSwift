// Written by Tools/encoders/manifest.py from the five .mlpackage folders it names and the
// checkpoint's files at its pinned revision. Do not edit it by hand: convert the packages, run the
// script and publish the files it prints the commands for (D-033).

import Foundation

extension EncoderPackageManifest {
    /// `laya-1.0` on Core ML for the Mac: the `laya-m18-fp16` package (float16, one function per
    /// shape, iOS 18 and macOS 15) from the GitHub release `laya-m18-fp16-v1` of
    /// Algorythm-Canada/openjev-models, and the tokenizer and configuration of
    /// convaiinnovations/laya-typed-decisions at `1a793eb`.
    public static let laya = EncoderPackageManifest(
        model: "laya-1.0",
        package: "laya-m18-fp16",
        minimumOS: MinimumOS(iOS: 18, macOS: 15),
        checkpoint: Checkpoint(
            repository: "convaiinnovations/laya-typed-decisions",
            revision: "1a793eb568e6718f15941d08f85432581df534e3"),
        packageDownloadsEnabled: false,
        packageFiles: [
            File(
                path: "Data/com.apple.CoreML/model.mlmodel",
                url: layaRelease("laya-m18-fp16-v1", "Data--com.apple.CoreML--model.mlmodel"),
                bytes: 3_245_651,
                sha256: "56d8b6fdbb8f5830000b2ab6f92c174ef7bc09829db2b3b867df40b301de1118"),
            File(
                path: "Data/com.apple.CoreML/weights/weight.bin",
                url: layaRelease("laya-m18-fp16-v1", "Data--com.apple.CoreML--weights--weight.bin"),
                bytes: 845_861_120,
                sha256: "ac278ef3d54ede9b5c1bede05af83aba8d28c841e1a2d4b6091d51fa5c52c309"),
            File(
                path: "Manifest.json",
                url: layaRelease("laya-m18-fp16-v1", "Manifest.json"),
                bytes: 617,
                sha256: "f4744429270f3023891e99fba395ba027c2914c94206f22c0d07cfbd9daa66de"),
        ],
        tokenizerFiles: [
            File(
                path: "tokenizer.json",
                url: layaCheckpoint("tokenizer/tokenizer.json"),
                bytes: 3_583_228,
                sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
            File(
                path: "tokenizer_config.json",
                url: layaCheckpoint("tokenizer/tokenizer_config.json"),
                bytes: 337,
                sha256: "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1"),
        ],
        calibrator: File(
            path: "rl_agent_config.json",
            url: layaCheckpoint("rl_agent_config.json"),
            bytes: 847,
            sha256: "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8"),
        checkpointTokenizerFolder: "tokenizer")

    /// `laya-1.0` on Core ML for the iPhone, by sequence length: the packages
    /// `laya-f18-b1s128-fp16` to `laya-f18-b1s1024-fp16` (float16, one program for one shape at
    /// batch 1, iOS 18 and macOS 15) for 128, 256, 512 and 1,024 tokens, each from its GitHub
    /// release of Algorythm-Canada/openjev-models (`laya-f18-b1s128-fp16-v1` and so on), with the
    /// same tokenizer and configuration.
    public static let layaByLength: [Int: EncoderPackageManifest] = [
        128: EncoderPackageManifest(
            model: "laya-1.0",
            package: "laya-f18-b1s128-fp16",
            minimumOS: MinimumOS(iOS: 18, macOS: 15),
            checkpoint: Checkpoint(
                repository: "convaiinnovations/laya-typed-decisions",
                revision: "1a793eb568e6718f15941d08f85432581df534e3"),
            packageDownloadsEnabled: false,
            packageFiles: [
                File(
                    path: "Data/com.apple.CoreML/model.mlmodel",
                    url: layaRelease(
                        "laya-f18-b1s128-fp16-v1", "Data--com.apple.CoreML--model.mlmodel"),
                    bytes: 406_153,
                    sha256: "4652ae4d799d0501651f61e6a2ccfbdcbc4ebdcdd8f41ee5f525eb3da2b9a3f5"),
                File(
                    path: "Data/com.apple.CoreML/weights/weight.bin",
                    url: layaRelease(
                        "laya-f18-b1s128-fp16-v1", "Data--com.apple.CoreML--weights--weight.bin"),
                    bytes: 842_190_144,
                    sha256: "615b255235b2bfa69db0c52ba776b66785e954eec0ee5020a5e8c6de2705c41f"),
                File(
                    path: "Manifest.json",
                    url: layaRelease("laya-f18-b1s128-fp16-v1", "Manifest.json"),
                    bytes: 617,
                    sha256: "468ebde54828ed047e28f0b237331ee7dd97a32de8cbf49c8ecbdb6d4ac350db"),
            ],
            tokenizerFiles: [
                File(
                    path: "tokenizer.json",
                    url: layaCheckpoint("tokenizer/tokenizer.json"),
                    bytes: 3_583_228,
                    sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
                File(
                    path: "tokenizer_config.json",
                    url: layaCheckpoint("tokenizer/tokenizer_config.json"),
                    bytes: 337,
                    sha256: "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1"),
            ],
            calibrator: File(
                path: "rl_agent_config.json",
                url: layaCheckpoint("rl_agent_config.json"),
                bytes: 847,
                sha256: "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8"),
            checkpointTokenizerFolder: "tokenizer"),
        256: EncoderPackageManifest(
            model: "laya-1.0",
            package: "laya-f18-b1s256-fp16",
            minimumOS: MinimumOS(iOS: 18, macOS: 15),
            checkpoint: Checkpoint(
                repository: "convaiinnovations/laya-typed-decisions",
                revision: "1a793eb568e6718f15941d08f85432581df534e3"),
            packageDownloadsEnabled: false,
            packageFiles: [
                File(
                    path: "Data/com.apple.CoreML/model.mlmodel",
                    url: layaRelease(
                        "laya-f18-b1s256-fp16-v1", "Data--com.apple.CoreML--model.mlmodel"),
                    bytes: 406_153,
                    sha256: "3e17af9faa847cc6c5dc951e441767b60095ab697d169f8cfb3f8d7018c21ba4"),
                File(
                    path: "Data/com.apple.CoreML/weights/weight.bin",
                    url: layaRelease(
                        "laya-f18-b1s256-fp16-v1", "Data--com.apple.CoreML--weights--weight.bin"),
                    bytes: 842_353_984,
                    sha256: "6e1ac60d72be272741d391ceca38731ab73e472e1405112c6050e1011bfc470e"),
                File(
                    path: "Manifest.json",
                    url: layaRelease("laya-f18-b1s256-fp16-v1", "Manifest.json"),
                    bytes: 617,
                    sha256: "43ff1bfc1c9ecf8b9b0be3533b838a5279008e0d6ea6f246ec21c1cda86b243d"),
            ],
            tokenizerFiles: [
                File(
                    path: "tokenizer.json",
                    url: layaCheckpoint("tokenizer/tokenizer.json"),
                    bytes: 3_583_228,
                    sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
                File(
                    path: "tokenizer_config.json",
                    url: layaCheckpoint("tokenizer/tokenizer_config.json"),
                    bytes: 337,
                    sha256: "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1"),
            ],
            calibrator: File(
                path: "rl_agent_config.json",
                url: layaCheckpoint("rl_agent_config.json"),
                bytes: 847,
                sha256: "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8"),
            checkpointTokenizerFolder: "tokenizer"),
        512: EncoderPackageManifest(
            model: "laya-1.0",
            package: "laya-f18-b1s512-fp16",
            minimumOS: MinimumOS(iOS: 18, macOS: 15),
            checkpoint: Checkpoint(
                repository: "convaiinnovations/laya-typed-decisions",
                revision: "1a793eb568e6718f15941d08f85432581df534e3"),
            packageDownloadsEnabled: false,
            packageFiles: [
                File(
                    path: "Data/com.apple.CoreML/model.mlmodel",
                    url: layaRelease(
                        "laya-f18-b1s512-fp16-v1", "Data--com.apple.CoreML--model.mlmodel"),
                    bytes: 406_153,
                    sha256: "a547976c301181d334d02af6f910d185b6afe44d32d5953326e44a58fd320078"),
                File(
                    path: "Data/com.apple.CoreML/weights/weight.bin",
                    url: layaRelease(
                        "laya-f18-b1s512-fp16-v1", "Data--com.apple.CoreML--weights--weight.bin"),
                    bytes: 842_878_272,
                    sha256: "4b662aaf0185e0336ca2bb307c829b1e0cbd87f7a62c479d06f12a4d7866f13d"),
                File(
                    path: "Manifest.json",
                    url: layaRelease("laya-f18-b1s512-fp16-v1", "Manifest.json"),
                    bytes: 617,
                    sha256: "1b77c57774dfb0f7b05132f8e7511de532d7f0eee312815706744a2cee74e091"),
            ],
            tokenizerFiles: [
                File(
                    path: "tokenizer.json",
                    url: layaCheckpoint("tokenizer/tokenizer.json"),
                    bytes: 3_583_228,
                    sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
                File(
                    path: "tokenizer_config.json",
                    url: layaCheckpoint("tokenizer/tokenizer_config.json"),
                    bytes: 337,
                    sha256: "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1"),
            ],
            calibrator: File(
                path: "rl_agent_config.json",
                url: layaCheckpoint("rl_agent_config.json"),
                bytes: 847,
                sha256: "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8"),
            checkpointTokenizerFolder: "tokenizer"),
        1024: EncoderPackageManifest(
            model: "laya-1.0",
            package: "laya-f18-b1s1024-fp16",
            minimumOS: MinimumOS(iOS: 18, macOS: 15),
            checkpoint: Checkpoint(
                repository: "convaiinnovations/laya-typed-decisions",
                revision: "1a793eb568e6718f15941d08f85432581df534e3"),
            packageDownloadsEnabled: false,
            packageFiles: [
                File(
                    path: "Data/com.apple.CoreML/model.mlmodel",
                    url: layaRelease(
                        "laya-f18-b1s1024-fp16-v1", "Data--com.apple.CoreML--model.mlmodel"),
                    bytes: 406_154,
                    sha256: "3f200d59e3656c4527185423f032571e0bc958b5ecf35c4d67cafd72b2a1d8be"),
                File(
                    path: "Data/com.apple.CoreML/weights/weight.bin",
                    url: layaRelease(
                        "laya-f18-b1s1024-fp16-v1", "Data--com.apple.CoreML--weights--weight.bin"),
                    bytes: 844_713_280,
                    sha256: "f562aa8ea8bbdb41e6e97a9f711409c41f21d75e5bf8f68a6bd5f0ba8b3175d8"),
                File(
                    path: "Manifest.json",
                    url: layaRelease("laya-f18-b1s1024-fp16-v1", "Manifest.json"),
                    bytes: 617,
                    sha256: "54b083fb29ff5a937c62f9cd69b6935e62718c95727bb58e1ba2ef4d590b2094"),
            ],
            tokenizerFiles: [
                File(
                    path: "tokenizer.json",
                    url: layaCheckpoint("tokenizer/tokenizer.json"),
                    bytes: 3_583_228,
                    sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
                File(
                    path: "tokenizer_config.json",
                    url: layaCheckpoint("tokenizer/tokenizer_config.json"),
                    bytes: 337,
                    sha256: "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1"),
            ],
            calibrator: File(
                path: "rl_agent_config.json",
                url: layaCheckpoint("rl_agent_config.json"),
                bytes: 847,
                sha256: "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8"),
            checkpointTokenizerFolder: "tokenizer"),
    ]
}

/// A package file's URL: an asset of the package's GitHub release (D-033).
private func layaRelease(_ tag: String, _ asset: String) -> URL {
    URL(
        string: "https://github.com/Algorythm-Canada/openjev-models/releases/download/"
            + tag + "/" + asset)!
}

/// A checkpoint file's URL: Hugging Face, at the pinned revision.
private func layaCheckpoint(_ path: String) -> URL {
    URL(
        string: "https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/"
            + "1a793eb568e6718f15941d08f85432581df534e3/" + path)!
}
