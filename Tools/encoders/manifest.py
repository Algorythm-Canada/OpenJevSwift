#!/usr/bin/env python3
"""Writes the manifest of Verdict's Core ML package that OpenJevEncoders embeds, and prints the
commands that publish the package (D-033).

The manifest lists every file the backend downloads with its size and SHA-256: the files of the
converted package (verdict-m18-fp16.mlpackage, as convert_verdict.py writes it to
~/Library/Caches/OpenJevSwift/encoders or OPENJEV_ENCODER_MODELS) under the URLs of a GitHub
release, and the checkpoint's tokenizer.json, tokenizer_config.json and calibrator.json under
their Hugging Face URLs at the pinned revision (from the Hugging Face cache that reference.py
fills). The package is published as its files, not as an archive, so an iPhone needs no unzip.
A release asset cannot hold a folder, so each package file is uploaded under its path with the
slashes replaced by "--".

Run it after converting a new package, commit the Swift file it writes, then run the printed
commands to publish exactly the files the manifest describes. It uploads nothing itself.
--check exits with status 1 when the committed file differs from what it would write.

Standard library only; it reads the pins from common.py without importing it, so it needs no
virtual environment and no upstream checkout.
"""

import argparse
import ast
import hashlib
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "Sources" / "OpenJevEncoders" / "Store" / "EncoderPackageManifest+Verdict.swift"
MODEL = "verdict-1.4"
PACKAGE = "verdict-m18-fp16"
MINIMUM_OS = {"iOS": 18, "macOS": 15}
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
CALIBRATOR = "calibrator.json"


def pins():
    """VERDICT_REPO and VERDICT_REVISION from common.py, read without running it."""
    tree = ast.parse((HERE / "common.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)):
            values[node.targets[0].id] = node.value.value
    return values["VERDICT_REPO"], values["VERDICT_REVISION"]


def default_models():
    return Path(os.environ.get("OPENJEV_ENCODER_MODELS")
                or Path.home() / "Library" / "Caches" / "OpenJevSwift" / "encoders")


def default_hub():
    """The Hugging Face hub cache, found as huggingface_hub finds it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    return cache / "huggingface" / "hub"


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    return path.stat().st_size, sha.hexdigest()


def package_files(package):
    """Every regular file of the package, as (path relative to it, absolute path), sorted."""
    files = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            files.append((path.relative_to(package).as_posix(), path))
    if not files:
        sys.exit(f"{package} holds no files; run convert_verdict.py first")
    return files


def asset_name(relative):
    """The release asset name of a package file: its path with "/" replaced by "--"."""
    return relative.replace("/", "--")


def number(value):
    """An integer literal as swift-format's GroupNumericLiterals rule wants it."""
    return f"{value:_}" if value >= 10_000 else str(value)


def swift_file(entry, release_url, checkpoint_url, repository, revision, tag, owner_repository):
    def file_literal(path, base, name, size, sha):
        return (f"            File(\n"
                f"                path: \"{path}\",\n"
                f"                url: {base}(\"{name}\"),\n"
                f"                bytes: {number(size)},\n"
                f"                sha256: \"{sha}\"),\n")

    package_lines = "".join(file_literal(path, "verdictRelease", asset_name(path), size, sha)
                            for path, size, sha in entry["package"])
    tokenizer_lines = "".join(file_literal(name, "verdictCheckpoint", name, size, sha)
                              for name, size, sha in entry["tokenizer"])
    name, size, sha = entry["calibrator"]
    calibrator = file_literal(name, "verdictCheckpoint", name, size, sha)
    calibrator = calibrator.replace("            File(", "        calibrator: File(", 1)
    calibrator = "\n".join(line[4:] if index > 0 else line
                           for index, line in enumerate(calibrator.rstrip(",\n").split("\n")))
    release_prefix, release_tag = release_url.rsplit("/", 2)[0] + "/", tag + "/"
    checkpoint_prefix, checkpoint_revision = checkpoint_url.rsplit("/", 2)[0] + "/", revision + "/"
    source = f"{PACKAGE}.mlpackage"
    return f"""// Written by Tools/encoders/manifest.py from {source} and the checkpoint's
// files at its pinned revision. Do not edit it by hand: convert the package, run the script and
// publish the files it prints the commands for (D-033).

import Foundation

extension EncoderPackageManifest {{
    /// `{MODEL}` on Core ML: the `{PACKAGE}` package (float16, one function per shape,
    /// iOS {MINIMUM_OS["iOS"]} and macOS {MINIMUM_OS["macOS"]}) from the GitHub release `{tag}` of
    /// {owner_repository}, and the tokenizer and calibrator of
    /// {repository} at `{revision[:7]}`.
    public static let verdict = EncoderPackageManifest(
        model: "{MODEL}",
        package: "{PACKAGE}",
        minimumOS: MinimumOS(iOS: {MINIMUM_OS["iOS"]}, macOS: {MINIMUM_OS["macOS"]}),
        checkpoint: Checkpoint(
            repository: "{repository}",
            revision: "{revision}"),
        packageFiles: [
{package_lines}        ],
        tokenizerFiles: [
{tokenizer_lines}        ],
{calibrator})
}}

/// A package file's URL: an asset of the GitHub release (proposed in D-033).
private func verdictRelease(_ asset: String) -> URL {{
    URL(
        string: "{release_prefix}"
            + "{release_tag}" + asset)!
}}

/// A checkpoint file's URL: Hugging Face, at the pinned revision.
private func verdictCheckpoint(_ name: String) -> URL {{
    URL(
        string: "{checkpoint_prefix}"
            + "{checkpoint_revision}" + name)!
}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", type=Path, default=default_models(),
                        help="the folder holding the converted package (default: %(default)s)")
    parser.add_argument("--hub", type=Path, default=default_hub(),
                        help="the Hugging Face hub cache (default: %(default)s)")
    parser.add_argument("--repository", default="Algorythm-Canada/openjev-models",
                        help="the GitHub repository whose release holds the package")
    parser.add_argument("--tag", default=f"{PACKAGE}-v1", help="the release tag")
    parser.add_argument("--gh", default="ghp", help="the GitHub CLI to print commands for")
    parser.add_argument("--stage", type=Path, default=Path("/tmp/openjev-models"),
                        help="where the printed commands copy the files under their asset names")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="the Swift file to write")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 when the Swift file differs")
    args = parser.parse_args()

    repository, revision = pins()
    package = args.models / f"{PACKAGE}.mlpackage"
    snapshot = args.hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    if not package.is_dir():
        sys.exit(f"{package} does not exist; run convert_verdict.py or set OPENJEV_ENCODER_MODELS")
    for name in TOKENIZER_FILES + [CALIBRATOR]:
        if not (snapshot / name).is_file():
            sys.exit(f"{snapshot / name} does not exist; run reference.py once to fill the cache")

    files = package_files(package)
    entry = {
        "package": [(path, *digest(absolute)) for path, absolute in files],
        "tokenizer": [(name, *digest(snapshot / name)) for name in TOKENIZER_FILES],
        "calibrator": (CALIBRATOR, *digest(snapshot / CALIBRATOR)),
    }
    release_url = f"https://github.com/{args.repository}/releases/download/{args.tag}/"
    checkpoint_url = f"https://huggingface.co/{repository}/resolve/{revision}/"
    text = swift_file(entry, release_url, checkpoint_url, repository, revision, args.tag,
                      args.repository)

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            sys.exit(f"{args.output.relative_to(ROOT)} differs from {package}; run "
                     "Tools/encoders/manifest.py")
        print(f"{args.output.relative_to(ROOT)} matches {package}")
        return
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output.relative_to(ROOT)}:")
    for path, size, sha in entry["package"] + entry["tokenizer"] + [entry["calibrator"]]:
        print(f"  {path}: {size:,} bytes, sha256 {sha}")

    stage = args.stage / args.tag
    q = shlex.quote
    notes = (f"{PACKAGE}.mlpackage for {MODEL}: Verdict by Heman10x ({repository} "
             f"at {revision[:7]}, Apache-2.0), converted to Core ML with "
             f"Tools/encoders/convert_verdict.py of Algorythm-Canada/OpenJevSwift. Float16, one "
             f"function per shape, iOS {MINIMUM_OS['iOS']} and macOS {MINIMUM_OS['macOS']}. "
             f"Each asset is a file of the package, its path with / replaced by --; "
             f"OpenJevEncoders checks every SHA-256 before use.")
    print()
    print("Publish the package (nothing has been uploaded):")
    print()
    print("# Once, if the repository does not exist yet. A release needs a commit to tag, so the")
    print("# repository starts with the Apache-2.0 license the checkpoints carry.")
    print(f"{args.gh} repo create {q(args.repository)} --public --license apache-2.0 "
          f"--description {q('Core ML conversions of the models OpenJevSwift serves')}")
    print(f"mkdir -p {q(str(stage))}")
    for path, absolute in files:
        print(f"cp {q(str(absolute))} {q(str(stage / asset_name(path)))}")
    print(f"{args.gh} release create {q(args.tag)} --repo {q(args.repository)} "
          f"--title {q(f'{args.tag} ({MODEL})')} --notes {q(notes)}")
    assets = " ".join(q(str(stage / asset_name(path))) for path, _ in files)
    print(f"{args.gh} release upload {q(args.tag)} --repo {q(args.repository)} {assets}")


if __name__ == "__main__":
    main()
