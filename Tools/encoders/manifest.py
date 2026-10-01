#!/usr/bin/env python3
"""Writes the manifests of the Core ML packages that OpenJevEncoders embeds, for Verdict and for
Laya, and prints the commands that publish the packages (D-033).

A manifest lists every file a backend downloads with its size and SHA-256: the files of each
converted package (as convert_verdict.py and convert_laya.py write them to
~/Library/Caches/OpenJevSwift/encoders or OPENJEV_ENCODER_MODELS) under the URLs of the package's
GitHub release, and the checkpoint's tokenizer.json, tokenizer_config.json and calibration file
under their Hugging Face URLs at the pinned revision (from the Hugging Face cache that
reference.py fills). Verdict has one package, verdict-m18-fp16; Laya has five, its Mac package
laya-m18-fp16 and the iPhone's laya-f18-b1s128-fp16 to laya-f18-b1s1024-fp16, one release each. A
package is published as its files, not as an archive, so an iPhone needs no unzip. A release asset
cannot hold a folder, so each package file is uploaded under its path with the slashes replaced by
"--".

Run it after converting a new package, commit the Swift files it writes, then run the printed
commands to publish exactly the files the manifests describe. It uploads nothing itself. --model
picks one model (both by default). --check exits with status 1 when a committed file differs from
what it would write.

Standard library only; it reads the pins from common.py without importing it, so it needs no
virtual environment and no upstream checkout.
"""

import argparse
import ast
import hashlib
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STORE = ROOT / "Sources" / "OpenJevEncoders" / "Store"
MINIMUM_OS = {"iOS": 18, "macOS": 15}
LINE_LENGTH = 100  # .swift-format's lineLength
# Enable a package's downloads only after its release assets are published and verified.
PACKAGE_DOWNLOADS_ENABLED = {
    "verdict-m18-fp16": True,  # verdict-m18-fp16-v1 is published; its assets match (D-033).
    "laya-m18-fp16": False,
    "laya-f18-b1s128-fp16": False,
    "laya-f18-b1s256-fp16": False,
    "laya-f18-b1s512-fp16": False,
    "laya-f18-b1s1024-fp16": False,
}
# A published asset is never replaced: a changed package gets the next release number.
RELEASE = {name: 1 for name in PACKAGE_DOWNLOADS_ENABLED}


@dataclass(frozen=True)
class Package:
    name: str
    description: str  # what it holds, for the release notes
    length: int = 0  # the sequence length of a per-length package, 0 for a multifunction one

    @property
    def tag(self):
        return f"{self.name}-v{RELEASE[self.name]}"


@dataclass(frozen=True)
class Model:
    key: str
    served: str  # the served model name
    pins: tuple  # the common.py names of the checkpoint's repository and revision
    output: Path
    prefix: str  # the Swift helpers' prefix
    converter: str
    credit: str  # the authors, for the release notes and the notice
    tokenizer_folder: str  # where the checkpoint keeps the tokenizer's files, "" for its root
    calibrator: str
    calibration_name: str  # what the Swift comments call the calibration file
    packages: tuple


TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
MODELS = {
    "verdict": Model(
        key="verdict", served="verdict-1.4", pins=("VERDICT_REPO", "VERDICT_REVISION"),
        output=STORE / "EncoderPackageManifest+Verdict.swift", prefix="verdict",
        converter="convert_verdict.py", credit="Verdict by Heman10x", tokenizer_folder="",
        calibrator="calibrator.json", calibration_name="calibrator",
        packages=(Package("verdict-m18-fp16", "Float16, one function per shape"),)),
    "laya": Model(
        key="laya", served="laya-1.0", pins=("LAYA_REPO", "LAYA_REVISION"),
        output=STORE / "EncoderPackageManifest+Laya.swift", prefix="laya",
        converter="convert_laya.py", credit="Laya by Nandakishor M / Convai Innovations",
        tokenizer_folder="tokenizer", calibrator="rl_agent_config.json",
        calibration_name="configuration",
        packages=(Package("laya-m18-fp16", "Float16, one function per shape, the Mac's package"),)
        + tuple(Package(f"laya-f18-b1s{n}-fp16", f"Float16, one program for batch 1 by {n:,} tokens, "
                        "one of the iPhone's packages for the Neural Engine", n)
                for n in (128, 256, 512, 1024))),
}


def pins():
    """The checkpoint repositories and revisions of common.py, read without running it."""
    tree = ast.parse((HERE / "common.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)):
            values[node.targets[0].id] = node.value.value
    return values


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


def package_files(package, converter):
    """Every regular file of the package, as (path relative to it, absolute path), sorted."""
    files = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            files.append((path.relative_to(package).as_posix(), path))
    if not files:
        sys.exit(f"{package} holds no files; run {converter} first")
    return files


def asset_name(relative):
    """The release asset name of a package file: its path with "/" replaced by "--"."""
    return relative.replace("/", "--")


def number(value):
    """An integer literal as swift-format's GroupNumericLiterals rule wants it."""
    return f"{value:_}" if value >= 10_000 else str(value)


def display_path(path):
    """A path relative to the checkout when possible, or its original spelling otherwise."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def call(indent, head, arguments, tail):
    """`head(arguments)tail` on one line, or with the arguments on the next line when it is too
    long, as swift-format breaks a call."""
    one = f"{indent}{head}({arguments}){tail}"
    if len(one) <= LINE_LENGTH:
        return one + "\n"
    return f"{indent}{head}(\n{indent}    {arguments}){tail}\n"


def file_literal(indent, path, url, size, sha, label=""):
    """One EncoderPackageManifest.File expression, its fields one level deeper than `indent`."""
    inner = indent + "    "
    return (f"{indent}{label}File(\n"
            f"{inner}path: \"{path}\",\n"
            + call(inner, "url: " + url[0], url[1], ",")
            + f"{inner}bytes: {number(size)},\n"
            f"{inner}sha256: \"{sha}\")")


def manifest_literal(model, package, entry, repository, revision, indent, head):
    """One EncoderPackageManifest(...) expression; `head` is what precedes it on its line."""
    inner = indent + "    "
    files = inner + "    "
    release = f"{model.prefix}Release"
    checkpoint = f"{model.prefix}Checkpoint"
    package_lines = ",\n".join(
        file_literal(files, path, (release, f"\"{package.tag}\", \"{asset_name(path)}\""), size, sha)
        for path, size, sha in entry["package"])
    folder = model.tokenizer_folder + "/" if model.tokenizer_folder else ""
    tokenizer_lines = ",\n".join(
        file_literal(files, name, (checkpoint, f"\"{folder}{name}\""), size, sha)
        for name, size, sha in entry["tokenizer"])
    name, size, sha = entry["calibrator"]
    calibrator = file_literal(inner, name, (checkpoint, f"\"{name}\""), size, sha, "calibrator: ")
    tail = (f",\n{inner}checkpointTokenizerFolder: \"{model.tokenizer_folder}\")"
            if model.tokenizer_folder else ")")
    enabled = str(PACKAGE_DOWNLOADS_ENABLED[package.name]).lower()
    return (f"{head}EncoderPackageManifest(\n"
            f"{inner}model: \"{model.served}\",\n"
            f"{inner}package: \"{package.name}\",\n"
            f"{inner}minimumOS: MinimumOS(iOS: {MINIMUM_OS['iOS']}, macOS: {MINIMUM_OS['macOS']}),\n"
            f"{inner}checkpoint: Checkpoint(\n"
            f"{inner}    repository: \"{repository}\",\n"
            f"{inner}    revision: \"{revision}\"),\n"
            f"{inner}packageDownloadsEnabled: {enabled},\n"
            f"{inner}packageFiles: [\n{package_lines},\n{inner}],\n"
            f"{inner}tokenizerFiles: [\n{tokenizer_lines},\n{inner}],\n"
            f"{calibrator}{tail}")


def swift_file(model, entries, repository, revision, owner_repository):
    count = len(model.packages)
    source = (f"{model.packages[0].name}.mlpackage" if count == 1
              else f"the {['two', 'three', 'four', 'five', 'six'][count - 2]} .mlpackage folders it names")
    header = (f"Written by Tools/encoders/manifest.py from {source} and the checkpoint's files at its "
              f"pinned revision. Do not edit it by hand: convert the packages, run the script and "
              f"publish the files it prints the commands for (D-033).")
    lines = wrap(header, "// ")
    calibration = model.calibration_name
    body = []
    first = model.packages[0]
    if not first.length:
        doc = (f"`{model.served}` on Core ML: the `{first.name}` package (float16, one function per "
               f"shape, iOS {MINIMUM_OS['iOS']} and macOS {MINIMUM_OS['macOS']}) from the GitHub release "
               f"`{first.tag}` of {owner_repository}, and the tokenizer and {calibration} of "
               f"{repository} at `{revision[:7]}`.")
        if len(model.packages) > 1:
            doc = doc.replace("on Core ML:", "on Core ML for the Mac:", 1)
        body.append(wrap(doc, "    /// ")
                    + manifest_literal(model, first, entries[first.name], repository, revision, "    ",
                                       f"    public static let {model.prefix} = ") + "\n")
    by_length = [p for p in model.packages if p.length]
    if by_length:
        numbers = [f"{p.length:,}" for p in by_length]
        lengths = ", ".join(numbers[:-1]) + " and " + numbers[-1] if len(numbers) > 1 else numbers[0]
        doc = (f"`{model.served}` on Core ML for the iPhone, by sequence length: the packages "
               f"`{by_length[0].name}` to `{by_length[-1].name}` (float16, one program for one shape "
               f"at batch 1, iOS {MINIMUM_OS['iOS']} and macOS {MINIMUM_OS['macOS']}) for {lengths} "
               f"tokens, each from its GitHub release of {owner_repository} (`{by_length[0].tag}` and so "
               f"on), with the same tokenizer and {calibration}.")
        items = ",\n".join(
            manifest_literal(model, p, entries[p.name], repository, revision, "        ",
                             f"        {p.length}: ")
            for p in by_length)
        body.append(wrap(doc, "    /// ")
                    + f"    public static let {model.prefix}ByLength: [Int: EncoderPackageManifest] = [\n"
                    + items + ",\n    ]\n")
    release_prefix = f"https://github.com/{owner_repository}/releases/download/"
    checkpoint_prefix = f"https://huggingface.co/{repository}/resolve/"
    return (lines + "\nimport Foundation\n\nextension EncoderPackageManifest {\n"
            + "\n".join(body) + "}\n\n"
            f"/// A package file's URL: an asset of the package's GitHub release (D-033).\n"
            f"private func {model.prefix}Release(_ tag: String, _ asset: String) -> URL {{\n"
            f"    URL(\n"
            f"        string: \"{release_prefix}\"\n"
            f"            + tag + \"/\" + asset)!\n"
            f"}}\n\n"
            f"/// A checkpoint file's URL: Hugging Face, at the pinned revision.\n"
            f"private func {model.prefix}Checkpoint(_ path: String) -> URL {{\n"
            f"    URL(\n"
            f"        string: \"{checkpoint_prefix}\"\n"
            f"            + \"{revision}/\" + path)!\n"
            f"}}\n")


def wrap(text, prefix):
    """A comment, its words filled to the line length."""
    out, line = [], ""
    for word in text.split():
        if line and len(prefix + line + " " + word) > LINE_LENGTH:
            out.append(prefix + line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    out.append(prefix + line)
    return "\n".join(out) + "\n"


def notice(owner_repository, values):
    """The NOTICE the release repository carries, crediting every checkpoint's authors."""
    return (f"{owner_repository}\n\n"
            "Core ML conversions of the encoder models OpenJevSwift serves, converted with\n"
            "Tools/encoders of https://github.com/Algorythm-Canada/OpenJevSwift. The weights are\n"
            "the checkpoints' own, under their Apache-2.0 license:\n\n"
            f"- Verdict by Heman10x: {values['VERDICT_REPO']} on Hugging Face, revision "
            f"{values['VERDICT_REVISION'][:7]}\n"
            f"- Laya by Nandakishor M / Convai Innovations (github.com/NandhaKishorM/laya): "
            f"{values['LAYA_REPO']} on Hugging Face, revision {values['LAYA_REVISION'][:7]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=["verdict", "laya", "all"], default="all",
                        help="the model whose manifest to write (default: both)")
    parser.add_argument("--models", type=Path, default=default_models(),
                        help="the folder holding the converted packages (default: %(default)s)")
    parser.add_argument("--hub", type=Path, default=default_hub(),
                        help="the Hugging Face hub cache (default: %(default)s)")
    parser.add_argument("--repository", default="Algorythm-Canada/openjev-models",
                        help="the GitHub repository whose releases hold the packages")
    parser.add_argument("--gh", default="ghp", help="the GitHub CLI to print commands for")
    parser.add_argument("--stage", type=Path, default=Path("/tmp/openjev-models"),
                        help="where the printed commands copy the files under their asset names")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 when a Swift file differs")
    args = parser.parse_args()

    values = pins()
    models = list(MODELS.values()) if args.model == "all" else [MODELS[args.model]]
    plans = []
    for model in models:
        repository, revision = values[model.pins[0]], values[model.pins[1]]
        snapshot = args.hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
        tokenizer = snapshot / model.tokenizer_folder if model.tokenizer_folder else snapshot
        for path in [tokenizer / name for name in TOKENIZER_FILES] + [snapshot / model.calibrator]:
            if not path.is_file():
                sys.exit(f"{path} does not exist; run reference.py once to fill the cache")
        checkpoint_files = {
            "tokenizer": [(name, *digest(tokenizer / name)) for name in TOKENIZER_FILES],
            "calibrator": (model.calibrator, *digest(snapshot / model.calibrator)),
        }
        entries, files = {}, {}
        for package in model.packages:
            folder = args.models / f"{package.name}.mlpackage"
            if not folder.is_dir():
                sys.exit(f"{folder} does not exist; run {model.converter} or set "
                         "OPENJEV_ENCODER_MODELS")
            files[package.name] = package_files(folder, model.converter)
            entries[package.name] = {
                "package": [(path, *digest(absolute)) for path, absolute in files[package.name]],
                **checkpoint_files,
            }
        text = swift_file(model, entries, repository, revision, args.repository)
        plans.append((model, repository, revision, entries, files, text))

    if args.check:
        stale = []
        for model, _, _, _, _, text in plans:
            current = model.output.read_text(encoding="utf-8") if model.output.exists() else ""
            if current != text:
                stale.append(display_path(model.output))
            else:
                print(f"{display_path(model.output)} matches the packages in {args.models}")
        if stale:
            sys.exit(f"{', '.join(stale)} differs from the packages in {args.models}; run "
                     "Tools/encoders/manifest.py")
        return

    q = shlex.quote
    print_once = True
    for model, repository, revision, entries, files, text in plans:
        model.output.write_text(text, encoding="utf-8")
        print(f"Wrote {display_path(model.output)}:")
        for package in model.packages:
            print(f"  {package.name} ({package.tag}):")
            entry = entries[package.name]
            for path, size, sha in entry["package"]:
                print(f"    {path}: {size:,} bytes, sha256 {sha}")
        first = entries[model.packages[0].name]
        for path, size, sha in first["tokenizer"] + [first["calibrator"]]:
            print(f"  {path}: {size:,} bytes, sha256 {sha}")
        print()
        print(f"Publish {model.served}'s packages (nothing has been uploaded):")
        print()
        if print_once:
            print_once = False
            print("# Once, if the repository does not exist yet. A release needs a commit to tag, so "
                  "the")
            print("# repository starts with the Apache-2.0 license the checkpoints carry; its NOTICE "
                  "credits")
            print("# the checkpoints' authors.")
            print(f"{args.gh} repo create {q(args.repository)} --public --license apache-2.0 "
                  f"--description {q('Core ML conversions of the models OpenJevSwift serves')}")
            print(f"mkdir -p {q(str(args.stage))}")
            print(f"printf %s {q(notice(args.repository, values))} > {q(str(args.stage / 'NOTICE'))}")
            print(f"{args.gh} api --method PUT repos/{args.repository}/contents/NOTICE "
                  f"-f message={q('Add the NOTICE crediting the checkpoints')} "
                  f"-f content=\"$(base64 < {q(str(args.stage / 'NOTICE'))} | tr -d '\\n')\"")
            print()
        for package in model.packages:
            stage = args.stage / package.tag
            notes = (f"{package.name}.mlpackage for {model.served}: {model.credit} ({repository} at "
                     f"{revision[:7]}, Apache-2.0), converted to Core ML with "
                     f"Tools/encoders/{model.converter} of Algorythm-Canada/OpenJevSwift. "
                     f"{package.description}, iOS {MINIMUM_OS['iOS']} and macOS {MINIMUM_OS['macOS']}. "
                     f"Each asset is a file of the package, its path with / replaced by --; "
                     f"OpenJevEncoders checks every SHA-256 before use.")
            print(f"mkdir -p {q(str(stage))}")
            for path, absolute in files[package.name]:
                print(f"cp {q(str(absolute))} {q(str(stage / asset_name(path)))}")
            print(f"{args.gh} release create {q(package.tag)} --repo {q(args.repository)} "
                  f"--title {q(f'{package.tag} ({model.served})')} --notes {q(notes)}")
            assets = " ".join(q(str(stage / asset_name(path))) for path, _ in files[package.name])
            print(f"{args.gh} release upload {q(package.tag)} --repo {q(args.repository)} {assets}")
            print()


if __name__ == "__main__":
    main()
