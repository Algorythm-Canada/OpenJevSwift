"""Download a pinned MLX checkpoint into the Hugging Face cache and list its files.

The weights stay in the cache (HF_HUB_CACHE, by default ~/.cache/huggingface/hub). Nothing is
copied into the repository. Interrupted downloads resume: run the script again.

Usage:
    Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py [--repo REPO --revision REV]
        [--index-only] [--json OUT]

With --index-only only config.json and model.safetensors.index.json are fetched, which is
enough to compare tensor names without the weights (spike #22, part D).
"""
import argparse
import json
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download

DIFFUSIONGEMMA = ("mlx-community/diffusiongemma-26B-A4B-it-4bit",
                  "a7a81407613811e8ba63af92ac0d852b809e191f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DIFFUSIONGEMMA[0])
    ap.add_argument("--revision", default=DIFFUSIONGEMMA[1])
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--json", help="write the file list with sizes and hashes here")
    ap.add_argument("--attempts", type=int, default=20)
    args = ap.parse_args()

    allow = ["config.json", "model.safetensors.index.json"] if args.index_only else None
    path = None
    for attempt in range(1, args.attempts + 1):
        try:
            started = time.time()
            path = snapshot_download(args.repo, revision=args.revision, allow_patterns=allow)
            print(f"snapshot at {path} in {time.time() - started:.0f} s (attempt {attempt})",
                  file=sys.stderr)
            break
        except Exception as exc:  # network errors: back off and resume
            print(f"attempt {attempt} failed: {exc!r}", file=sys.stderr)
            time.sleep(min(60, 5 * attempt))
    if path is None:
        sys.exit("download failed")

    # Sizes and hashes as the Hub reports them for this exact revision.
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    files = []
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        local = os.path.join(path, s.rfilename)
        present = os.path.exists(local)
        files.append({
            "name": s.rfilename,
            "size": s.size,
            "sha256": s.lfs.sha256 if s.lfs else None,
            "git_blob": s.blob_id,
            "local": present,
            "local_size": os.path.getsize(local) if present else None,
        })
    out = {"repo": args.repo, "revision": args.revision, "sha": info.sha,
           "total_bytes": sum(f["size"] or 0 for f in files), "files": files}
    for f in files:
        mark = "ok" if f["local"] and f["local_size"] == f["size"] else ("missing" if not f["local"] else "size mismatch")
        print(f"{f['size']:>14,}  {mark:<13} {f['name']}")
    print(f"{out['total_bytes']:>14,}  total")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")


if __name__ == "__main__":
    main()
