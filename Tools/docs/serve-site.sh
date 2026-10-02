#!/usr/bin/env bash
# Serves a site Tools/docs/build-site.sh built at http://localhost:<port>/OpenJevSwift/, the path
# GitHub Pages serves it under, so its links resolve as they will once published. Ctrl-C stops it.
#
#   Tools/docs/serve-site.sh [site-directory] [port]
#
# The defaults are .build/docs-site and 8000. Needs python3 for its http.server module.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
site="${1:-$root/.build/docs-site}"
port="${2:-8000}"

if [[ ! -f "$site/index.html" || ! -d "$site/documentation" ]]; then
    echo "serve-site.sh: no site at $site; run Tools/docs/build-site.sh (make docs) first" >&2
    exit 1
fi
site="$(cd "$site" && pwd)"

# The base path is a folder of its own, a link to the site, in a temporary directory.
served="$(mktemp -d)"
trap 'rm -rf "$served"' EXIT
ln -s "$site" "$served/OpenJevSwift"

echo "Serving $site at http://localhost:$port/OpenJevSwift/"
python3 -m http.server "$port" --bind 127.0.0.1 --directory "$served"
