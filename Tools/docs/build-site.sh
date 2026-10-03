#!/usr/bin/env bash
# Builds the DocC documentation of the five library modules into one static site, as the
# Documentation workflow (.github/workflows/docs.yml) publishes it to GitHub Pages. See
# docs/development.md, "API documentation".
#
#   Tools/docs/build-site.sh [output-directory]
#
# One plugin call builds the OpenJevCore, OpenJevServer, OpenJevDiffusionGemma, OpenJevEncoders and
# OpenJevLetterReadout archives, each transformed for static hosting under /OpenJevSwift/, and merges them into one
# site with a shared navigator, so that a page can link to another module's symbols
# (``/OpenJevCore/DecisionEngine``). Extended types are left out: OpenJevServer extends two core
# types, and the page DocC would make for them shadows the OpenJevCore module in its links, so
# Swift 6.2's DocC resolved none of them. Tools/docs/index.html becomes the site's front page. Any
# DocC warning, such as a link that does not resolve, fails the build.
#
# macOS only: OpenJevDiffusionGemma, OpenJevEncoders and OpenJevLetterReadout exist only on a macOS
# host. The output
# directory, .build/docs-site by default, is replaced; another directory must be empty, absent or
# a site this script built, whose front page carries the generator line below.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
default_output="$root/.build/docs-site"
output="${1:-$default_output}"
hosting_base_path="OpenJevSwift"
modules=(OpenJevCore OpenJevServer OpenJevDiffusionGemma OpenJevEncoders OpenJevLetterReadout)
# Tools/docs/index.html's generator line. Only a site this script built has it in its index.html,
# so a non-empty directory without it, another DocC site included, is never replaced.
marker='<meta name="generator" content="OpenJevSwift Tools/docs/build-site.sh">'

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build-site.sh: the documentation needs a macOS host, where every module exists" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"
output="$(cd "$(dirname "$output")" && pwd)/$(basename "$output")"
if [[ -e "$output" || -L "$output" ]] && [[ ! -d "$output" ]]; then
    echo "build-site.sh: $output exists and is not a directory; not replacing it" >&2
    exit 1
fi
if [[ "$output" != "$default_output" ]] && [[ -d "$output" ]] && [[ -n "$(ls -A "$output")" ]]; then
    if ! grep -qsF "$marker" "$output/index.html"; then
        echo "build-site.sh: $output is not empty and is not a site this script built; not replacing it" >&2
        exit 1
    fi
fi
rm -rf "$output"

targets=()
for module in "${modules[@]}"; do
    targets+=(--target "$module")
done

cd "$root"
swift package --allow-writing-to-directory "$output" generate-documentation \
    "${targets[@]}" \
    --enable-experimental-combined-documentation \
    --exclude-extended-types \
    --transform-for-static-hosting --hosting-base-path "$hosting_base_path" \
    --warnings-as-errors \
    --output-path "$output"

cp "$root/Tools/docs/index.html" "$output/index.html"
if ! grep -qF "$marker" "$output/index.html"; then
    echo "build-site.sh: Tools/docs/index.html lacks the generator line that marks the script's sites" >&2
    exit 1
fi

for module in "${modules[@]}"; do
    page="$output/documentation/$(echo "$module" | tr '[:upper:]' '[:lower:]')/index.html"
    if [[ ! -f "$page" ]]; then
        echo "build-site.sh: the site lacks $module's page ($page)" >&2
        exit 1
    fi
done
echo "Documentation site: $output (served under /$hosting_base_path/)"
