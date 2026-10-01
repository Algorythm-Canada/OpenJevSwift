#!/usr/bin/env bash
# Checks the output of `swift test`, saved to a file, for what the exit status alone does not show.
#
#   swift test 2>&1 | tee test.log
#   .github/scripts/check-test-log.sh test.log
#
# It fails unless the log holds at least one Swift Testing run, every run passed, and every skipped
# or cancelled test or suite stopped because a model opt-in variable is unset. Tests that need the
# weights, a converted encoder package or its tokenizer, a live server or the Hugging Face Hub name
# OPENJEV_TEST_MODEL, OPENJEV_ENCODER_MODELS, OPENJEV_LIVE_URL or OPENJEV_TEST_DOWNLOAD in their
# skip comment
# (docs/09-conformance-and-testing.md). Any other skip fails the check, a skip without a comment
# included: CI has every fixture, so a model-free test that skips there would silently stop
# testing anything.
set -euo pipefail

log=${1:?usage: check-test-log.sh <file holding the output of swift test>}
opt_in='OPENJEV_TEST_MODEL|OPENJEV_ENCODER_MODELS|OPENJEV_LIVE_URL|OPENJEV_TEST_DOWNLOAD'

indent() { sed '/^$/d; s/^/  /'; }

# Swift Testing ends every run with "Test run with N tests in M suites passed|failed ...". There is
# one run per test product, so a log can hold several.
runs=$(grep -E 'Test run with [0-9]+ tests? in [0-9]+ suites? (passed|failed)' "$log" || true)
if [ -z "$runs" ]; then
    echo "::error::$log holds no Swift Testing run summary"
    exit 1
fi
echo "Swift Testing runs:"
printf '%s\n' "$runs" | indent
if printf '%s\n' "$runs" | grep -q ' failed '; then
    echo "::error::a Swift Testing run failed"
    exit 1
fi

# A skipped test or suite prints <symbol> Test <name> skipped: "<comment>", or ends in "skipped."
# when it has no comment; the tests of a skipped suite follow one by one. Test.cancel prints
# <symbol> Test <name> was cancelled after <seconds> seconds: "<comment>".
skips=$(grep -E '(Test|Suite) .+ (skipped: "|skipped\.$|was cancelled( after [0-9.]+ seconds?)?(: "|\.$))' "$log" || true)
model=$(printf '%s\n' "$skips" | grep -E "$opt_in" || true)
other=$(printf '%s\n' "$skips" | grep -Ev "$opt_in" || true)

tests=$(printf '%s\n' "$model" | grep -cE '(^| )Test ' || true)
suites=$(printf '%s\n' "$model" | grep -cE '(^| )Suite ' || true)
echo "Skipped because a model opt-in variable is unset: $tests tests, $suites suites"
printf '%s\n' "$model" | indent
if [ -n "$(printf '%s' "$other" | tr -d '[:space:]')" ]; then
    echo "::error::tests were skipped or cancelled for another reason; every model-free test must run in CI"
    printf '%s\n' "$other" | indent
    exit 1
fi
