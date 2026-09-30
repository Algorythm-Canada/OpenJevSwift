#!/usr/bin/env bash
# Checks the output of `swift test`, saved to a file, for what the exit status alone does not show.
#
#   swift test 2>&1 | tee test.log
#   .github/scripts/check-test-log.sh test.log
#
# It fails unless the log holds at least one Swift Testing run, every run passed, and every skipped
# test or suite was skipped because a model opt-in variable is unset. Tests that need the weights
# or a live server name OPENJEV_TEST_MODEL or OPENJEV_LIVE_URL in their skip comment
# (docs/09-conformance-and-testing.md). Any other skip fails the check: CI has every fixture, so a
# model-free test that skips there would silently stop testing anything.
set -euo pipefail

log=${1:?usage: check-test-log.sh <file holding the output of swift test>}
opt_in='OPENJEV_TEST_MODEL|OPENJEV_LIVE_URL'

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

# A skipped test or suite prints: <symbol> Test <name> skipped: "<comment>". The tests of a skipped
# suite are listed one by one after the suite.
skips=$(grep -E '(Test|Suite) .+ skipped: "' "$log" || true)
model=$(printf '%s\n' "$skips" | grep -E "$opt_in" || true)
other=$(printf '%s\n' "$skips" | grep -Ev "$opt_in" || true)

tests=$(printf '%s\n' "$model" | grep -cE '(^| )Test .+ skipped: "' || true)
suites=$(printf '%s\n' "$model" | grep -cE '(^| )Suite .+ skipped: "' || true)
echo "Skipped because a model opt-in variable is unset: $tests tests, $suites suites"
printf '%s\n' "$model" | indent
if [ -n "$(printf '%s' "$other" | tr -d '[:space:]')" ]; then
    echo "::error::tests were skipped for another reason; every model-free test must run in CI"
    printf '%s\n' "$other" | indent
    exit 1
fi
