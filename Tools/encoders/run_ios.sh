#!/bin/bash
# Measure converted packages on a connected iPhone through Tools/encoders/HarnessApp.swiftpm.
#
#   DEVELOPMENT_TEAM=<team id> Tools/encoders/run_ios.sh <device id> package:units [package:units ...]
#
# `xcrun devicectl list devices` shows the device id. units is cpuOnly, cpuAndGPU,
# cpuAndNeuralEngine or all. The script stages the packages named (Tools/encoders/stage_harness.sh),
# builds the app in Release and installs it. One launch then measures the configurations in the
# order given, with the screen kept on, so the phone cannot lock between them. If Core ML ends the
# process, the configuration that was running is recorded as a failure with the device's newest
# crash log for the app, and the rest run in a new launch. Each result is copied from the app's
# Documents folder to docs/spikes/encoder-runtime/iphone/<package>-<units>.json; failures go to
# docs/spikes/encoder-runtime/iphone/failures.txt. A configuration that runs again replaces its
# earlier result, or removes it if it fails, so the folder never mixes runs; a result that cannot be
# copied from the phone is recorded as a failure and the script exits with status 1. iOS launches
# apps only on an unlocked phone, so each launch waits up to 30 minutes for it to be unlocked.
# TIMEOUT (seconds, default 7200) ends a launch that hangs; PASSES16 (default 3) is how many times
# the corpus is read in calls of 16; COOLDOWN (default 30) is the app's pause in seconds between
# configurations; SETTLE (default 0) is the longest the app waits, before each configuration, for
# the phone's thermal state to return to nominal; RESULTS_DIR replaces the results folder.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
APP_PACKAGE="$ROOT/Tools/encoders/HarnessApp.swiftpm"
DERIVED="$ROOT/Tools/encoders/Harness/results/DerivedData-app"
APP="$DERIVED/Build/Products/Release-iphoneos/EncoderHarnessApp.app"
BUNDLE=org.openjevswift.encoderharness
OUT="${RESULTS_DIR:-$ROOT/docs/spikes/encoder-runtime/iphone}"
LOGS="$ROOT/Tools/encoders/Harness/results/iphone-logs/$(date +%Y%m%d-%H%M%S)"
DEVICE=${1:?usage: run_ios.sh <device id> package:units [package:units ...]}
shift
TEAM=${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM to the team that signs the app}
[ $# -gt 0 ] || { echo "name at least one package:units configuration" >&2; exit 1; }

# A portable time limit: perl's alarm ends the command it runs.
with_timeout() {
    /usr/bin/perl -e 'alarm shift; exec @ARGV' "$@"
}

# The app's newest crash report, or the newest jetsam report, written since the launch began
# ($1, as YYYY-MM-DD-HHMMSS, the stamp report names carry), copied next to the logs. iOS can take
# a few seconds to write a report, so the listing is retried. The app's output reaches this script
# only when a launch ends, so a report cannot be matched to the configuration that was running;
# check its time against the result files.
fetch_crash_log() {
    local since=$1 listing="$LOGS/crash-listing.txt" name="" attempt
    for attempt in 1 2 3 4 5 6; do
        xcrun devicectl device info files --device "$DEVICE" --domain-type systemCrashLogs > "$listing" 2>/dev/null
        name=$(grep -oE '(EncoderHarnessApp|JetsamEvent)-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6}\.ips' "$listing" \
            | awk -F- -v since="$since" '{stamp = $2 "-" $3 "-" $4 "-" substr($5, 1, 6); if (stamp >= since) print stamp " " $0}' \
            | sort | tail -1 | cut -d' ' -f2)
        [ -n "$name" ] && break
        sleep 10
    done
    [ -n "$name" ] || return 0
    xcrun devicectl device copy from --device "$DEVICE" --domain-type systemCrashLogs --source "$name" \
        --destination "$LOGS/$name" > /dev/null 2>&1 && echo "$LOGS/$name"
}

packages=$(printf '%s\n' "$@" | cut -d: -f1 | sort -u)
# shellcheck disable=SC2086
"$ROOT/Tools/encoders/stage_harness.sh" $packages || exit 1
(cd "$APP_PACKAGE" && xcodebuild build -quiet -scheme EncoderHarnessApp -configuration Release \
    -destination "platform=iOS,id=$DEVICE" -derivedDataPath "$DERIVED" \
    DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic) || exit 1
xcrun devicectl device install app --device "$DEVICE" "$APP" > /dev/null || exit 1

mkdir -p "$OUT" "$LOGS"
touch "$OUT/failures.txt"
failed=0
remaining=("$@")
launch=0
while [ ${#remaining[@]} -gt 0 ]; do
    launch=$((launch + 1))
    log="$LOGS/launch-$launch.log"
    list=$(IFS=,; echo "${remaining[*]}")
    # iOS refuses to launch an app on a locked phone; wait for it to be unlocked, up to 30 minutes.
    for wait in $(seq 1 180); do
        xcrun devicectl device info lockState --device "$DEVICE" 2>/dev/null | grep -q "passcodeRequired: false" && break
        [ "$wait" -eq 1 ] && echo "waiting for the phone to be unlocked"
        sleep 10
    done
    started=$(date +%Y-%m-%d-%H%M%S)
    echo "launch $launch: $list"
    with_timeout "${TIMEOUT:-7200}" xcrun devicectl device process launch --device "$DEVICE" --console \
        --terminate-existing "$BUNDLE" --configs "$list" --passes16 "${PASSES16:-3}" --cooldown "${COOLDOWN:-30}" --settle "${SETTLE:-0}" 2>&1 | tr -d '\r' > "$log"
    if ! grep -q '^HARNESS_START ' "$log"; then
        { echo "launch $launch did not start ($list)"; tail -4 "$log" | sed 's/^/    /'; } >> "$OUT/failures.txt"
        echo "launch $launch did not start; see $log (is the phone locked?)"
        exit 1
    fi
    running=$(grep '^HARNESS_START ' "$log" | tail -1 | awk '{print $2 ":" $3}')
    next=()
    for config in "${remaining[@]}"; do
        package=${config%%:*}
        units=${config##*:}
        name="$package-$units"
        if grep -q "^HARNESS_OK $package $units\$" "$log"; then
            rm -f "$OUT/$name.json"
            if xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer \
                --domain-identifier "$BUNDLE" --source "Documents/results/$name.json" \
                --destination "$OUT/$name.json" > /dev/null 2>&1; then
                echo "$config: done"
            else
                echo "$config: finished, but its result could not be copied from the phone" >> "$OUT/failures.txt"
                echo "$config: result not copied (see failures.txt)"
                failed=1
            fi
        elif grep -q "^HARNESS_ERROR $package $units" "$log"; then
            rm -f "$OUT/$name.json"
            grep "^HARNESS_ERROR $package $units" "$log" | head -1 >> "$OUT/failures.txt"
            echo "$config: error (see failures.txt)"
        elif [ "$config" = "$running" ] && ! grep -q '^HARNESS_DONE' "$log"; then
            rm -f "$OUT/$name.json"
            crash=$(fetch_crash_log "$started")
            { echo "$config: the process ended while it ran${crash:+; newest device report since the launch began: $(basename "$crash"), which can belong to an earlier configuration of the launch}"
              grep -v '^HARNESS_RESULT ' "$log" | tail -3 | sed 's/^/    /'; } >> "$OUT/failures.txt"
            echo "$config: process ended while it ran"
        else
            next+=("$config")
        fi
    done
    if [ ${#next[@]} -gt 0 ] && grep -q '^HARNESS_DONE' "$log"; then
        echo "launch $launch finished without running ${next[*]}" >> "$OUT/failures.txt"
        break
    fi
    remaining=("${next[@]+"${next[@]}"}")
done
exit $failed
