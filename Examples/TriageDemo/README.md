# Triage demo

An iPhone app that makes typed decisions about a customer message on the device, with Verdict
(`verdict-1.4`) through `OpenJevCore` and `OpenJevEncoders`. It asks the three questions of
upstream OpenJev's README example ("Try it"):

| Question | Type | Options |
|---|---|---|
| `urgent`: Does the customer need a reply within the hour? | noul | yes, no |
| `team`: Which team should handle it? | choice | outage, billing, feature |
| `tone`: How upset is the customer? | score | calm, annoyed, furious |

Each answer shows every option's probability as a bar, with the confidence, and the line above
the answers says the read ran on this device and how long it took in the model. The app answers
as you type, a quarter of a second after the last keystroke, and a new keystroke cancels the
read that was waiting. Three sample messages are one tap away: a support ticket, an outage report
(upstream's own example state) and a billing complaint.

The app uses the library's public API only. If Verdict ever refuses one of the question types,
the app asks each question alone, keeps the ones Verdict answers and says which it dropped.

## Run it

You need Xcode 26.4 or later and an iPhone or a simulator with iOS 18 or later.

1. Close any Xcode window that has the OpenJevSwift package open. Xcode lets only one window use a
   local package, and this project depends on the repository's package by its path, `../..`.
2. Open `TriageDemo.xcodeproj`.
3. To run on an iPhone, choose your team for the `TriageDemo` target under Signing &
   Capabilities. The committed project leaves the team empty.
4. Run the `TriageDemo` scheme.

### The first launch

The first launch downloads Verdict through `EncoderPackageStore`, the library's own download:
the converted Core ML package (306 MB) from the
[openjev-models](https://github.com/Algorythm-Canada/openjev-models) release, and the tokenizer
and calibrator (about 4 MB) from Hugging Face, about 310 MB in all. The store checks each file's
size and SHA-256 against the manifest the library embeds before it keeps the file, and puts the
files in `Application Support/OpenJevSwift/encoders`, excluded from backups. The app shows the
download's progress and, if it fails, why, with a button to try again; files already checked are
not downloaded again.

Later launches find the checked files and compiled model in place and need no network: on
2026-10-06 the app answered on an iPhone in Airplane Mode with Wi-Fi off.

A network that inspects encrypted traffic, such as a corporate proxy, presents its own
certificate for the hosts it inspects. A device or simulator that does not trust that certificate
fails the download with an SSL error. On a simulator, add the proxy's root certificate with
`xcrun simctl keychain <device> add-root-cert <certificate.pem>`, or use another network.

### Launch argument

`-TriageText "<message>"` fills in the message at launch, for screenshots and checks that cannot
type:

```bash
xcrun simctl launch booted ca.algorythm.openjev.TriageDemo -TriageText "I was charged twice."
```

## Tests

The `TriageDemo` scheme runs three test groups.

- `RequestTests` checks that the view model builds upstream's README request: it decodes the
  README's JSON with `SystemOneRequest(json:)`, as the server would, and requires the app's
  request to be equal to it, with the questions and each choice's options in the same order, and
  to encode to the same bytes.
- `MacAnswersTests` compares the app's answers to the three samples with what
  `openjev decide --backend verdict` answered on a Mac, kept in `mac-answers.json`, and records
  the largest difference in any probability as an attachment. A Mac reads on the GPU, 16 questions
  per call; an iPhone on the Neural Engine, one at a time; so small differences are expected, and
  the test allows 0.05. It runs only once Verdict is on the device, after the app has launched
  online once; it never downloads.
- `TriageDemoUITests` taps a sample and checks the three answers and each bar's VoiceOver label
  and value, then types a billing complaint a few words at a time. The README's recording is that
  second test. The first run downloads Verdict, so it allows ten minutes.

Measured on 2026-10-06 against the Mac's answers (an M3 Max on macOS 27.0.1):

| Device | All tests | Largest difference from the Mac |
|---|---|---|
| iPhone 16 Pro Max simulator, iOS 18.5 | pass | 0.0066, `urgent` on the outage report |
| iPhone 17 Pro simulator, iOS 26.5 | pass | 0.0043, `urgent` on the billing complaint |
| iPhone 13 Pro Max, iOS 27.0 | pass | 0.0023, `tone` (calm) on the support ticket |

To record `mac-answers.json` again, send each sample through `openjev decide --backend verdict`
with the request `TriageModel.request(for:)` builds, and keep each response under the sample's
name.

## Pinned dependencies

`TriageDemo.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` pins the same
versions as the repository's `Package.resolved`, so the demo builds against what the package's
tests test. The CI iOS job builds the demo for the simulator and fails when the two files
disagree. After changing the root's pins, copy the root's `Package.resolved` over the demo's,
open the demo in Xcode, which keeps those versions and drops the pins the demo does not use, and
commit the file it writes.
