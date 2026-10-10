# Contributing

Contributions are welcome from anyone: bug reports, compatibility reports, benchmark rows,
documentation fixes and code. Release 0.1.0 is the first version a package can depend on; the
README's [Status](README.md#status) section lists what is done and what is not there yet. Work is
organised as GitHub issues grouped into milestones; see
[docs/08-implementation-plan.md](docs/08-implementation-plan.md).

## Ways to help

- **Report a problem or ask for a feature.** Open an issue with one of the
  [templates](https://github.com/Algorythm-Canada/OpenJevSwift/issues/new/choose): a bug report
  when something does not work as the documentation says, a compatibility report when
  OpenJevSwift answers a request differently from upstream OpenJev, or a feature request. Report a
  vulnerability privately, as [SECURITY.md](SECURITY.md) describes, never in a public issue.
- **Ask a question or share an idea** in
  [Discussions](https://github.com/Algorythm-Canada/OpenJevSwift/discussions).
- **Pick up an issue.** Issues labelled
  [help wanted](https://github.com/Algorythm-Canada/OpenJevSwift/labels/help%20wanted) are open to
  anyone. Comment on the issue before you start, so two people do not do the same work.
- **Measure your Mac.** Every figure in [docs/benchmarks.md](docs/benchmarks.md) comes from one
  M3 Max with 128 GB. Rows from a 32 GB or 48 GB Mac and from a 64 GB one would complete it:
  [#143](https://github.com/Algorythm-Canada/OpenJevSwift/issues/143) says how to measure and
  submit them.
- **Add your organization** to [ADOPTERS.md](ADOPTERS.md) if it uses OpenJevSwift.

## Set up

On an Apple silicon Mac with Xcode 27, install Xcode's Metal Toolchain component once; the build
needs it to compile MLX's Metal shaders. Then build and test your fork:

```bash
xcodebuild -downloadComponent MetalToolchain
git clone https://github.com/YOUR-USERNAME/OpenJevSwift.git
cd OpenJevSwift
make test
make lint
```

With Xcode 26.4 to 26.6, run `swift test --build-system swiftbuild` instead of `make test`. On
Linux, Swift 6.2 or later builds and tests the core, the server and the `openjev` tool, without any
backend. Tests that need model weights skip cleanly without them, so the suite passes on a machine
that has none. [docs/development.md](docs/development.md) covers the toolchains, every command and
how to run each CI job locally.

## Open a pull request

1. Fork the repository and create a branch from `main`.
2. Keep the change to one issue. For anything larger than a small fix, open or comment on an issue
   first, so the approach is agreed before you write it.
3. Run `make format`, then `make lint` and `make test`.
4. Open the pull request against `main` and fill in its template, linking the issue it closes.

CI builds and tests the change on Linux, macOS and the iOS simulator, runs TypeSafe's SDKs against
the server and checks the formatting. It skips a pull request that changes only Markdown files that
no test reads. A maintainer may need to approve the workflow runs of a first-time contributor before
they start. A pull request merges once a maintainer has approved it and every review thread is
resolved.

Contributions are accepted under the [Apache License 2.0](LICENSE), as its section 5 describes.

## How issues are written

Every implementation issue states what to build, why it is needed, the relevant upstream
findings, the expected behaviour, acceptance criteria, the tests required, its dependencies
and the upstream files to read. An issue should be workable on its own by an engineer or a
coding agent who has not read the whole repository.

Labels:

- `good first issue` marks a self-contained task that needs little knowledge of the codebase.
- `help wanted` marks work that anyone is welcome to pick up.
- `area/*` says which module the work lands in.
- `kind/spike` is a time-boxed investigation. Its output is a written finding in
  `docs/06-decisions.md` or `docs/07-risks-and-unknowns.md`, and possibly follow-up issues, not
  production code.
- `kind/decision` marks an issue that must record a decision before its code merges.
- `kind/tracking` marks a milestone's tracking issue.
- `needs-hardware` marks work that needs an Apple silicon Mac with 32 GB or more of unified
  memory and the DiffusionGemma weights (about 17 GB).

## Ground rules for the implementation phase

- Byte-exact compatibility with upstream OpenJev's prompts, templates, labels, canvases and
  wire shapes is a requirement, not a preference. Deviations are recorded in
  `docs/06-decisions.md` before they merge.
- Golden fixtures generated from the pinned upstream commit are the primary test oracle for the
  model-free core. Tests that need weights are opt-in and skip cleanly without them.
- Never commit model weights, tokenizer files or generated caches.
- Every port of third-party code keeps its attribution in `THIRD_PARTY.md` and in the file header.
- Pull requests target `main`. Keep a change scoped to one issue.

## Writing style

Plain sentences. No em dashes. Say what a thing is and why, then stop.
