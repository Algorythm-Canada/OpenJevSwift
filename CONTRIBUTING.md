# Contributing

Milestones 0 to 5 are complete, and release 0.1.0 is the first version a package can depend on;
the README's [Status](README.md#status) section lists what is done and what is not there yet.
Work is organised as GitHub issues grouped into milestones; see
[docs/08-implementation-plan.md](docs/08-implementation-plan.md).

Open an issue with one of the templates in [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE/): a
bug report when something does not work as the documentation says, or a compatibility report when
OpenJevSwift answers a request differently from upstream OpenJev. Report a vulnerability
privately, as [SECURITY.md](SECURITY.md) describes, never in a public issue.

## How issues are written

Every implementation issue states what to build, why it is needed, the relevant upstream
findings, the expected behaviour, acceptance criteria, the tests required, its dependencies
and the upstream files to read. An issue should be workable on its own by an engineer or a
coding agent who has not read the whole repository.

Labels:

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
