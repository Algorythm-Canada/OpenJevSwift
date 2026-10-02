# Documentation index

Read in order the first time. Later, each document stands alone.

1. [01-upstream-openjev.md](01-upstream-openjev.md): how upstream OpenJev works, from its source.
2. [02-jev-wire-api.md](02-jev-wire-api.md): the wire contract and client behaviour to satisfy.
3. [03-diffusiongemma.md](03-diffusiongemma.md): the model, the checkpoint, and what a read is.
4. [04-swift-inference-landscape.md](04-swift-inference-landscape.md): what exists in Swift and on Apple platforms, and what does not.
5. [05-architecture.md](05-architecture.md): the proposed design.
6. [06-decisions.md](06-decisions.md): decisions and rejected alternatives.
7. [07-risks-and-unknowns.md](07-risks-and-unknowns.md): what could go wrong and which spike resolves it.
8. [08-implementation-plan.md](08-implementation-plan.md): milestones, order, issue index.
9. [09-conformance-and-testing.md](09-conformance-and-testing.md): how correctness is proven.
10. [10-other-models.md](10-other-models.md): the non-DiffusionGemma models upstream serves.

[deployment.md](deployment.md) runs the server on a Mac: the build, the settings, a launchd job,
the logs, the graceful shutdown and the exit statuses. [development.md](development.md) builds and
tests the package, and builds and previews the API documentation.
[compatibility.md](compatibility.md) says what is identical to upstream, what agrees within a
measured tolerance and what differs and why, with the matrix of what runs on macOS, iOS and Linux.
[credits.md](credits.md) credits the models, their authors and licenses, and the upstream projects.
[quality.md](quality.md) compares the Swift server's answers with upstream's on JevBench and
TypeSafe's public evaluations, and both with the published results, and measures how well
DiffusionGemma's probabilities are calibrated. [benchmarks.md](benchmarks.md) measures
DiffusionGemma's reads: latency, throughput, memory and prefill.

The API documentation lives beside the code, in a DocC catalog per library module
(`Sources/<module>/Documentation.docc`): making decisions in an app, the request and answer types,
implementing a backend, running the server and the configuration reference of every `OPENJEV_*`
variable. `make docs` builds it ([development.md](development.md#api-documentation)), and the
Documentation workflow publishes it to <https://algorythm-canada.github.io/OpenJevSwift/> once
GitHub Pages is enabled for the repository.

[spikes/](spikes/) holds the written outcome of each spike: what was measured, how, and the
minimal reproductions of anything that differed. The decision each one feeds is in
06-decisions.md.

The numbered documents' findings date from 2026-09-29; each spike outcome under spikes/ carries
its own date. Upstream projects referenced here move quickly; pinned revisions are listed in
[../THIRD_PARTY.md](../THIRD_PARTY.md).
