"""Calibration of the raw reads (issue #62), from the result files under results/.

    python3 Tools/jevbench/harness.py calibration

For every run of a model, over the items with an expected label: JevBench's Brier score and ECE,
the negative log-likelihood of the expected label, the reliability bins (the probability of the
chosen answer against how often it is right, 10 equal-width bins with their counts), and how the
answer's `confidence` relates to observed accuracy per question type and option count. Then
SemIf's post-hoc temperature scaling, fitted offline on the same items: one scalar T, the ECE out
of fold under group-disjoint 5-fold cross-validation, and 95% bootstrap intervals over the groups.

The formulas, and where each comes from:

- Brier: the mean over items of sum_k (p_k - y_k)^2 over the item's labels, JevBench's
  `summarize.metric` (vendor/jevbench, `metrics.py` documents it). For a noul it is
  2 (p_yes - y)^2.
- ECE: sum_b (n_b / N) |acc_b - conf_b| over 10 equal-width bins of the top label's probability,
  the bin of c being min(floor(10 c), 9): JevBench's `metrics.ece_top_label`, called here, and the
  same definition as SemIf's `calibrate.ece`.
- NLL: -(1 / N) sum_i ln max(p_i(expected), 1e-12), SemIf's `calibrate.mean_nll`.
- `confidence`: upstream's `engine.confidence` at dcd2094, 1 - H(p) / ln K clamped to [0, 1], with
  H(p) = -sum p_k ln p_k and K the number of labels. A choice or score answer carries it; a noul
  answer does not, so for a noul it is computed here from (p_yes, p_no) with the same formula.
- AUROC: the probability that a right answer has a higher value than a wrong one, ties counting
  one half (the Mann-Whitney statistic), for the top label's probability and for `confidence`.
- Temperature scaling (SemIf `docs/CALIBRATION.md` and `benchmarks/calibrate.py` at 23cf1f3):
  q = softmax(z / T), with T fitted by golden-section search over [0.05, 20] (60 iterations) to
  minimise the mean NLL, and the ECE reported out of fold: the groups are shuffled with seed 217
  and dealt into 5 folds, each fold scored with the T fitted on the other four, and the 95%
  interval is the 2.5th and 97.5th of 1,000 resamples of the groups with replacement. The logits
  z are the logarithms of the answer's own probabilities, so q_k = p_k^(1/T) / sum_j p_j^(1/T):
  the server's label probabilities are a softmax over label log-probabilities, averaged over the
  reads of one question, and only that average reaches the wire. Dividing by T keeps the argmax,
  so accuracy does not move; only the probabilities do.

`fit_temperature`, `fold_map`, `bootstrap` and `paired_bootstrap` follow SemIf's
`benchmarks/calibrate.py` (TheoLeeCJ/SemIf-OpenJev at 23cf1f3, MIT, Copyright (c) 2026 TheoLeeCJ;
the license is vendor/semif/LICENSE), with two departures: the folds and the bootstrap draw from
Python's `random.Random(217)` rather than NumPy's `default_rng(217)`, because the harness uses only
the standard library, so the method is SemIf's and the assignment of groups is not; and a
temperature divides the logarithms of the answer's probabilities, since an answer carries no
logits. SemIf's paired interval is of the change between two temperatures on the same rows; here it
is also reported for the change from T = 1, beside SemIf's own test (the two unpaired intervals
not overlapping), which is stricter.

A JevBench item's group is its paraphrase group, or the item alone; a TypeSafe row's is its case.
A T is fitted on every run with hard labels: JevBench's, or a deployment's own items run with
`harness.py run --items`. SemIf leaves rows with a reference distribution, as TypeSafe's are, out
of its fits; here they are scored against the reference's top option, as the harness scores them,
and the T fitted on JevBench is applied to them, never fitted on them.
"""

from __future__ import annotations

import math
import random
import textwrap
from pathlib import Path

import harness
import report
from harness import markdown_table, num, pct
from jevbench import metrics as jb_metrics  # noqa: E402  (harness puts vendor/jevbench on the path)
from jevbench import scoring as jb_scoring  # noqa: E402

BINS = 10
FOLDS = 5
SEED = 217
SAMPLES = 1000
BOUNDS = (0.05, 20.0)
ITERATIONS = 60
NLL_FLOOR = 1e-12
TYPES = ("noul", "choice", "score")
# The bins of `confidence` in the table of accuracy by confidence: 0.2 wide, the last closed.
CONFIDENCE_BINS = 5
# Datasets whose expected label is the top option of a reference distribution rather than a hard
# label. SemIf fits no temperature on such rows; neither does this, it only applies one to them.
SOFT_LABELLED = {"typesafe102"}


# The rows a calibration is computed over


def upstream_confidence(values: list) -> float:
    """Upstream's `confidence` (engine.py at dcd2094): 1 - H(p) / ln K, clamped to [0, 1]."""
    k = len(values)
    if k <= 1:
        return 1.0
    entropy = -sum(x * math.log(x) for x in values if x > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(k)))


def rows_of(doc: dict, dataset=None) -> list:
    """One row per answered item with a valid distribution and an expected label: the distribution
    JevBench scores, the expected label, the argmax (the smallest label on a tie), whether it is
    right, the top label's probability, `confidence` and the group the cross-validation keeps
    together. A TypeSafe run needs its dataset, which holds the reference answers and the cases."""
    typesafe = doc["dataset"]["name"] == "typesafe102"
    if typesafe and dataset is None:
        raise ValueError("a TypeSafe run is calibrated against the dataset; pass it")
    rows = []
    for item in doc["items"]:
        if item["status"] != "answered" or not item.get("valid"):
            continue
        task = harness.task_of(item, dataset if typesafe else None)
        if task.expected is None:
            continue
        scored = jb_scoring.score_task(item["probabilities"] or {}, task)
        if not scored["valid"]:
            continue
        probs = {label: scored["probs"][label] for label in task.labels}
        answer = item.get("answer") or {}
        if isinstance(answer.get("confidence"), (int, float)):
            confidence, source = float(answer["confidence"]), "answer"
        else:
            confidence, source = upstream_confidence(list(probs.values())), "computed"
        group = dataset.gold[item["id"]]["group_id"] if typesafe else (item.get("group")
                                                                       or item["id"])
        rows.append({"id": item["id"], "type": item["type"], "options": len(task.labels),
                     "tier": item["tier"], "group": group, "probs": probs,
                     "expected": str(task.expected),
                     "predicted": scored["predicted"], "correct": bool(scored["correct"]),
                     "top": max(probs.values()), "confidence": confidence,
                     "confidence_source": source})
    return rows


# Metrics


def brier(row: dict) -> float:
    return sum((p - (label == row["expected"])) ** 2 for label, p in row["probs"].items())


def nll(row: dict) -> float:
    return -math.log(max(row["probs"][row["expected"]], NLL_FLOOR))


def ece(rows: list, key: str = "top") -> dict:
    """JevBench's top-label ECE over 10 equal-width bins, with the bins: lo, hi, n, mean
    confidence and accuracy."""
    return jb_metrics.ece_top_label([(row[key], row["correct"]) for row in rows], BINS)


def ece_value(rows: list) -> float:
    return ece(rows)["ece"] if rows else None


def auroc(rows: list, key: str):
    """The probability that a right answer's `key` exceeds a wrong one's, ties counting 1/2."""
    right = [row[key] for row in rows if row["correct"]]
    wrong = [row[key] for row in rows if not row["correct"]]
    if not right or not wrong:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in right for b in wrong)
    return wins / (len(right) * len(wrong))


def mean(values: list):
    values = list(values)
    return sum(values) / len(values) if values else None


def metrics(rows: list) -> dict:
    """Accuracy, the mean top-label probability and `confidence`, Brier, ECE, NLL and both
    AUROCs of a set of rows."""
    return {"n": len(rows), "accuracy": mean(row["correct"] for row in rows),
            "mean_top": mean(row["top"] for row in rows),
            "mean_confidence": mean(row["confidence"] for row in rows),
            "brier": mean(brier(row) for row in rows), "ece": ece_value(rows),
            "nll": mean(nll(row) for row in rows),
            "auroc_top": auroc(rows, "top"), "auroc_confidence": auroc(rows, "confidence")}


# Temperature scaling, SemIf's method (docs/CALIBRATION.md, benchmarks/calibrate.py at 23cf1f3)


def tempered(probs: dict, temperature: float) -> dict:
    """softmax(ln p / T), as p_k^(1/T) / sum_j p_j^(1/T) over p / max(p) so the largest term is 1
    and a zero stays zero."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"a temperature must be finite and positive, not {temperature!r}")
    top = max(probs.values())
    weights = {label: (p / top) ** (1.0 / temperature) for label, p in probs.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def at_temperature(row: dict, temperature: float) -> dict:
    """The row with its probabilities scaled. p^(1/T) keeps the order of the labels, so the answer
    and whether it is right are the row's own (the smoke test checks the argmax)."""
    probs = tempered(row["probs"], temperature)
    return {**row, "probs": probs, "top": max(probs.values()), "temperature": temperature}


def mean_nll(rows: list, temperature: float) -> float:
    return mean(-math.log(max(tempered(row["probs"], temperature)[row["expected"]], NLL_FLOOR))
                for row in rows)


def fit_temperature(rows: list, bounds: tuple = BOUNDS, iterations: int = ITERATIONS) -> float:
    """The T that minimises the mean NLL, by SemIf's golden-section search."""
    ratio = (math.sqrt(5) - 1) / 2
    low, high = bounds
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = mean_nll(rows, left), mean_nll(rows, right)
    for _ in range(iterations):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = mean_nll(rows, left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = mean_nll(rows, right)
    return (low + high) / 2


def fold_map(rows: list, folds: int = FOLDS, seed: int = SEED) -> dict:
    """{group: fold}: the sorted groups shuffled with `seed` and dealt round the folds, so every
    member of a group (a paraphrase pair, a TypeSafe case) is held out together."""
    groups = sorted({row["group"] for row in rows})
    random.Random(seed).shuffle(groups)
    return {group: index % folds for index, group in enumerate(groups)}


def out_of_fold(rows: list, folds: int = FOLDS, seed: int = SEED) -> tuple:
    """Every row scored with the T fitted on the folds that do not hold it, and those Ts."""
    fold_of = fold_map(rows, folds, seed)
    if len(fold_of) < 2:
        raise ValueError("out-of-fold scores need at least two groups")
    held, temperatures = {}, []
    for fold in range(folds):
        train = [row for row in rows if fold_of[row["group"]] != fold]
        test = [row for row in rows if fold_of[row["group"]] == fold]
        if not test:
            continue
        temperature = fit_temperature(train)
        temperatures.append(temperature)
        held.update({row["id"]: at_temperature(row, temperature) for row in test})
    return [held[row["id"]] for row in rows], temperatures


def bootstrap(rows: list, statistic, samples: int = SAMPLES, seed: int = SEED) -> list:
    """SemIf's 95% interval: the groups resampled with replacement `samples` times, the 2.5th and
    97.5th of the sorted statistics."""
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    groups = list(by_group.values())
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = []
        for _ in groups:
            draw.extend(groups[rng.randrange(len(groups))])
        values.append(statistic(draw))
    values.sort()
    return [values[int(0.025 * samples)], values[min(samples - 1, int(0.975 * samples))]]


def paired_bootstrap(rows: list, scaled: list, statistic, samples: int = SAMPLES,
                     seed: int = SEED) -> list:
    """The 95% interval of statistic(scaled) - statistic(rows) over the same resampled groups:
    `scaled[i]` is `rows[i]` at another temperature (SemIf's paired_bootstrap_delta)."""
    by_group = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row["group"], []).append(index)
    groups = list(by_group.values())
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = []
        for _ in groups:
            draw.extend(groups[rng.randrange(len(groups))])
        values.append(statistic([scaled[i] for i in draw]) - statistic([rows[i] for i in draw]))
    values.sort()
    return [values[int(0.025 * samples)], values[min(samples - 1, int(0.975 * samples))]]


def mean_brier(rows: list) -> float:
    return mean(brier(row) for row in rows)


def mean_of_nll(rows: list) -> float:
    return mean(nll(row) for row in rows)


def temperature_report(rows: list) -> dict:
    """SemIf's report for one workload: the T fitted on every row (the one a deployment would
    use), the fold Ts, and the ECE, Brier and NLL at T = 1 and out of fold, with the ECE's
    bootstrap intervals and whether they separate."""
    temperature = fit_temperature(rows)
    held, fold_temperatures = out_of_fold(rows)
    base_ci, held_ci = bootstrap(rows, ece_value), bootstrap(held, ece_value)
    return {"rows": len(rows), "groups": len({row["group"] for row in rows}),
            "temperature": temperature, "fold_temperatures": fold_temperatures,
            "accuracy": mean(row["correct"] for row in rows),
            "ece": ece_value(rows), "ece_ci": base_ci,
            "ece_out_of_fold": ece_value(held), "ece_out_of_fold_ci": held_ci,
            "separated": held_ci[1] < base_ci[0] or held_ci[0] > base_ci[1],
            "ece_change_ci": paired_bootstrap(rows, held, ece_value),
            "brier": mean_brier(rows), "brier_out_of_fold": mean_brier(held),
            "brier_change_ci": paired_bootstrap(rows, held, mean_brier),
            "nll": mean_of_nll(rows), "nll_out_of_fold": mean_of_nll(held),
            "nll_change_ci": paired_bootstrap(rows, held, mean_of_nll),
            "mean_top": mean(row["top"] for row in rows),
            "mean_top_out_of_fold": mean(row["top"] for row in held)}


def per_type_control(rows: list) -> list:
    """SemIf's negative control with the question types as workloads: out of fold, the ECE of each
    type under its own T and under one T fitted on every type, on the same folds, with the paired
    bootstrap interval of the difference (pooled minus own)."""
    fold_of = fold_map(rows)
    own, pooled = {}, {}
    for fold in range(FOLDS):
        train = [row for row in rows if fold_of[row["group"]] != fold]
        test = [row for row in rows if fold_of[row["group"]] == fold]
        if not test:
            continue
        everything = fit_temperature(train)
        pooled.update({row["id"]: at_temperature(row, everything) for row in test})
        for kind in TYPES:
            members = [row for row in train if row["type"] == kind]
            if members:
                temperature = fit_temperature(members)
                own.update({row["id"]: at_temperature(row, temperature) for row in test
                            if row["type"] == kind})
    out = []
    for kind in TYPES:
        members = [row for row in rows if row["type"] == kind and row["id"] in own]
        if not members:
            continue

        def delta(draw):
            return (ece_value([pooled[row["id"]] for row in draw])
                    - ece_value([own[row["id"]] for row in draw]))

        out.append({"type": kind, "rows": len(members),
                    "temperature": fit_temperature([row for row in rows if row["type"] == kind]),
                    "ece": ece_value(members),
                    "ece_own": ece_value([own[row["id"]] for row in members]),
                    "ece_pooled": ece_value([pooled[row["id"]] for row in members]),
                    "delta_ci": bootstrap(members, delta)})
    return out


# Tables


def runs_of(docs: list, model: str) -> list:
    """(dataset, server, doc) for every run of `model`, JevBench first and Swift first."""
    found = [(doc["dataset"]["name"], doc["server"]["name"], doc) for doc in docs
             if doc["model"] == model]
    return sorted(found, key=lambda run: (run[0] != "jevbench", run[0], run[1] != "swift", run[1]))


def interval(values: list) -> str:
    return f"{values[0]:.3f} to {values[1]:.3f}"


def fixed(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summary_table(runs: list) -> str:
    rows = []
    for dataset, server, rows_ in runs:
        found = metrics(rows_)
        rows.append([dataset, server, str(found["n"]), pct(found["accuracy"]),
                     fixed(found["mean_top"]), num(found["brier"]), num(found["ece"]),
                     fixed(found["nll"]), fixed(found["auroc_top"]),
                     fixed(found["auroc_confidence"]), fixed(found["mean_confidence"])])
    return markdown_table(["dataset", "server", "items", "accuracy", "mean p(top)", "Brier", "ECE",
                           "NLL", "AUROC p(top)", "AUROC confidence", "mean confidence"], rows)


def reliability_table(dataset: str, runs: list) -> str:
    """The 10 bins of one dataset, one column group per server."""
    header, columns = ["p(top)"], []
    for name, server, rows_ in runs:
        if name != dataset:
            continue
        header += [f"{server} n", f"{server} mean p", f"{server} accuracy"]
        columns.append(ece(rows_)["bins"])
    rows = []
    for index in range(BINS):
        row = [f"{index / BINS:.1f} to {(index + 1) / BINS:.1f}"]
        for bins in columns:
            entry = bins[index]
            row += [str(entry["n"]), fixed(entry["mean_confidence"]), pct(entry["accuracy"])]
        rows.append(row)
    return markdown_table(header, rows)


def type_table(runs: list) -> str:
    rows = []
    for dataset, server, rows_ in runs:
        for kind in TYPES:
            members = [row for row in rows_ if row["type"] == kind]
            if not members:
                continue
            found = metrics(members)
            rows.append([dataset, kind, server, str(found["n"]), pct(found["accuracy"]),
                         fixed(found["mean_top"]), fixed(found["mean_confidence"]),
                         num(found["ece"]), num(found["brier"]), fixed(found["auroc_confidence"])])
    return markdown_table(["dataset", "type", "server", "items", "accuracy", "mean p(top)",
                           "mean confidence", "ECE", "Brier", "AUROC confidence"], rows)


def tier_table(runs: list) -> str | None:
    """Each tier of a run with more than one (JevBench's easy, standard and hard), with the T
    fitted on that tier alone."""
    rows = []
    for dataset, server, rows_ in runs:
        tiers = list(dict.fromkeys(row["tier"] for row in rows_))
        if len(tiers) < 2:
            continue
        for tier in tiers:
            members = [row for row in rows_ if row["tier"] == tier]
            found = metrics(members)
            rows.append([dataset, tier, server, str(found["n"]), pct(found["accuracy"]),
                         fixed(found["mean_top"]), num(found["ece"]), num(found["brier"]),
                         fixed(found["nll"]), fitted_cell(members)])
    if not rows:
        return None
    return markdown_table(["dataset", "tier", "server", "items", "accuracy", "mean p(top)", "ECE",
                           "Brier", "NLL", "the tier's own T"], rows)


def option_table(runs: list, server: str) -> str:
    rows = []
    for dataset, name, rows_ in runs:
        if name != server:
            continue
        for kind in TYPES:
            for options in sorted({row["options"] for row in rows_ if row["type"] == kind}):
                members = [row for row in rows_ if row["type"] == kind
                           and row["options"] == options]
                found = metrics(members)
                rows.append([dataset, kind, str(options), str(found["n"]),
                             pct(found["accuracy"]), fixed(found["mean_top"]),
                             fixed(found["mean_confidence"]), num(found["ece"]),
                             fixed(found["auroc_confidence"])])
    return markdown_table(["dataset", "type", "options", "items", "accuracy", "mean p(top)",
                           "mean confidence", "ECE", "AUROC confidence"], rows)


def confidence_bin(value: float) -> int:
    return min(int(value * CONFIDENCE_BINS), CONFIDENCE_BINS - 1)


def confidence_table(runs: list, server: str) -> str:
    """Accuracy by the answer's `confidence`, per type, in bins 0.2 wide."""
    rows = []
    for dataset, name, rows_ in runs:
        if name != server:
            continue
        for index in range(CONFIDENCE_BINS):
            row = [dataset, f"{index / CONFIDENCE_BINS:.1f} to {(index + 1) / CONFIDENCE_BINS:.1f}"]
            for kind in TYPES:
                members = [r for r in rows_ if r["type"] == kind
                           and confidence_bin(r["confidence"]) == index]
                row += [str(len(members)),
                        pct(mean(r["correct"] for r in members)) if members else ""]
            rows.append(row)
    header = ["dataset", "confidence"]
    for kind in TYPES:
        header += [f"{kind} n", f"{kind} accuracy"]
    return markdown_table(header, rows)


def temperature_tables(runs: list) -> tuple:
    """SemIf's report per hard-labelled run, the paired changes out of fold, and the T fitted on
    a server's JevBench run applied to its soft-labelled runs."""
    fitted, rows, changes = {}, [], []
    for dataset, server, rows_ in runs:
        if dataset in SOFT_LABELLED:
            continue
        fit = temperature_report(rows_)
        if dataset == "jevbench":
            fitted[server] = fit["temperature"]
        rows.append([dataset, server, str(fit["rows"]), str(fit["groups"]),
                     temperature_cell(fit["temperature"]),
                     f"{min(fit['fold_temperatures']):.2f} to "
                     f"{max(fit['fold_temperatures']):.2f}",
                     f"{num(fit['ece'])} ({interval(fit['ece_ci'])})",
                     f"{num(fit['ece_out_of_fold'])} ({interval(fit['ece_out_of_fold_ci'])})",
                     "yes" if fit["separated"] else "no"])
        changes.append([dataset, server, interval(fit["ece_change_ci"]),
                        f"{num(fit['brier'])} to {num(fit['brier_out_of_fold'])}",
                        interval(fit["brier_change_ci"]),
                        f"{fixed(fit['nll'])} to {fixed(fit['nll_out_of_fold'])}",
                        interval(fit["nll_change_ci"]),
                        f"{fixed(fit['mean_top'])} to {fixed(fit['mean_top_out_of_fold'])}"])
    table = markdown_table(["dataset", "server", "items", "groups", "fitted T", "fold Ts",
                            "ECE at T = 1 (95%)", "ECE out of fold (95%)", "intervals separate"],
                           rows)
    change_table = markdown_table(["dataset", "server", "ECE change (95%)", "Brier",
                                   "Brier change (95%)", "NLL", "NLL change (95%)",
                                   "mean p(top)"], changes)
    transfer = []
    for dataset, server, rows_ in runs:
        if dataset not in SOFT_LABELLED or server not in fitted:
            continue
        scaled = [at_temperature(row, fitted[server]) for row in rows_]
        transfer.append([dataset, server, fixed(fitted[server], 2), str(len(rows_)),
                         f"{num(ece_value(rows_))} to {num(ece_value(scaled))}",
                         f"{num(mean(brier(r) for r in rows_))} to "
                         f"{num(mean(brier(r) for r in scaled))}",
                         f"{fixed(mean(nll(r) for r in rows_))} to "
                         f"{fixed(mean(nll(r) for r in scaled))}"])
    transfer_table = markdown_table(["dataset", "server", "JevBench's T", "rows",
                                     "ECE, T = 1 to JevBench's T", "Brier", "NLL"], transfer)
    return table, change_table, transfer_table if transfer else None


def temperature_cell(value: float) -> str:
    """A fitted T, marked when the search stopped at an end of its range."""
    at_end = min(abs(value - end) for end in BOUNDS) < 1e-3
    return f"{value:.2f}" + (" (the end of the range)" if at_end else "")


def fitted_cell(rows: list) -> str:
    """The T fitted on `rows`, or why there is none: when every answer is right the NLL only falls
    as T falls (and is 0 within a float's resolution well before the end of the range), and when
    every answer is wrong it only falls as T rises."""
    if all(row["correct"] for row in rows):
        return "none: every answer is right"
    if not any(row["correct"] for row in rows):
        return "none: every answer is wrong"
    return temperature_cell(fit_temperature(rows))


def control_table(runs: list) -> str:
    rows = []
    for dataset, server, rows_ in runs:
        if dataset in SOFT_LABELLED:
            continue
        for entry in per_type_control(rows_):
            rows.append([dataset, server, entry["type"], str(entry["rows"]),
                         temperature_cell(entry["temperature"]), num(entry["ece"]),
                         num(entry["ece_own"]), num(entry["ece_pooled"]),
                         interval(entry["delta_ci"])])
    return markdown_table(["dataset", "server", "type", "items", "the type's own T",
                           "ECE at T = 1", "ECE own T, out of fold", "ECE one T, out of fold",
                           "one minus own, 95%"], rows)


def render(results: Path, cache: Path, model: str = "openjev-0.1") -> str:
    """The calibration tables of docs/quality.md for `model`'s runs."""
    docs = harness.load_docs(report.result_files(results), cache)
    datasets = {}
    runs = []
    for dataset, server, doc in runs_of(docs, model):
        if dataset == "typesafe102":
            datasets.setdefault(dataset, harness.load_dataset(dataset, cache))
        runs.append((dataset, server, rows_of(doc, datasets.get(dataset))))
    if not runs:
        return f"no result files for {model} under {results}\n"
    out = ["### The raw reads", summary_table(runs)]
    for dataset in dict.fromkeys(name for name, _, _ in runs):
        out += [f"Reliability on {dataset}: the probability of the chosen answer against how often "
                "it is right, 10 equal-width bins:", reliability_table(dataset, runs)]
    tiers = tier_table(runs)
    if tiers:
        out += ["By tier, with the temperature each tier alone would be fitted to:", tiers]
    out += ["### Confidence and accuracy by question type", type_table(runs)]
    reference = "upstream" if any(server == "upstream" for _, server, _ in runs) else runs[0][1]
    out += [f"By option count ({reference}'s runs):", option_table(runs, reference),
            f"Accuracy by the answer's `confidence` ({reference}'s runs; a noul's is computed):",
            confidence_table(runs, reference)]
    table, changes, transfer = temperature_tables(runs)
    out += ["### Temperature scaling fitted offline", table,
            "Out of fold against T = 1 on the same items, with the paired 95% interval of each "
            "change over the same resampled groups (below 0 is better):", changes]
    if transfer:
        out += ["The T fitted on JevBench applied to the TypeSafe rows, scored against the "
                "reference's top option:", transfer]
    out += ["One T for every type against a T per type, out of fold on the same folds:",
            control_table(runs)]
    # prose wraps at 100 columns as report.py's does; tables and headings stay whole
    out = [block if block.startswith(("|", "#"))
           else textwrap.fill(block, 100, break_on_hyphens=False) for block in out]
    return "\n\n".join(out) + "\n"
