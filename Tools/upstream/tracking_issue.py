#!/usr/bin/env python3
"""Opens or updates the one issue that tracks what upstream changed since the pins (issue #66), from
a review that review.py --json saved. The Upstream review workflow runs it after each review; it is
the only part of the review that writes anything.

    python3 Tools/upstream/tracking_issue.py --repo OWNER/NAME REVIEW.json [--gh gh] [--dry-run]

The tracking issue is the issue labelled area/ci whose body starts with MARKER, open or closed.

- The review found nothing to look at (every pin it follows at its project's head, no error, no
  disagreement between pins): nothing changes, and an open tracking issue stays as it is.
- No tracking issue exists: it opens one with the review as its body.
- The tracking issue's body records the review's fingerprint: nothing changes. A closed issue with
  that fingerprint was closed by a maintainer who reviewed exactly this state, so it stays closed.
- Otherwise it replaces the body with the new review, reopens the issue if it was closed, and
  comments with what moved since the body was last written, so that watchers are notified.

It refuses a review in which every project failed, since that says the run is broken, not upstream:
the workflow run fails instead of filling the issue with errors. --dry-run lists the issues and
prints what it would do without doing it. Standard library only, Python 3.9 or later.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import review as review_script  # noqa: E402

LABEL = "area/ci"
TITLE = "Upstream changes since the pinned revisions"
MARKER = "<!-- upstream-review"
BODY_LIMIT = 65536  # GitHub's limit on an issue body, in characters
WORKFLOW = ".github/workflows/upstream-review.yml"


class TrackingError(Exception):
    pass


def header_of(body):
    """(fingerprint, state) that a tracking issue's body records, each None when absent."""
    body = body or ""
    fingerprint = re.search(r"<!-- upstream-review fingerprint=([0-9a-f]+) -->", body)
    state = re.search(r"<!-- upstream-review-state (\{.*?\}) -->", body, re.S)
    try:
        state = json.loads(state.group(1)) if state else None
    except ValueError:
        state = None
    return (fingerprint.group(1) if fingerprint else None), state


def tracking_issue(issues):
    """The tracking issue among `issues` (gh issue list --json number,state,body), or None: the
    oldest open one, else the newest closed one."""
    marked = [i for i in issues if (i.get("body") or "").lstrip().startswith(MARKER)]
    opened = sorted((i for i in marked if i["state"].upper() == "OPEN"), key=lambda i: i["number"])
    if opened:
        return opened[0]
    closed = sorted(marked, key=lambda i: i["number"], reverse=True)
    return closed[0] if closed else None


def body_of(review, repo, run_url=None):
    """The issue body for a review: the header comments, the review, and a footer that says where
    the body comes from. The review is cut at a line to fit GitHub's limit."""
    state = review_script.state_of(review)
    # An error message could hold "-->", which would end the comment; JSON can spell it escaped.
    encoded = json.dumps(state, sort_keys=True).replace("-->", "--\\u003e")
    header = (f"<!-- upstream-review fingerprint={review['fingerprint']} -->\n"
              f"<!-- upstream-review-state {encoded} -->\n")
    workflow = f"https://github.com/{repo}/blob/main/{WORKFLOW}"
    log = f"https://github.com/{repo}/blob/main/docs/upstream-log.md"
    footer = ("\n---\n\nThe [Upstream review workflow](" + workflow + ") writes this issue from "
              "`Tools/upstream/review.py`" + (f" ([this run]({run_url}))" if run_url else "")
              + ". [docs/upstream-log.md](" + log + ") says what a review checks and what moving a "
              "pin takes. Close the issue once the review note is written or the pins have moved; "
              "the workflow reopens it when upstream moves again.\n")
    markdown = review_script.render(review)
    room = BODY_LIMIT - len(header) - len(footer) - 200
    if len(markdown) > room:
        cut = markdown.rfind("\n", 0, room)
        markdown = (markdown[:cut if cut > 0 else room] + "\n\nThe review was cut to fit an issue "
                    "body. The workflow run's summary and its `upstream-review` artifact hold all "
                    "of it.\n")
    return header + markdown + footer


def changes(old_state, new_state):
    """Sentences naming each project whose state differs between two reviews."""
    if not old_state:
        return ["The previous body recorded no state; the review below replaces it."]
    lines = []
    for name in sorted(set(old_state) | set(new_state)):
        before, after = old_state.get(name), new_state.get(name)
        if before == after:
            continue
        if before is None:
            lines.append(f"- {name}: now reviewed, `{after}`.")
        elif after is None:
            lines.append(f"- {name}: no longer reviewed (was `{before}`).")
        else:
            lines.append(f"- {name}: was `{before}`, now `{after}`.")
    return lines or ["- Nothing a project records changed; the review's other details did."]


def plan(review, issues, repo, run_url=None):
    """The actions that bring the tracking issue up to date: a list of (action, number, text),
    where action is "create", "edit", "reopen" or "comment" and number is None for "create"."""
    projects = review.get("projects", [])
    if projects and all(p.get("error") for p in projects):
        raise TrackingError("every project failed to be read; the run is broken, not upstream, so "
                            "the tracking issue is left alone")
    if not review.get("attention"):
        return []
    issue = tracking_issue(issues)
    body = body_of(review, repo, run_url)
    if issue is None:
        return [("create", None, body)]
    fingerprint, state = header_of(issue.get("body"))
    if fingerprint == review["fingerprint"]:
        return []
    actions = []
    reopened = issue["state"].upper() != "OPEN"
    if reopened:
        actions.append(("reopen", issue["number"], ""))
    actions.append(("edit", issue["number"], body))
    when = review["reviewed_at"][:10]
    intro = (f"Reopened by the review of {when}, which differs from the one this issue was "
             "closed with." if reopened else f"Updated by the review of {when}.")
    if run_url:
        intro += f" [The run]({run_url}) holds the review in its summary."
    comment = "\n".join([intro, "", "What changed since the body was last written:", ""]
                        + changes(state, review_script.state_of(review))) + "\n"
    actions.append(("comment", issue["number"], comment))
    return actions


def command_for(gh, repo, action, number):
    if action == "create":
        return [gh, "issue", "create", "--repo", repo, "--title", TITLE, "--label", LABEL,
                "--body-file", "-"]
    if action == "edit":
        return [gh, "issue", "edit", str(number), "--repo", repo, "--body-file", "-"]
    if action == "reopen":
        return [gh, "issue", "reopen", str(number), "--repo", repo]
    if action == "comment":
        return [gh, "issue", "comment", str(number), "--repo", repo, "--body-file", "-"]
    raise ValueError(action)


def list_issues(gh, repo):
    result = subprocess.run([gh, "issue", "list", "--repo", repo, "--label", LABEL, "--state",
                             "all", "--limit", "1000", "--json", "number,state,title,body,url"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise TrackingError(f"{gh} issue list: {result.stderr.strip()}")
    return json.loads(result.stdout)


def apply(actions, gh, repo, dry_run, run=subprocess.run, out=print):
    """Runs the actions with the GitHub CLI, or prints them under --dry-run."""
    for action, number, text in actions:
        command = command_for(gh, repo, action, number)
        if dry_run:
            out("would run: " + shlex.join(command))
            if text:
                lines = text.splitlines()
                out(f"  with {len(text):,} characters on standard input, beginning:")
                for line in lines[:12]:
                    out("  | " + line)
            continue
        result = run(command, input=text, capture_output=True, text=True)
        if result.returncode != 0:
            raise TrackingError(f"{' '.join(command[:4])}: {result.stderr.strip()}")
        out(f"{action} {number or ''}: {(result.stdout or '').strip()}".strip())


def describe(actions, dry_run=False):
    if not actions:
        return "Nothing to change on the tracking issue."
    names = [a for a, _, _ in actions]
    number = actions[0][1]
    if names == ["create"]:
        done = "would open the tracking issue" if dry_run else "opened the tracking issue"
    elif "reopen" in names:
        done = (f"would reopen and update #{number}" if dry_run
                else f"reopened and updated #{number}")
    else:
        done = f"would update #{number}" if dry_run else f"updated #{number}"
    return ("Dry run: " if dry_run else "Done: ") + done + "."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("review", type=Path, help="the JSON that review.py --json printed")
    parser.add_argument("--repo", required=True, help="the repository of the tracking issue")
    parser.add_argument("--gh", default="ghp", help="the GitHub CLI (default ghp; gh in CI)")
    parser.add_argument("--run-url", help="the workflow run, linked from the body and comment")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the issues and print what would change, changing nothing")
    args = parser.parse_args(argv)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    try:
        issues = list_issues(args.gh, args.repo)
        issue = tracking_issue(issues)
        actions = plan(review, issues, args.repo, args.run_url)
        if issue is not None:
            print(f"Tracking issue: #{issue['number']} ({issue['state'].lower()}), {issue['url']}")
        else:
            print("No tracking issue yet.")
        if not review.get("attention"):
            print("The review found nothing new since the pins.")
        apply(actions, args.gh, args.repo, args.dry_run)
        print(describe(actions, args.dry_run))
    except TrackingError as error:
        print(f"tracking_issue.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
