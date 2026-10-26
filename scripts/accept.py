#!/usr/bin/env python3
"""Check results posted to the crucible-bench board, and build board.json.

    python3 scripts/accept.py accept <event.json> <repo_root> <out_dir>
    python3 scripts/accept.py rebuild <repo_root>

`accept` reads the payload of a GitHub `issues` event. An issue whose body has
no `<!-- crucible-bench -->` line is ignored. Otherwise the first ```json block
after that line is parsed and checked against the crucible-bench-1 rules (see
README.md). A result that passes is stored as results/<issue number>.json and
board.json is rebuilt. <out_dir>/decision.json and <out_dir>/comment.md are
always written, for the workflow to act on.

`rebuild` makes board.json again from results/*.json, skipping (and naming)
any file that no longer passes.

Whoever opens an issue writes its text. That text is only ever parsed as JSON
and checked: nothing in it is run, and messages repeat only short names made
of safe characters, so a comment can never be made to mention or link anyone.

Standard library only, Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any

SUITE = "crucible-bench-1"
MARKER = "<!-- crucible-bench -->"
FALLBACK_REPO_URL = "https://github.com/mattsaund/crucible-bench"

# id: (name, skill, rows in the dataset split)
BENCHMARKS: dict[str, tuple[str, str, int]] = {
    "mmlu_pro": ("MMLU-Pro", "knowledge", 12032),
    "arc": ("ARC-Challenge", "reasoning", 1172),
    "gsm8k": ("GSM8K", "math", 1319),
    "math500": ("MATH-500", "math", 500),
    "humaneval": ("HumanEval", "code", 164),
    "mbpp": ("MBPP", "code", 257),
    "ifeval": ("IFEval", "instructions", 541),
    "hellaswag": ("HellaSwag", "commonsense", 10042),
    "truthfulqa": ("TruthfulQA", "truthfulness", 817),
}
SKILLS = ("knowledge", "reasoning", "math", "code", "instructions", "commonsense", "truthfulness")
# Items per benchmark: min(limit, rows), or every row when the limit is None.
SIZES: dict[str, int | None] = {"quick": 50, "standard": 200, "full": None}
EFFORTS = ("default", "low", "medium", "high")

# Every Crucible reads board.json, so integers stay within what JavaScript
# (and any 64-bit reader) holds exactly.
MAX_INT = 2**53 - 1

TOP_REQUIRED = ("schema", "suite", "size", "crucible", "llama", "date", "model", "settings", "benchmarks")
TOP_OPTIONAL = ("hardware", "speed", "seconds")
MODEL_REQUIRED = ("file", "sha256", "bytes")
MODEL_OPTIONAL = ("name", "arch", "params", "quant", "repo")
SETTINGS_KEYS = ("context", "max_tokens", "temperature", "effort")
HARDWARE_KEYS = ("backend", "gpus", "memory_gb")
SPEED_KEYS = ("prompt_tps", "gen_tps")
BENCH_KEYS = ("n", "correct")
POSTED_KEYS = ("issue", "user", "url", "at")

VERSION_RE = re.compile(r"[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}")
LLAMA_RE = re.compile(r"b[0-9]{1,9}")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
LOGIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}(?:\[bot\])?")
URL_RE = re.compile(r"https://[!-~]{1,300}")
TIME_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]{1,100}")
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,40}")
FENCE_RE = re.compile(r"(`{3,})[ \t]*json[ \t]*", re.IGNORECASE | re.ASCII)
RESULT_NAME_RE = re.compile(r"[1-9][0-9]{0,15}")


class Invalid(Exception):
    """A result that breaks a rule. The message is one sentence."""


def item_count(size: str, bench: str) -> int:
    """How many items a run of `bench` at `size` has."""
    rows = BENCHMARKS[bench][2]
    limit = SIZES[size]
    return rows if limit is None else min(limit, rows)


# Strict JSON ----------------------------------------------------------------


def _shown(key: object) -> str:
    """The key in backticks after a space, when it is safe to repeat; else nothing."""
    if isinstance(key, str) and SAFE_NAME_RE.fullmatch(key):
        return f" `{key}`"
    return ""


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            shown = _shown(key)
            raise Invalid(f"The JSON repeats {'the key' + shown if shown else 'a key'} within one object.")
        obj[key] = value
    return obj


def _no_constant(name: str) -> Any:
    raise Invalid(f"The JSON uses {name}, which is not a JSON number.")


def _parse_int(text: str) -> int:
    if len(text.lstrip("-")) > 16:
        raise Invalid("The JSON has a number that is too large.")
    return int(text)


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise Invalid("The JSON has a number that is too large.")
    return value


def parse_json(text: str, what: str) -> Any:
    """Parse strict JSON: no NaN or Infinity, no repeated keys, no huge numbers."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_no_constant,
            parse_int=_parse_int,
            parse_float=_parse_float,
        )
    except Invalid:
        raise
    except json.JSONDecodeError as err:
        raise Invalid(f"{what} is not valid JSON ({err.msg} at line {err.lineno}, column {err.colno}).") from None
    except (ValueError, RecursionError):
        raise Invalid(f"{what} is not valid JSON.") from None


# The rules ------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return type(value) is int and -MAX_INT <= value <= MAX_INT


def _is_number(value: object) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    return _is_int(value)


def _check_object(value: object, where: str) -> None:
    if not isinstance(value, dict):
        raise Invalid(f"{where} must be a JSON object.")


def _check_keys(obj: dict[str, Any], where: str, allowed: tuple[str, ...], required: tuple[str, ...]) -> None:
    for key in obj:
        if key not in allowed:
            raise Invalid(f"{where} has an unknown key{_shown(key)}.")
    for key in required:
        if key not in obj:
            raise Invalid(f"{where} is missing `{key}`.")


def _check_text(value: object, where: str, limit: int) -> None:
    if not isinstance(value, str):
        raise Invalid(f"{where} must be text.")
    if len(value) > limit:
        raise Invalid(f"{where} is longer than {limit} characters.")
    # Control characters, and halves of surrogate pairs that no UTF-8 text can hold.
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in value):
        raise Invalid(f"{where} contains a character that is not allowed.")


def _check_pattern(value: object, pattern: re.Pattern[str], message: str) -> None:
    if not (isinstance(value, str) and pattern.fullmatch(value)):
        raise Invalid(message)


def _check_date(value: object) -> None:
    message = "`date` must be a date like 2026-10-26."
    _check_pattern(value, DATE_RE, message)
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        raise Invalid(message) from None


def _check_model(model: object) -> None:
    _check_object(model, "`model`")
    _check_keys(model, "`model`", MODEL_REQUIRED + MODEL_OPTIONAL, MODEL_REQUIRED)
    file = model["file"]
    _check_text(file, "`model.file`", 200)
    if not file or "/" in file or "\\" in file:
        raise Invalid("`model.file` must be a bare file name, with no slashes.")
    _check_pattern(model["sha256"], SHA256_RE, "`model.sha256` must be 64 lowercase hex characters.")
    if not (_is_int(model["bytes"]) and model["bytes"] > 0):
        raise Invalid("`model.bytes` must be a positive integer.")
    for key in ("name", "arch"):
        if key in model:
            _check_text(model[key], f"`model.{key}`", 100)
    if "params" in model and not (_is_int(model["params"]) and model["params"] >= 0):
        raise Invalid("`model.params` must be a non-negative integer.")
    if "quant" in model:
        _check_text(model["quant"], "`model.quant`", 32)
    if "repo" in model:
        _check_pattern(model["repo"], REPO_RE, "`model.repo` must look like owner/name.")


def _check_settings(settings: object) -> None:
    _check_object(settings, "`settings`")
    _check_keys(settings, "`settings`", SETTINGS_KEYS, SETTINGS_KEYS)
    for key in ("context", "max_tokens"):
        if not _is_int(settings[key]):
            raise Invalid(f"`settings.{key}` must be an integer.")
    if not _is_number(settings["temperature"]):
        raise Invalid("`settings.temperature` must be a number.")
    effort = settings["effort"]
    if not (isinstance(effort, str) and effort in EFFORTS):
        raise Invalid("`settings.effort` must be default, low, medium or high.")


def _check_hardware(hardware: object) -> None:
    _check_object(hardware, "`hardware`")
    _check_keys(hardware, "`hardware`", HARDWARE_KEYS, ())
    if "backend" in hardware:
        _check_text(hardware["backend"], "`hardware.backend`", 20)
    if "gpus" in hardware:
        gpus = hardware["gpus"]
        if not (isinstance(gpus, list) and len(gpus) <= 8):
            raise Invalid("`hardware.gpus` must be a list of at most 8 names.")
        for gpu in gpus:
            _check_text(gpu, "A name in `hardware.gpus`", 80)
    if "memory_gb" in hardware and not _is_number(hardware["memory_gb"]):
        raise Invalid("`hardware.memory_gb` must be a number.")


def _check_speed(speed: object) -> None:
    _check_object(speed, "`speed`")
    _check_keys(speed, "`speed`", SPEED_KEYS, ())
    for key in SPEED_KEYS:
        if key in speed and not (_is_number(speed[key]) and speed[key] >= 0):
            raise Invalid(f"`speed.{key}` must be a non-negative number.")


def _check_benchmarks(benchmarks: object, size: str) -> None:
    _check_object(benchmarks, "`benchmarks`")
    for bench in benchmarks:
        if bench not in BENCHMARKS:
            raise Invalid(f"`benchmarks` has an unknown benchmark{_shown(bench)}.")
    if not benchmarks:
        raise Invalid("`benchmarks` must hold at least one benchmark.")
    for bench, (name, _skill, _rows) in BENCHMARKS.items():
        if bench not in benchmarks:
            continue
        where = f"`benchmarks.{bench}`"
        entry = benchmarks[bench]
        _check_object(entry, where)
        _check_keys(entry, where, BENCH_KEYS, BENCH_KEYS)
        n, correct = entry["n"], entry["correct"]
        if not _is_int(n):
            raise Invalid(f"`benchmarks.{bench}.n` must be an integer.")
        expected = item_count(size, bench)
        if n != expected:
            raise Invalid(f"`benchmarks.{bench}.n` is {n}, but a {size} run of {name} has {expected} items.")
        if not (_is_int(correct) and 0 <= correct <= n):
            raise Invalid(f"`benchmarks.{bench}.correct` must be an integer from 0 to {n}.")


def _check_posted(posted: object) -> None:
    _check_object(posted, "`posted`")
    _check_keys(posted, "`posted`", POSTED_KEYS, POSTED_KEYS)
    if not (_is_int(posted["issue"]) and posted["issue"] > 0):
        raise Invalid("`posted.issue` must be a positive integer.")
    _check_pattern(posted["user"], LOGIN_RE, "`posted.user` must be a GitHub login.")
    _check_pattern(posted["url"], URL_RE, "`posted.url` must be an https URL.")
    _check_pattern(posted["at"], TIME_RE, "`posted.at` must be a UTC time like 2026-10-26T12:00:00Z.")


def validate(result: object, *, stored: bool = False) -> None:
    """Raise Invalid, with one sentence, on the first rule that `result` breaks.

    A stored result (results/<n>.json) must also carry `posted`; a posted one
    must not.
    """
    _check_object(result, "The result")
    extra = ("posted",) if stored else ()
    _check_keys(result, "The result", TOP_REQUIRED + TOP_OPTIONAL + extra, TOP_REQUIRED + extra)
    if not (_is_int(result["schema"]) and result["schema"] == 1):
        raise Invalid("`schema` must be 1.")
    if result["suite"] != SUITE:
        raise Invalid(f"`suite` must be {SUITE}.")
    size = result["size"]
    if not (isinstance(size, str) and size in SIZES):
        raise Invalid("`size` must be quick, standard or full.")
    _check_pattern(result["crucible"], VERSION_RE, "`crucible` must be a version like 1.2.3.")
    _check_pattern(result["llama"], LLAMA_RE, "`llama` must be a llama.cpp build like b12345.")
    _check_date(result["date"])
    _check_model(result["model"])
    _check_settings(result["settings"])
    if "hardware" in result:
        _check_hardware(result["hardware"])
    if "speed" in result:
        _check_speed(result["speed"])
    if "seconds" in result and not (_is_int(result["seconds"]) and result["seconds"] >= 0):
        raise Invalid("`seconds` must be a non-negative integer.")
    _check_benchmarks(result["benchmarks"], size)
    if stored:
        _check_posted(result["posted"])


# Scores ---------------------------------------------------------------------
# Worked out exactly with fractions and rounded only at the end, so the same
# result gives the same numbers on every machine.


def exact_scores(result: dict[str, Any]) -> dict[str, Fraction]:
    """correct / n for each benchmark that was run."""
    benchmarks = result["benchmarks"]
    return {
        bench: Fraction(benchmarks[bench]["correct"], benchmarks[bench]["n"])
        for bench in BENCHMARKS
        if bench in benchmarks
    }


def exact_skills(scores: dict[str, Fraction]) -> dict[str, Fraction]:
    """Each skill is the mean of its benchmarks that were run; a skill with none is left out."""
    skills: dict[str, Fraction] = {}
    for skill in SKILLS:
        parts = [score for bench, score in scores.items() if BENCHMARKS[bench][1] == skill]
        if parts:
            skills[skill] = sum(parts, Fraction(0)) / len(parts)
    return skills


def exact_overall(skills: dict[str, Fraction]) -> Fraction:
    """The mean of the skills present."""
    return sum(skills.values(), Fraction(0)) / len(skills)


def round4(value: Fraction) -> float:
    """Round to four decimal places, halves to even."""
    return round(value * 10000) / 10000


def _percent(value: Fraction) -> str:
    tenths = round(value * 1000)
    return f"{tenths // 10}.{tenths % 10}%"


# The board ------------------------------------------------------------------


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def load_stored(path: Path) -> dict[str, Any]:
    """Read and check one results/<n>.json."""
    if not RESULT_NAME_RE.fullmatch(path.stem) or path.suffix != ".json":
        raise Invalid("The file name is not an issue number.")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise Invalid("The file cannot be read as UTF-8 text.") from None
    stored = parse_json(text, "The file")
    validate(stored, stored=True)
    if stored["posted"]["issue"] != int(path.stem):
        raise Invalid("`posted.issue` does not match the file name.")
    return stored


def load_results(repo_root: Path) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Every stored result that still passes, and (file, reason) for each one that does not."""
    folder = repo_root / "results"
    paths = sorted(folder.glob("*.json"), key=lambda p: (len(p.name), p.name)) if folder.is_dir() else []
    results: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for path in paths:
        try:
            results.append(load_stored(path))
        except Invalid as err:
            skipped.append((f"results/{path.name}", str(err)))
    return results, skipped


def dedupe_key(stored: dict[str, Any]) -> tuple[str, str, str]:
    """One board entry per person, model file and size. GitHub logins ignore case."""
    return (stored["posted"]["user"].lower(), stored["model"]["sha256"], stored["size"])


def board_entry(stored: dict[str, Any]) -> dict[str, Any]:
    """A stored result plus id, scores, skills and overall."""
    scores = exact_scores(stored)
    skills = exact_skills(scores)
    entry = dict(stored)
    entry["id"] = stored["posted"]["issue"]
    entry["scores"] = {bench: round4(score) for bench, score in scores.items()}
    entry["skills"] = {skill: round4(value) for skill, value in skills.items()}
    entry["overall"] = round4(exact_overall(skills))
    return entry


def board_entries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The newest result for each dedupe key, highest overall first (ties: lower id first)."""
    newest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for stored in results:
        key = dedupe_key(stored)
        if key not in newest or stored["posted"]["issue"] > newest[key]["posted"]["issue"]:
            newest[key] = stored
    entries = [board_entry(stored) for stored in newest.values()]
    entries.sort(key=lambda entry: (-entry["overall"], entry["id"]))
    return entries


def write_board(repo_root: Path, entries: list[dict[str, Any]], now: str | None = None) -> bool:
    """Write board.json when its entries changed. `updated` says when they last did."""
    path = repo_root / "board.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if (
        isinstance(old, dict)
        and set(old) == {"suite", "updated", "entries"}
        and isinstance(old["updated"], str)
        and old["suite"] == SUITE
        and old["entries"] == entries
    ):
        return False
    _write_json(path, {"suite": SUITE, "updated": now or utc_now(), "entries": entries})
    return True


def rebuild(repo_root: Path, now: str | None = None) -> tuple[list[dict[str, Any]], list[tuple[str, str]], bool]:
    """Build board.json from results/. Returns the entries, the skipped files and whether it changed."""
    results, skipped = load_results(repo_root)
    entries = board_entries(results)
    return entries, skipped, write_board(repo_root, entries, now)


# Accepting an issue -----------------------------------------------------------


def find_block(body: str) -> str | None:
    """The text of the first ```json block after the marker line.

    None when the body has no marker line (an ordinary issue). Raises Invalid
    when the marker is there but no closed block follows it.
    """
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip() == MARKER), None)
    if start is None:
        return None
    for i in range(start + 1, len(lines)):
        opening = FENCE_RE.fullmatch(lines[i].strip())
        if not opening:
            continue
        fence = opening.group(1)
        for j in range(i + 1, len(lines)):
            closing = lines[j].strip()
            if closing.startswith(fence) and set(closing) == {"`"}:
                return "\n".join(lines[i + 1 : j])
        raise Invalid("The `json` code block after the marker line is never closed.")
    raise Invalid(f"There is no `json` code block after the `{MARKER}` line.")


def _repo_link(event: dict[str, Any]) -> tuple[str, str]:
    """The repository's web address and default branch, from GitHub's part of the event."""
    url, branch = FALLBACK_REPO_URL, "main"
    repo = event.get("repository")
    if isinstance(repo, dict):
        if isinstance(repo.get("html_url"), str) and URL_RE.fullmatch(repo["html_url"]):
            url = repo["html_url"].rstrip("/")
        if isinstance(repo.get("default_branch"), str) and BRANCH_RE.fullmatch(repo["default_branch"]):
            branch = repo["default_branch"]
    return url, branch


def _decide(out_dir: Path, status: str, message: str, comment: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = {"status": status, "message": message}
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "comment.md").write_text(comment, encoding="utf-8", newline="\n")
    return decision


def _rejected_comment(reason: str, repo_url: str) -> str:
    return (
        "Thanks for posting. This result was not added to the board:\n"
        "\n"
        f"> {reason}\n"
        "\n"
        f"The [README]({repo_url}#readme) says what a result has to look like.\n"
    )


def _accepted_comment(
    number: int, result: dict[str, Any], older: list[int], newer: list[int], repo_url: str, branch: str
) -> str:
    skills = exact_skills(exact_scores(result))
    link = f"[results/{number}.json]({repo_url}/blob/{branch}/results/{number}.json)"
    if newer:
        lines = [
            f"Thanks, your result is saved as {link}.",
            f"Your newer result for this model file and size, #{newer[-1]}, stays on the board.",
        ]
    else:
        lines = [f"Thanks, your result is on the board: {link}."]
        if older:
            lines.append(f"It replaces your earlier result for this model file and size, #{older[-1]}.")
    lines += ["", "| Skill | Score |", "| :-- | --: |"]
    lines += [f"| {skill.capitalize()} | {_percent(value)} |" for skill, value in skills.items()]
    lines.append(f"| **Overall** | **{_percent(exact_overall(skills))}** |")
    return "\n".join(lines) + "\n"


def accept_event(event: Any, repo_root: Path, out_dir: Path, now: str | None = None) -> dict[str, str]:
    """Decide on one `issues` event, store the result if it passes, and write the decision."""
    issue = event.get("issue") if isinstance(event, dict) else None
    if not (isinstance(issue, dict) and _is_int(issue.get("number")) and issue["number"] > 0):
        raise ValueError("the event has no issue number; is it an issues event?")
    number = issue["number"]
    body = issue.get("body")
    repo_url, branch = _repo_link(event)

    try:
        block = find_block(body if isinstance(body, str) else "")
        if block is None:
            return _decide(out_dir, "ignored", f"No {MARKER} line, so this is an ordinary issue.", "")
        result = parse_json(block, "The `json` code block")
        validate(result)
    except Invalid as err:
        reason = str(err)
        return _decide(out_dir, "rejected", reason, _rejected_comment(reason, repo_url))

    user = issue.get("user")
    stored = dict(result)
    stored["posted"] = {
        "issue": number,
        "user": user.get("login") if isinstance(user, dict) else None,
        "url": issue.get("html_url"),
        "at": issue.get("created_at"),
    }
    try:
        validate(stored, stored=True)
    except Invalid as err:
        # Only GitHub's own fields were added, so this is not the poster's doing.
        raise ValueError(f"the issue's own details do not check out: {err}") from None

    _write_json(repo_root / "results" / f"{number}.json", stored)
    results, skipped = load_results(repo_root)
    for name, reason in skipped:
        print(f"skipped {name}: {reason}", file=sys.stderr)
    write_board(repo_root, board_entries(results), now)

    key = dedupe_key(stored)
    same = sorted(r["posted"]["issue"] for r in results if dedupe_key(r) == key and r["posted"]["issue"] != number)
    older = [i for i in same if i < number]
    newer = [i for i in same if i > number]
    comment = _accepted_comment(number, result, older, newer, repo_url, branch)
    return _decide(out_dir, "accepted", f"Accepted as results/{number}.json.", comment)


# Command line -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check results posted to the crucible-bench board, and build board.json.")
    commands = parser.add_subparsers(dest="command", required=True)
    accept_cmd = commands.add_parser("accept", help="check the result in a GitHub issues event")
    accept_cmd.add_argument("event", type=Path, help="the event payload, such as $GITHUB_EVENT_PATH")
    accept_cmd.add_argument("repo_root", type=Path, help="the board repository")
    accept_cmd.add_argument("out_dir", type=Path, help="where decision.json and comment.md go")
    rebuild_cmd = commands.add_parser("rebuild", help="build board.json again from results/")
    rebuild_cmd.add_argument("repo_root", type=Path, help="the board repository")
    args = parser.parse_args(argv)

    if args.command == "accept":
        try:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            decision = accept_event(event, args.repo_root, args.out_dir)
        except (OSError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(f"{decision['status']}: {decision['message']}")
        return 0

    entries, skipped, changed = rebuild(args.repo_root)
    for name, reason in skipped:
        print(f"skipped {name}: {reason}")
    count = f"{len(entries)} {'entry' if len(entries) == 1 else 'entries'}"
    print(f"board.json: {count}, {'rewritten' if changed else 'unchanged'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
