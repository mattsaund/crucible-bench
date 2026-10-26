#!/usr/bin/env python3
"""Check results posted to the crucible-bench board, and build board.json.

    python3 scripts/accept.py accept <event.json> <repo_root> <out_dir>
    python3 scripts/accept.py restore <event.json> <repo_root> <out_dir>
    python3 scripts/accept.py rebuild <repo_root>

`accept` reads the payload of a GitHub `issues` event for a new issue. An issue
whose body has no `<!-- crucible-bench -->` line is not a result. Otherwise the
first ```json block after that line is parsed and checked against the
crucible-bench-1 rules (see README.md), and then against Hugging Face: the
model file must be public there, with the SHA-256 and size the result gives. A
result that passes is stored as results/<issue number>.json and board.json is
rebuilt. Three files are always written to <out_dir> for the workflow to act
on: decision.json, comment.md, and issue.json, which is what the issue becomes
as it is closed (the result alone, or nothing).

`restore` reads the payload of an `edited` event. When the author of an issue
the board has closed edits it, <out_dir>/issue.json is written with what the
board left there, to be put back.

`rebuild` makes board.json again from results/*.json, skipping (and naming)
any file that no longer passes.

The board takes benchmark results and nothing else. Every string in a result
is an identifier of a fixed shape, the model's names come from Hugging Face,
and the issue a person opened is rewritten to the result alone (or emptied).
Its text is only ever parsed as JSON and checked: nothing in it is run, and
no message repeats any of it, so a comment the board writes can never be made
to say a word, mention or link anyone.

Standard library only, Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

SUITE = "crucible-bench-1"
MARKER = "<!-- crucible-bench -->"
FALLBACK_REPO_URL = "https://github.com/mattsaund/crucible-bench"
CRUCIBLE_ISSUES = "https://github.com/mattsaund/Crucible/issues"
REJECTED_LABEL = "not accepted"
NOT_ACCEPTED_TITLE = "Not accepted"
NOT_A_RESULT_TITLE = "Not a result"

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
BACKENDS = ("CPU", "CUDA", "Metal", "Vulkan", "ROCm", "SYCL")
MAX_GPUS = 64

# Every Crucible reads board.json, so integers stay within what JavaScript
# (and any 64-bit reader) holds exactly.
MAX_INT = 2**53 - 1

TOP_REQUIRED = ("schema", "suite", "size", "crucible", "llama", "date", "model", "settings", "benchmarks")
TOP_OPTIONAL = ("hardware", "speed", "seconds")
MODEL_REQUIRED = ("file", "sha256", "bytes", "repo")
MODEL_OPTIONAL = ("arch", "params", "quant")
SETTINGS_KEYS = ("context", "max_tokens", "temperature", "effort")
HARDWARE_KEYS = ("backend", "gpu_count", "memory_gb")
SPEED_KEYS = ("prompt_tps", "gen_tps")
BENCH_KEYS = ("n", "correct")
POSTED_KEYS = ("issue", "user", "url", "at")

VERSION_RE = re.compile(r"[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}")
LLAMA_RE = re.compile(r"b[0-9]{1,9}")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
# The other strings a result holds are names of a fixed shape, never free text:
# a GGUF file's name, an architecture as llama.cpp spells it, a quantization.
FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,194}\.gguf")
SPLIT_RE = re.compile(r"(.+)-([0-9]{5})-of-([0-9]{5})\.gguf")
ARCH_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,31}")
QUANT_RE = re.compile(
    r"(?:UD-)?(?:I?Q[1-8](?:_(?:[0-9]|K|M|S|L|P|NL|XS|XL|XXS|XXL)){0,3}|TQ[12]_0|B?F16|F32|MXFP4(?:_MOE)?)"
)
LOGIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}(?:\[bot\])?")
URL_RE = re.compile(r"https://[!-~]{1,300}")
TIME_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]{1,100}")
FENCE_RE = re.compile(r"(`{3,})[ \t]*json[ \t]*", re.IGNORECASE | re.ASCII)
RESULT_NAME_RE = re.compile(r"[1-9][0-9]{0,15}")

# Hugging Face's API, asked only for what the check needs: the name it gives
# the repository, whether that is public, and every file with its LFS record,
# which holds the file's SHA-256 (the LFS object's id) and its size.
HUB_API = "https://huggingface.co/api/models/"
HUB_QUERY = "?blobs=true&expand%5B%5D=siblings&expand%5B%5D=private&expand%5B%5D=disabled"
HUB_TIMEOUT = 30
# A longer reply is a repository of tens of thousands of files, which is no
# model anybody runs.
HUB_LIMIT = 32 * 1024 * 1024
# Seconds between tries when Hugging Face does not answer.
HUB_WAITS = (5, 15)


class Invalid(Exception):
    """A result that breaks a rule. The message is one sentence."""


class HubDown(Exception):
    """Hugging Face did not answer. Not the poster's doing, so the run fails and can be run again."""


def item_count(size: str, bench: str) -> int:
    """How many items a run of `bench` at `size` has."""
    rows = BENCHMARKS[bench][2]
    limit = SIZES[size]
    return rows if limit is None else min(limit, rows)


# Strict JSON ----------------------------------------------------------------


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise Invalid("The JSON repeats a key within one object.")
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
            raise Invalid(f"{where} has an unknown key.")
    for key in required:
        if key not in obj:
            raise Invalid(f"{where} is missing `{key}`.")


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
    _check_pattern(model["file"], FILE_RE, "`model.file` must be a GGUF file name of letters, digits and ._+- only.")
    # A split GGUF is run, and hashed, from its first part.
    split = SPLIT_RE.fullmatch(model["file"])
    if split and (split.group(2) != "00001" or split.group(3) == "00000"):
        raise Invalid("`model.file` must be the first part of a split GGUF, like x-00001-of-00003.gguf.")
    _check_pattern(model["sha256"], SHA256_RE, "`model.sha256` must be 64 lowercase hex characters.")
    if not (_is_int(model["bytes"]) and model["bytes"] > 0):
        raise Invalid("`model.bytes` must be a positive integer.")
    _check_pattern(model["repo"], REPO_RE, "`model.repo` must be a Hugging Face repository like owner/name.")
    if "arch" in model:
        _check_pattern(model["arch"], ARCH_RE, "`model.arch` must be an architecture name like llama or gpt-oss.")
    if "params" in model and not (_is_int(model["params"]) and model["params"] >= 0):
        raise Invalid("`model.params` must be a non-negative integer.")
    if "quant" in model:
        _check_pattern(model["quant"], QUANT_RE, "`model.quant` must be a quantization name like Q4_K_M or MXFP4.")


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
    backend = hardware.get("backend")
    if "backend" in hardware and not (isinstance(backend, str) and backend in BACKENDS):
        raise Invalid(f"`hardware.backend` must be {', '.join(BACKENDS[:-1])} or {BACKENDS[-1]}.")
    if "gpu_count" in hardware and not (_is_int(hardware["gpu_count"]) and 0 <= hardware["gpu_count"] <= MAX_GPUS):
        raise Invalid(f"`hardware.gpu_count` must be an integer from 0 to {MAX_GPUS}.")
    if "memory_gb" in hardware and not (_is_number(hardware["memory_gb"]) and hardware["memory_gb"] >= 0):
        raise Invalid("`hardware.memory_gb` must be a non-negative number.")


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
            raise Invalid("`benchmarks` has an unknown benchmark.")
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
    must not. Hugging Face is not asked here: see check_hub.
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


# Hugging Face ---------------------------------------------------------------
# The model must be a file anybody can fetch from Hugging Face, the very bytes
# that were run. That also makes its names Hugging Face's: the repository, the
# file and its size are checked there, not taken from whoever posted.


def _hub_get(url: str) -> tuple[int, bytes]:
    """One GET of Hugging Face's API: the status and the body, or status 0 when nothing answered."""
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "crucible-bench"})
    try:
        with urllib.request.urlopen(request, timeout=HUB_TIMEOUT) as response:
            return response.status, response.read(HUB_LIMIT + 1)
    except urllib.error.HTTPError as err:
        err.close()
        return err.code, b""
    except (OSError, http.client.HTTPException, ValueError):
        return 0, b""


def hub_repository(repo: str) -> dict[str, Any]:
    """What Hugging Face says about the model repository `repo`: its own name and its files.

    Raises Invalid when there is no such public repository, and HubDown when
    Hugging Face does not answer.
    """
    owner, name = repo.split("/")
    url = f"{HUB_API}{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}{HUB_QUERY}"
    for wait in (*HUB_WAITS, None):
        status, body = _hub_get(url)
        if status == 200:
            break
        # To a stranger, a private repository and one that was never there
        # both answer 401: either way, nobody can fetch the model from it.
        if 400 <= status < 500 and status not in (408, 429):
            raise Invalid("`model.repo` is not a public model on Hugging Face.")
        if wait is None:
            raise HubDown(f"Hugging Face did not answer for {repo} (status {status or 'none'}).")
        time.sleep(wait)
    if len(body) > HUB_LIMIT:
        raise Invalid("`model.repo` has more files than the board can check.")
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise HubDown(f"Hugging Face described {repo} in something that is not JSON.") from None
    if not (isinstance(data, dict) and isinstance(data.get("id"), str) and isinstance(data.get("siblings"), list)):
        raise HubDown(f"Hugging Face described {repo} in a shape the board does not know.")
    return data


def _hub_files(data: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """Every file of a repository kept in LFS, by its path there: its SHA-256 and size."""
    files: dict[str, tuple[str, int]] = {}
    for sibling in data["siblings"]:
        if not isinstance(sibling, dict):
            continue
        path, lfs = sibling.get("rfilename"), sibling.get("lfs")
        if (isinstance(path, str) and isinstance(lfs, dict) and isinstance(lfs.get("sha256"), str)
                and _is_int(lfs.get("size"))):
            files[path] = (lfs["sha256"].lower(), lfs["size"])
    return files


def _model_size(files: dict[str, tuple[str, int]], path: str, name: str) -> int | None:
    """The size of the model at `path`, all of its parts for a split GGUF; None when a part is missing."""
    split = SPLIT_RE.fullmatch(name)
    if not split:
        return files[path][1]
    folder, count = path[: len(path) - len(name)], int(split.group(3))
    parts = [f"{folder}{split.group(1)}-{n:05d}-of-{count:05d}.gguf" for n in range(1, count + 1)]
    if any(part not in files for part in parts):
        return None
    return sum(files[part][1] for part in parts)


def check_hub(model: dict[str, Any]) -> None:
    """Raise Invalid unless `model.repo` is a public model on Hugging Face holding
    `model.file`, with `model.sha256` as its SHA-256 and `model.bytes` as its size.

    A split GGUF is hashed from its first part, which is the file named, and
    its size is that of all its parts, which must sit beside the first.
    """
    data = hub_repository(model["repo"])
    # A name in other capitals, or one a repository had before it moved:
    # Hugging Face still answers for it, under the name it has now.
    if data["id"] != model["repo"]:
        raise Invalid("`model.repo` is not the name Hugging Face gives that repository.")
    if data.get("private") is True or data.get("disabled") is True:
        raise Invalid("`model.repo` is not a public model on Hugging Face.")
    files = _hub_files(data)
    name = model["file"]
    found = [path for path, (sha256, _size) in files.items()
             if path.rsplit("/", 1)[-1] == name and sha256 == model["sha256"]]
    if not found:
        raise Invalid("`model.repo` on Hugging Face has no `model.file` with that SHA-256.")
    sizes = [_model_size(files, path, name) for path in found]
    if model["bytes"] in sizes:
        return
    size = next((size for size in sizes if size is not None), None)
    if size is None:
        raise Invalid("`model.repo` on Hugging Face is missing a part of `model.file`.")
    whole = " in all its parts" if SPLIT_RE.fullmatch(name) else ""
    raise Invalid(f"`model.bytes` is {model['bytes']}, but `model.file` is {size} bytes{whole} on Hugging Face.")


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


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
    """The newest result for each dedupe key. Those with every skill come first, since
    an overall of fewer skills is not the same measure; then highest overall first
    (ties: lower id first)."""
    newest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for stored in results:
        key = dedupe_key(stored)
        if key not in newest or stored["posted"]["issue"] > newest[key]["posted"]["issue"]:
            newest[key] = stored
    entries = [board_entry(stored) for stored in newest.values()]
    entries.sort(key=lambda entry: (len(entry["skills"]) < len(SKILLS), -entry["overall"], entry["id"]))
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


# Issues -----------------------------------------------------------------------
# An issue ends up holding the result and nothing else: the board rewrites its
# title and body from the result once it has decided, and empties an issue that
# holds none. The workflow then closes and locks it.


def find_block(body: str) -> str | None:
    """The text of the first ```json block after the marker line.

    None when the body has no marker line (an issue that is not a result).
    Raises Invalid when the marker is there but no closed block follows it.
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


def result_title(result: dict[str, Any]) -> str:
    """A result's issue title, made only of what the result holds."""
    return f"Result: {result['model']['file']}, {result['size']}"


def result_body(result: dict[str, Any]) -> str:
    """A result's issue body: the marker line and the result, nothing else."""
    return f"{MARKER}\n```json\n{json.dumps(result, indent=2, sort_keys=True)}\n```\n"


def _closed(title: str, body: str, reason: str) -> dict[str, str]:
    """What an issue becomes once the board has decided, in the fields GitHub's API takes."""
    return {"title": title, "body": body, "state": "closed", "state_reason": reason}


def _issue_of(event: Any) -> dict[str, Any]:
    issue = event.get("issue") if isinstance(event, dict) else None
    if not (isinstance(issue, dict) and _is_int(issue.get("number")) and issue["number"] > 0):
        raise ValueError("the event has no issue number; is it an issues event?")
    return issue


def _login(user: object) -> str | None:
    """A GitHub user's login in lower case, as GitHub compares them."""
    if isinstance(user, dict) and isinstance(user.get("login"), str):
        return user["login"].lower()
    return None


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


def _decide(out_dir: Path, status: str, message: str, comment: str | None = None,
            issue: dict[str, str] | None = None) -> dict[str, str]:
    """Write what the workflow acts on. A file with nothing to say is left out."""
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = {"status": status, "message": message}
    _write_text(out_dir / "decision.json", json.dumps(decision, indent=2) + "\n")
    for name, content in (("comment.md", comment), ("issue.json", issue)):
        path = out_dir / name
        if content is None:
            path.unlink(missing_ok=True)
        else:
            _write_text(path, content if isinstance(content, str) else json.dumps(content, indent=2) + "\n")
    return decision


def _other_comment() -> str:
    return f"This board takes only results from Crucible's Bench tab. Anything else goes to {CRUCIBLE_ISSUES}.\n"


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
    """Decide on one newly opened issue, store the result if it passes, and write the decision.

    Raises HubDown when Hugging Face does not answer, so that the run fails
    and can be run again rather than turn a good result away.
    """
    issue = _issue_of(event)
    number = issue["number"]
    body = issue.get("body")
    repo_url, branch = _repo_link(event)

    try:
        block = find_block(body if isinstance(body, str) else "")
        if block is None:
            return _decide(out_dir, "other", f"No {MARKER} line, so this is not a result.",
                           _other_comment(), _closed(NOT_A_RESULT_TITLE, "", "not_planned"))
        result = parse_json(block, "The `json` code block")
        validate(result)
        check_hub(result["model"])
    except Invalid as err:
        reason = str(err)
        return _decide(out_dir, "rejected", reason, _rejected_comment(reason, repo_url),
                       _closed(NOT_ACCEPTED_TITLE, "", "not_planned"))

    stored = dict(result)
    stored["posted"] = {
        "issue": number,
        "user": issue["user"].get("login") if isinstance(issue.get("user"), dict) else None,
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
    return _decide(out_dir, "accepted", f"Accepted as results/{number}.json.", comment,
                   _closed(result_title(result), result_body(result), "completed"))


def restore_event(event: Any, repo_root: Path, out_dir: Path) -> dict[str, str]:
    """Undo an edit to an issue the board has closed: write the title and body it left there.

    Only an edit by the issue's own author is undone, and only once the board
    has closed the issue: until then the accept run is still deciding, and it
    rewrites the issue from what was opened, whatever was edited meanwhile.
    """
    issue = _issue_of(event)
    number = issue["number"]
    if issue.get("state") != "closed":
        return _decide(out_dir, "unchanged", "The issue is open, so the accept run has yet to rewrite it.")
    editor = _login(event.get("sender"))
    if editor is None or editor != _login(issue.get("user")):
        return _decide(out_dir, "unchanged", "Edited by someone other than its author.")
    try:
        stored = load_stored(repo_root / "results" / f"{number}.json")
    except Invalid:
        labels = issue.get("labels") if isinstance(issue.get("labels"), list) else []
        rejected = any(isinstance(label, dict) and label.get("name") == REJECTED_LABEL for label in labels)
        title = NOT_ACCEPTED_TITLE if rejected else NOT_A_RESULT_TITLE
        return _decide(out_dir, "restored", f"Emptied again, as {title.lower()}.",
                       issue={"title": title, "body": ""})
    result = {key: value for key, value in stored.items() if key != "posted"}
    return _decide(out_dir, "restored", f"Put back as results/{number}.json.",
                   issue={"title": result_title(result), "body": result_body(result)})


# Command line -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check results posted to the crucible-bench board, and build board.json.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (("accept", "check the result in a GitHub issues event for a new issue"),
                               ("restore", "put back an issue the board closed, after its author edited it")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("event", type=Path, help="the event payload, such as $GITHUB_EVENT_PATH")
        sub.add_argument("repo_root", type=Path, help="the board repository")
        sub.add_argument("out_dir", type=Path, help="where decision.json, comment.md and issue.json go")
    rebuild_cmd = commands.add_parser("rebuild", help="build board.json again from results/")
    rebuild_cmd.add_argument("repo_root", type=Path, help="the board repository")
    args = parser.parse_args(argv)

    if args.command in ("accept", "restore"):
        decide = accept_event if args.command == "accept" else restore_event
        try:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            decision = decide(event, args.repo_root, args.out_dir)
        except (OSError, ValueError, HubDown) as err:
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
