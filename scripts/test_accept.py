"""Tests for accept.py. From the repository root:

    python3 -m unittest discover -s scripts -p 'test_*.py'
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import accept

SCRIPT = Path(__file__).resolve().parent / "accept.py"
NOW = "2026-10-26T12:00:00Z"
LATER = "2026-10-27T08:30:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64
REMOVE = object()

QUICK = dict.fromkeys(accept.BENCHMARKS, 50)
STANDARD = {
    "mmlu_pro": 200, "arc": 200, "gsm8k": 200, "math500": 200, "humaneval": 164,
    "mbpp": 200, "ifeval": 200, "hellaswag": 200, "truthfulqa": 200,
}
FULL = {
    "mmlu_pro": 12032, "arc": 1172, "gsm8k": 1319, "math500": 500, "humaneval": 164,
    "mbpp": 257, "ifeval": 541, "hellaswag": 10042, "truthfulqa": 817,
}


def valid_result(**top: Any) -> dict[str, Any]:
    """A result that passes, with every optional field filled in."""
    result: dict[str, Any] = {
        "schema": 1,
        "suite": "crucible-bench-1",
        "size": "quick",
        "crucible": "0.9.5",
        "llama": "b11846",
        "date": "2026-10-26",
        "model": {
            "file": "flint-Q6_K.gguf",
            "sha256": SHA_A,
            "bytes": 32720880480,
            "name": "Recursive Flint",
            "arch": "flint",
            "params": 39800000000,
            "quant": "Q6_K",
            "repo": "unsloth/flint-GGUF",
        },
        "settings": {"context": 8192, "max_tokens": 2048, "temperature": 0, "effort": "default"},
        "hardware": {"backend": "CUDA", "gpus": ["NVIDIA GeForce RTX 4070"], "memory_gb": 40.2},
        "benchmarks": {
            "mmlu_pro": {"n": 50, "correct": 31},
            "gsm8k": {"n": 50, "correct": 44},
            "math500": {"n": 50, "correct": 30},
        },
        "speed": {"prompt_tps": 512.3, "gen_tps": 38.1},
        "seconds": 1234,
    }
    result.update(top)
    return result


def minimal_result() -> dict[str, Any]:
    """A result with only the required fields."""
    result = valid_result()
    for key in ("hardware", "speed", "seconds"):
        del result[key]
    result["model"] = {"file": "flint-Q6_K.gguf", "sha256": SHA_A, "bytes": 32720880480}
    return result


def changed(path: str, value: Any) -> dict[str, Any]:
    """valid_result() with the value at a dotted path replaced, or removed when it is REMOVE."""
    result = valid_result()
    *parents, last = path.split(".")
    target = result
    for key in parents:
        target = target[key]
    if value is REMOVE:
        del target[last]
    else:
        target[last] = value
    return result


def all_nine(counts: dict[str, int], correct: dict[str, int]) -> dict[str, Any]:
    return {bench: {"n": counts[bench], "correct": correct.get(bench, 0)} for bench in counts}


def posted(number: int, login: str = "octocat") -> dict[str, Any]:
    return {
        "issue": number,
        "user": login,
        "url": f"https://github.com/mattsaund/crucible-bench/issues/{number}",
        "at": "2026-10-26T10:00:00Z",
    }


def issue_body(result: Any) -> str:
    text = result if isinstance(result, str) else json.dumps(result, indent=2)
    return f"Posted from Crucible's Bench tab.\n\n{accept.MARKER}\n```json\n{text}\n```\n"


def issue_event(body: Any, number: int = 7, login: str = "octocat") -> dict[str, Any]:
    return {
        "action": "opened",
        "issue": {
            "number": number,
            "title": "Bench: Recursive Flint, quick",
            "body": body,
            "user": {"login": login, "id": 583231},
            "html_url": f"https://github.com/mattsaund/crucible-bench/issues/{number}",
            "created_at": "2026-10-26T10:00:00Z",
        },
        "repository": {
            "full_name": "mattsaund/crucible-bench",
            "html_url": "https://github.com/mattsaund/crucible-bench",
            "default_branch": "main",
        },
    }


class BoardCase(unittest.TestCase):
    """A fresh board repository in a temporary folder."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "repo"
        (self.root / "results").mkdir(parents=True)
        self.out = Path(temp.name) / "out"

    def post_body(self, body: Any, now: str = NOW, **issue: Any) -> dict[str, str]:
        return accept.accept_event(issue_event(body, **issue), self.root, self.out, now=now)

    def post(self, result: Any, now: str = NOW, **issue: Any) -> dict[str, str]:
        return self.post_body(issue_body(result), now=now, **issue)

    def board(self) -> dict[str, Any]:
        return json.loads((self.root / "board.json").read_text(encoding="utf-8"))

    def board_ids(self) -> list[int]:
        return [entry["id"] for entry in self.board()["entries"]]

    def comment(self) -> str:
        return (self.out / "comment.md").read_text(encoding="utf-8")

    def stored_files(self) -> list[str]:
        return sorted(p.name for p in (self.root / "results").glob("*.json"))


class AcceptTest(BoardCase):
    def test_valid_result_is_stored_and_put_on_the_board(self) -> None:
        result = valid_result()
        decision = self.post(result)
        self.assertEqual(decision, {"status": "accepted", "message": "Accepted as results/7.json."})
        self.assertEqual(json.loads((self.out / "decision.json").read_text(encoding="utf-8")), decision)

        stored = dict(result, posted=posted(7))
        text = (self.root / "results" / "7.json").read_text(encoding="utf-8")
        self.assertEqual(text, json.dumps(stored, indent=2, sort_keys=True) + "\n")

        board = self.board()
        board_text = (self.root / "board.json").read_text(encoding="utf-8")
        self.assertEqual(board_text, json.dumps(board, indent=2, sort_keys=True) + "\n")
        self.assertEqual(board["suite"], "crucible-bench-1")
        self.assertEqual(board["updated"], NOW)
        [entry] = board["entries"]
        self.assertEqual(entry, dict(
            stored,
            id=7,
            scores={"mmlu_pro": 0.62, "gsm8k": 0.88, "math500": 0.6},
            skills={"knowledge": 0.62, "math": 0.74},
            overall=0.68,
        ))

        comment = self.comment()
        self.assertIn("[results/7.json](https://github.com/mattsaund/crucible-bench/blob/main/results/7.json)", comment)
        self.assertIn("| Knowledge | 62.0% |", comment)
        self.assertIn("| Math | 74.0% |", comment)
        self.assertIn("| **Overall** | **68.0%** |", comment)
        self.assertNotIn("Reasoning", comment)

    def test_minimal_result_is_accepted(self) -> None:
        self.assertEqual(self.post(minimal_result())["status"], "accepted")

    def test_every_size_with_all_nine_benchmarks(self) -> None:
        for number, (size, counts) in enumerate((("quick", QUICK), ("standard", STANDARD), ("full", FULL)), 1):
            with self.subTest(size):
                result = valid_result(size=size, benchmarks=all_nine(counts, {"arc": 7}))
                self.assertEqual(self.post(result, number=number)["status"], "accepted", self.comment())

    def test_hardware_and_speed_may_be_partial(self) -> None:
        result = valid_result(hardware={"backend": "CPU", "gpus": []}, speed={"gen_tps": 0})
        self.assertEqual(self.post(result)["status"], "accepted")

    def test_text_beyond_ascii_is_kept_as_is(self) -> None:
        result = valid_result()
        result["model"]["name"] = "Flint \u00fc \u706b"
        self.assertEqual(self.post(result)["status"], "accepted")
        text = (self.root / "results" / "7.json").read_text(encoding="utf-8")
        self.assertIn("Flint \u00fc \u706b", text)

    def test_body_with_windows_line_endings(self) -> None:
        body = issue_body(valid_result()).replace("\n", "\r\n")
        self.assertEqual(self.post_body(body)["status"], "accepted")

    def test_block_may_sit_inside_details_and_use_a_longer_fence(self) -> None:
        text = json.dumps(valid_result(), indent=2)
        body = f"A summary.\n  {accept.MARKER}  \n<details><summary>Result</summary>\n\n```` JSON\n{text}\n````\n\n</details>\n"
        self.assertEqual(self.post_body(body)["status"], "accepted")

    def test_only_the_first_block_after_the_marker_counts(self) -> None:
        good = json.dumps(valid_result())
        body = f"{accept.MARKER}\n```json\n{good}\n```\n\n```json\n{{\"junk\": 1}}\n```\n"
        self.assertEqual(self.post_body(body)["status"], "accepted")
        body = f"```json\n{good}\n```\n{accept.MARKER}\n```json\n{{\"junk\": 1}}\n```\n"
        self.assertEqual(self.post_body(body)["status"], "rejected")


class MarkerTest(BoardCase):
    def assert_ignored(self, body: Any) -> None:
        decision = self.post_body(body)
        self.assertEqual(decision["status"], "ignored")
        self.assertEqual(self.comment(), "")
        self.assertEqual(self.stored_files(), [])
        self.assertFalse((self.root / "board.json").exists())

    def assert_rejected(self, body: str, fragment: str) -> None:
        decision = self.post_body(body)
        self.assertEqual(decision["status"], "rejected")
        self.assertIn(fragment, decision["message"])
        self.assertEqual(self.stored_files(), [])
        self.assertFalse((self.root / "board.json").exists())

    def test_ordinary_issue_is_ignored(self) -> None:
        self.assert_ignored("The Bench tab crashes when I pick a second model.")

    def test_empty_body_is_ignored(self) -> None:
        self.assert_ignored(None)

    def test_marker_inside_a_sentence_is_ignored(self) -> None:
        self.assert_ignored(f"Should the `{accept.MARKER}` line come first?")

    def test_json_without_the_marker_is_ignored(self) -> None:
        self.assert_ignored(f"```json\n{json.dumps(valid_result())}\n```\n")

    def test_marker_without_a_json_block_is_rejected(self) -> None:
        self.assert_rejected(f"{accept.MARKER}\nI forgot the result.\n", "There is no `json` code block after")

    def test_block_in_another_language_is_not_used(self) -> None:
        self.assert_rejected(f"{accept.MARKER}\n```python\nprint(1)\n```\n", "There is no `json` code block after")

    def test_unclosed_block_is_rejected(self) -> None:
        self.assert_rejected(f"{accept.MARKER}\n```json\n{json.dumps(valid_result())}\n", "never closed")

    def test_bad_json_is_rejected(self) -> None:
        self.assert_rejected(issue_body('{"schema": 1,'), "The `json` code block is not valid JSON (")


class JsonTextTest(BoardCase):
    """Rules that live in the JSON text itself, before it is a Python value."""

    def test_strict_json(self) -> None:
        base = json.dumps(valid_result())
        cases = [
            ("NaN", '"temperature": 0', '"temperature": NaN', "uses NaN"),
            ("Infinity", '"gen_tps": 38.1', '"gen_tps": Infinity', "uses Infinity"),
            ("overflowing float", '"memory_gb": 40.2', '"memory_gb": 1e400', "too large"),
            ("huge integer", '"bytes": 32720880480', '"bytes": 327208804800000000000', "too large"),
            ("repeated key", '"schema": 1', '"schema": 1, "schema": 1', "repeats the key `schema`"),
            ("control character", '"name": "Recursive Flint"', '"name": "Recursive\\u0007Flint"',
             "`model.name` contains a character that is not allowed"),
            ("lone surrogate", '"name": "Recursive Flint"', '"name": "Recursive \\ud800Flint"',
             "`model.name` contains a character that is not allowed"),
            ("deep nesting", '"seconds": 1234', '"seconds": ' + "[" * 100000 + "]" * 100000, "not valid JSON"),
            ("not an object", base, "[1, 2, 3]", "The result must be a JSON object"),
        ]
        for label, old, new, fragment in cases:
            with self.subTest(label):
                self.assertIn(old, base)
                decision = self.post(base.replace(old, new, 1))
                self.assertEqual(decision["status"], "rejected")
                self.assertIn(fragment, decision["message"])
        self.assertEqual(self.stored_files(), [])


class RuleTest(unittest.TestCase):
    """Each rule failing on its own."""

    def assert_invalid(self, result: Any, fragment: str) -> None:
        with self.assertRaises(accept.Invalid) as caught:
            accept.validate(result)
        message = str(caught.exception)
        self.assertIn(fragment, message)
        self.assertTrue(message.endswith(".") and "\n" not in message, message)

    def test_the_valid_result_passes(self) -> None:
        accept.validate(valid_result())
        accept.validate(minimal_result())

    def test_every_required_top_level_key(self) -> None:
        for key in accept.TOP_REQUIRED:
            with self.subTest(key):
                self.assert_invalid(changed(key, REMOVE), f"The result is missing `{key}`.")

    def test_every_required_model_and_settings_key(self) -> None:
        for path in ("model.file", "model.sha256", "model.bytes", "settings.context",
                     "settings.max_tokens", "settings.temperature", "settings.effort"):
            with self.subTest(path):
                where, key = path.split(".")
                self.assert_invalid(changed(path, REMOVE), f"`{where}` is missing `{key}`.")

    def test_each_rule(self) -> None:
        cases = [
            # the whole result
            ("unknown top-level key", changed("comment", "hi"), "The result has an unknown key `comment`."),
            ("posted is not for posters", changed("posted", posted(7)), "unknown key `posted`"),
            ("scores are never posted", changed("overall", 0.68), "unknown key `overall`"),
            ("schema 2", changed("schema", 2), "`schema` must be 1"),
            ("schema as text", changed("schema", "1"), "`schema` must be 1"),
            ("schema as a float", changed("schema", 1.0), "`schema` must be 1"),
            ("schema as true", changed("schema", True), "`schema` must be 1"),
            ("other suite", changed("suite", "crucible-bench-2"), "`suite` must be crucible-bench-1"),
            ("unknown size", changed("size", "huge"), "`size` must be quick, standard or full"),
            ("size as a list", changed("size", ["quick"]), "`size` must be quick, standard or full"),
            ("short version", changed("crucible", "0.9"), "`crucible` must be a version"),
            ("version with a tag", changed("crucible", "0.9.5-beta"), "`crucible` must be a version"),
            ("version in other digits", changed("crucible", "\u0660.\u0669.\u0665"), "`crucible` must be a version"),
            ("llama without b", changed("llama", "11846"), "`llama` must be a llama.cpp build"),
            ("llama with B", changed("llama", "B11846"), "`llama` must be a llama.cpp build"),
            ("date order", changed("date", "26/10/2026"), "`date` must be a date"),
            ("date that does not exist", changed("date", "2026-02-30"), "`date` must be a date"),
            ("date with a newline", changed("date", "2026-10-26\n"), "`date` must be a date"),
            # model
            ("model not an object", changed("model", "flint.gguf"), "`model` must be a JSON object"),
            ("unknown model key", changed("model.license", "MIT"), "`model` has an unknown key `license`."),
            ("file in a folder", changed("model.file", "models/flint.gguf"), "`model.file` must be a bare file name"),
            ("windows path", changed("model.file", "C:\\models\\flint.gguf"), "`model.file` must be a bare file name"),
            ("empty file", changed("model.file", ""), "`model.file` must be a bare file name"),
            ("long file", changed("model.file", "f" * 196 + ".gguf"), "`model.file` is longer than 200 characters"),
            ("file with a newline", changed("model.file", "flint\n.gguf"), "`model.file` contains a character"),
            ("file as a number", changed("model.file", 7), "`model.file` must be text"),
            ("uppercase hash", changed("model.sha256", "A" * 64), "`model.sha256` must be 64 lowercase hex"),
            ("short hash", changed("model.sha256", "a" * 63), "`model.sha256` must be 64 lowercase hex"),
            ("zero bytes", changed("model.bytes", 0), "`model.bytes` must be a positive integer"),
            ("negative bytes", changed("model.bytes", -1), "`model.bytes` must be a positive integer"),
            ("fractional bytes", changed("model.bytes", 1.5), "`model.bytes` must be a positive integer"),
            ("bytes as text", changed("model.bytes", "100"), "`model.bytes` must be a positive integer"),
            ("bytes as true", changed("model.bytes", True), "`model.bytes` must be a positive integer"),
            ("bytes past 2^53", changed("model.bytes", 2**53), "`model.bytes` must be a positive integer"),
            ("long name", changed("model.name", "n" * 101), "`model.name` is longer than 100 characters"),
            ("long arch", changed("model.arch", "a" * 101), "`model.arch` is longer than 100 characters"),
            ("negative params", changed("model.params", -1), "`model.params` must be a non-negative integer"),
            ("fractional params", changed("model.params", 39.8e9), "`model.params` must be a non-negative integer"),
            ("long quant", changed("model.quant", "q" * 33), "`model.quant` is longer than 32 characters"),
            ("repo without owner", changed("model.repo", "flint-GGUF"), "`model.repo` must look like owner/name"),
            ("repo too deep", changed("model.repo", "a/b/c"), "`model.repo` must look like owner/name"),
            ("repo as a URL", changed("model.repo", "https://huggingface.co/a/b"), "`model.repo` must look like"),
            ("repo climbing up", changed("model.repo", "../.."), "`model.repo` must look like owner/name"),
            # settings
            ("settings not an object", changed("settings", []), "`settings` must be a JSON object"),
            ("unknown setting", changed("settings.top_p", 0.9), "`settings` has an unknown key `top_p`."),
            ("context as text", changed("settings.context", "8192"), "`settings.context` must be an integer"),
            ("context as a float", changed("settings.context", 8192.0), "`settings.context` must be an integer"),
            ("max_tokens as null", changed("settings.max_tokens", None), "`settings.max_tokens` must be an integer"),
            ("temperature as text", changed("settings.temperature", "0"), "`settings.temperature` must be a number"),
            ("temperature as false", changed("settings.temperature", False), "`settings.temperature` must be a number"),
            ("unknown effort", changed("settings.effort", "max"), "`settings.effort` must be default, low, medium"),
            # hardware
            ("hardware not an object", changed("hardware", "RTX 4070"), "`hardware` must be a JSON object"),
            ("unknown hardware key", changed("hardware.cpu", "Ryzen"), "`hardware` has an unknown key `cpu`."),
            ("long backend", changed("hardware.backend", "b" * 21), "`hardware.backend` is longer than 20"),
            ("nine gpus", changed("hardware.gpus", ["RTX 4070"] * 9), "`hardware.gpus` must be a list of at most 8"),
            ("gpus as text", changed("hardware.gpus", "RTX 4070"), "`hardware.gpus` must be a list of at most 8"),
            ("long gpu name", changed("hardware.gpus", ["g" * 81]), "A name in `hardware.gpus` is longer than 80"),
            ("gpu name as a number", changed("hardware.gpus", [4070]), "A name in `hardware.gpus` must be text"),
            ("memory as text", changed("hardware.memory_gb", "40"), "`hardware.memory_gb` must be a number"),
            # speed and seconds
            ("speed not an object", changed("speed", 38.1), "`speed` must be a JSON object"),
            ("unknown speed key", changed("speed.tps", 1), "`speed` has an unknown key `tps`."),
            ("negative prompt speed", changed("speed.prompt_tps", -1), "`speed.prompt_tps` must be a non-negative"),
            ("generation speed as text", changed("speed.gen_tps", "fast"), "`speed.gen_tps` must be a non-negative"),
            ("negative seconds", changed("seconds", -5), "`seconds` must be a non-negative integer"),
            ("fractional seconds", changed("seconds", 12.5), "`seconds` must be a non-negative integer"),
            # benchmarks
            ("benchmarks not an object", changed("benchmarks", []), "`benchmarks` must be a JSON object"),
            ("no benchmarks", changed("benchmarks", {}), "`benchmarks` must hold at least one benchmark"),
            ("unknown benchmark", changed("benchmarks.mmlu", {"n": 50, "correct": 1}),
             "`benchmarks` has an unknown benchmark `mmlu`."),
            ("entry not an object", changed("benchmarks.gsm8k", 44), "`benchmarks.gsm8k` must be a JSON object"),
            ("score is never posted", changed("benchmarks.gsm8k.score", 0.88),
             "`benchmarks.gsm8k` has an unknown key `score`."),
            ("missing correct", changed("benchmarks.gsm8k.correct", REMOVE), "`benchmarks.gsm8k` is missing `correct`."),
            ("missing n", changed("benchmarks.gsm8k.n", REMOVE), "`benchmarks.gsm8k` is missing `n`."),
            ("n as text", changed("benchmarks.gsm8k.n", "50"), "`benchmarks.gsm8k.n` must be an integer"),
            ("n short for quick", changed("benchmarks.gsm8k.n", 49),
             "`benchmarks.gsm8k.n` is 49, but a quick run of GSM8K has 50 items."),
            ("standard HumanEval has 164", valid_result(size="standard", benchmarks={"humaneval": {"n": 200, "correct": 9}}),
             "`benchmarks.humaneval.n` is 200, but a standard run of HumanEval has 164 items."),
            ("full MMLU-Pro has every row", valid_result(size="full", benchmarks={"mmlu_pro": {"n": 200, "correct": 9}}),
             "`benchmarks.mmlu_pro.n` is 200, but a full run of MMLU-Pro has 12032 items."),
            ("correct above n", changed("benchmarks.gsm8k.correct", 51),
             "`benchmarks.gsm8k.correct` must be an integer from 0 to 50."),
            ("negative correct", changed("benchmarks.gsm8k.correct", -1), "`benchmarks.gsm8k.correct` must be an integer"),
            ("fractional correct", changed("benchmarks.gsm8k.correct", 44.0), "`benchmarks.gsm8k.correct` must be an integer"),
            # the top level must be an object
            ("result as a list", [valid_result()], "The result must be a JSON object."),
        ]
        for label, result, fragment in cases:
            with self.subTest(label):
                self.assert_invalid(result, fragment)

    def test_unsafe_names_are_not_repeated(self) -> None:
        self.assert_invalid(changed("@everyone", 1), "The result has an unknown key.")
        self.assert_invalid(changed("benchmarks.<b>x</b>", {"n": 50, "correct": 1}),
                            "`benchmarks` has an unknown benchmark.")

    def test_stored_results_need_a_matching_posted_object(self) -> None:
        stored = dict(valid_result(), posted=posted(7))
        accept.validate(stored, stored=True)
        with self.assertRaises(accept.Invalid):
            accept.validate(valid_result(), stored=True)
        for key, value in (("issue", 0), ("user", "no spaces allowed"), ("url", "http://x"), ("at", "2026-10-26")):
            with self.subTest(key):
                bad = dict(valid_result(), posted=dict(posted(7), **{key: value}))
                with self.assertRaises(accept.Invalid) as caught:
                    accept.validate(bad, stored=True)
                self.assertIn(f"`posted.{key}`", str(caught.exception))


class ArithmeticTest(unittest.TestCase):
    def entry(self, size: str, benchmarks: dict[str, Any]) -> dict[str, Any]:
        stored = dict(valid_result(size=size, benchmarks=benchmarks), posted=posted(7))
        accept.validate(stored, stored=True)
        return accept.board_entry(stored)

    def test_item_counts(self) -> None:
        for bench in accept.BENCHMARKS:
            with self.subTest(bench):
                self.assertEqual(accept.item_count("quick", bench), QUICK[bench])
                self.assertEqual(accept.item_count("standard", bench), STANDARD[bench])
                self.assertEqual(accept.item_count("full", bench), FULL[bench])

    def test_scores_skills_and_overall(self) -> None:
        entry = self.entry("quick", valid_result()["benchmarks"])
        self.assertEqual(entry["scores"], {"mmlu_pro": 0.62, "gsm8k": 0.88, "math500": 0.6})
        self.assertEqual(entry["skills"], {"knowledge": 0.62, "math": 0.74})
        self.assertEqual(entry["overall"], 0.68)

    def test_skills_without_benchmarks_are_absent_not_zero(self) -> None:
        entry = self.entry("quick", {"humaneval": {"n": 50, "correct": 20}, "arc": {"n": 50, "correct": 45}})
        self.assertEqual(entry["skills"], {"reasoning": 0.9, "code": 0.4})
        self.assertEqual(entry["overall"], 0.65)

    def test_all_nine(self) -> None:
        correct = {"mmlu_pro": 25, "arc": 40, "gsm8k": 45, "math500": 35, "humaneval": 30,
                   "mbpp": 40, "ifeval": 38, "hellaswag": 42, "truthfulqa": 21}
        entry = self.entry("quick", all_nine(QUICK, correct))
        self.assertEqual(entry["skills"], {
            "knowledge": 0.5, "reasoning": 0.8, "math": 0.8, "code": 0.7,
            "instructions": 0.76, "commonsense": 0.84, "truthfulness": 0.42,
        })
        self.assertEqual(entry["overall"], 0.6886)  # 4.82 / 7

    def test_full_size_rounds_to_four_places(self) -> None:
        entry = self.entry("full", {"arc": {"n": 1172, "correct": 1000}, "humaneval": {"n": 164, "correct": 82},
                                    "mbpp": {"n": 257, "correct": 257}})
        self.assertEqual(entry["scores"], {"arc": 0.8532, "humaneval": 0.5, "mbpp": 1.0})
        self.assertEqual(entry["skills"], {"reasoning": 0.8532, "code": 0.75})
        self.assertEqual(entry["overall"], 0.8016)

    def test_rounding_happens_only_at_the_end(self) -> None:
        # 1001/1319 rounds to 0.7589; (0.7589 + 0.6) / 2 would round to 0.6794.
        entry = self.entry("full", {"gsm8k": {"n": 1319, "correct": 1001}, "math500": {"n": 500, "correct": 300}})
        self.assertEqual(entry["scores"], {"gsm8k": 0.7589, "math500": 0.6})
        self.assertEqual(entry["skills"], {"math": 0.6795})
        self.assertEqual(entry["overall"], 0.6795)


class DedupeTest(BoardCase):
    def test_repost_of_the_same_file_and_size_replaces_the_earlier_one(self) -> None:
        self.post(valid_result(), number=3)
        better = valid_result()
        better["benchmarks"]["gsm8k"]["correct"] = 50
        self.assertEqual(self.post(better, number=5, now=LATER)["status"], "accepted")
        self.assertEqual(self.stored_files(), ["3.json", "5.json"])
        self.assertEqual(self.board_ids(), [5])
        self.assertEqual(self.board()["updated"], LATER)
        self.assertIn("It replaces your earlier result for this model file and size, #3.", self.comment())

    def test_other_sizes_files_and_people_each_keep_an_entry(self) -> None:
        self.post(valid_result(), number=3)
        self.post(valid_result(size="standard", benchmarks={"gsm8k": {"n": 200, "correct": 150}}), number=4)
        other_file = valid_result()
        other_file["model"]["sha256"] = SHA_B
        self.post(other_file, number=5)
        self.post(valid_result(), number=6, login="hubot")
        self.assertEqual(sorted(self.board_ids()), [3, 4, 5, 6])
        self.assertNotIn("replaces", self.comment())

    def test_logins_that_differ_only_in_case_are_one_person(self) -> None:
        self.post(valid_result(), number=3, login="OctoCat")
        self.post(valid_result(), number=4, login="octocat")
        self.assertEqual(self.board_ids(), [4])

    def test_the_newest_wins_even_when_an_older_issue_comes_last(self) -> None:
        self.post(valid_result(), number=9)
        self.assertEqual(self.post(valid_result(), number=8)["status"], "accepted")
        self.assertEqual(self.board_ids(), [9])
        self.assertIn("#9, stays on the board", self.comment())

    def test_entries_are_sorted_by_overall_then_id(self) -> None:
        for number, (correct, sha) in enumerate(((20, "1"), (45, "2"), (20, "3"), (30, "4")), 10):
            result = valid_result(benchmarks={"gsm8k": {"n": 50, "correct": correct}})
            result["model"]["sha256"] = sha * 64
            self.post(result, number=number)
        self.assertEqual(self.board_ids(), [11, 13, 10, 12])
        self.assertEqual([e["overall"] for e in self.board()["entries"]], [0.9, 0.6, 0.4, 0.4])
        results, _ = accept.load_results(self.root)
        self.assertEqual([e["id"] for e in accept.board_entries(results[::-1])], [11, 13, 10, 12])


class RebuildTest(BoardCase):
    def test_rebuild_skips_and_reports_files_that_no_longer_pass(self) -> None:
        self.post(valid_result(), number=3)
        results = self.root / "results"
        good = json.loads((results / "3.json").read_text(encoding="utf-8"))
        (results / "4.json").write_text("{ not json", encoding="utf-8")
        wrong_count = json.loads(json.dumps(good))
        wrong_count["benchmarks"]["gsm8k"]["n"] = 49
        wrong_count["posted"]["issue"] = 5
        (results / "5.json").write_text(json.dumps(wrong_count), encoding="utf-8")
        (results / "6.json").write_text(json.dumps(good), encoding="utf-8")  # passes, but says it is #3
        (results / "notes.json").write_text("{}", encoding="utf-8")
        (results / ".gitkeep").write_text("", encoding="utf-8")

        entries, skipped, board_changed = accept.rebuild(self.root, now=LATER)
        self.assertEqual([e["id"] for e in entries], [3])
        reasons = dict(skipped)
        self.assertEqual(set(reasons), {"results/4.json", "results/5.json", "results/6.json", "results/notes.json"})
        self.assertIn("The file is not valid JSON", reasons["results/4.json"])
        self.assertIn("`benchmarks.gsm8k.n` is 49", reasons["results/5.json"])
        self.assertIn("does not match the file name", reasons["results/6.json"])
        self.assertIn("not an issue number", reasons["results/notes.json"])
        self.assertFalse(board_changed)
        self.assertEqual(self.board()["updated"], NOW)

    def test_an_accepted_post_still_lands_when_another_file_is_broken(self) -> None:
        (self.root / "results" / "2.json").write_text("[]", encoding="utf-8")
        report = io.StringIO()
        with contextlib.redirect_stderr(report):
            self.assertEqual(self.post(valid_result(), number=3)["status"], "accepted")
        self.assertEqual(self.board_ids(), [3])
        self.assertEqual(report.getvalue(), "skipped results/2.json: The result must be a JSON object.\n")

    def test_deleting_a_result_brings_back_the_one_it_replaced(self) -> None:
        self.post(valid_result(), number=3)
        newer = valid_result()
        newer["benchmarks"]["gsm8k"]["correct"] = 10
        self.post(newer, number=5)
        (self.root / "results" / "5.json").unlink()
        entries, skipped, board_changed = accept.rebuild(self.root, now=LATER)
        self.assertEqual(([e["id"] for e in entries], skipped, board_changed), ([3], [], True))
        self.assertEqual(self.board()["updated"], LATER)

    def test_an_unchanged_board_is_left_alone(self) -> None:
        self.post(valid_result(), number=3)
        before = (self.root / "board.json").read_bytes()
        self.assertFalse(accept.rebuild(self.root, now=LATER)[2])
        self.assertEqual((self.root / "board.json").read_bytes(), before)

    def test_no_results_give_an_empty_board(self) -> None:
        self.assertEqual(accept.rebuild(self.root, now=NOW), ([], [], True))
        self.assertEqual(self.board(), {"suite": "crucible-bench-1", "updated": NOW, "entries": []})


class CommentTest(BoardCase):
    def test_rejection_gives_the_reason_and_points_to_the_readme(self) -> None:
        result = valid_result()
        result["benchmarks"]["gsm8k"]["n"] = 49
        decision = self.post(result)
        self.assertEqual(decision["message"], "`benchmarks.gsm8k.n` is 49, but a quick run of GSM8K has 50 items.")
        comment = self.comment()
        self.assertIn(f"> {decision['message']}\n", comment)
        self.assertIn("[README](https://github.com/mattsaund/crucible-bench#readme)", comment)

    def test_nothing_from_the_poster_reaches_a_comment(self) -> None:
        result = valid_result()
        result["model"]["name"] = "@everyone"
        self.post(result)
        self.assertNotIn("@", self.comment())
        self.post(changed("@everyone", 1))
        self.assertNotIn("@", self.comment())


class CommandLineTest(BoardCase):
    def run_script(self, *args: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=60, check=False
        )

    def test_accept_and_rebuild(self) -> None:
        event = self.out.parent / "event.json"
        event.write_text(json.dumps(issue_event(issue_body(valid_result()))), encoding="utf-8")
        done = self.run_script("accept", event, self.root, self.out)
        self.assertEqual((done.returncode, done.stdout), (0, "accepted: Accepted as results/7.json.\n"), done.stderr)

        event.write_text(json.dumps(issue_event(issue_body(changed("schema", 2)), number=8)), encoding="utf-8")
        done = self.run_script("accept", event, self.root, self.out)
        self.assertEqual((done.returncode, done.stdout), (0, "rejected: `schema` must be 1.\n"), done.stderr)

        (self.root / "results" / "9.json").write_text("{}", encoding="utf-8")
        done = self.run_script("rebuild", self.root)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout, "skipped results/9.json: The result is missing `schema`.\n"
                                      "board.json: 1 entry, unchanged.\n")

    def test_an_event_without_an_issue_is_an_error(self) -> None:
        event = self.out.parent / "event.json"
        event.write_text(json.dumps({"action": "opened"}), encoding="utf-8")
        done = self.run_script("accept", event, self.root, self.out)
        self.assertEqual(done.returncode, 1)
        self.assertIn("no issue number", done.stderr)


if __name__ == "__main__":
    unittest.main()
