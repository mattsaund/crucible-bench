# crucible-bench

The shared results board for the Bench tab in [Crucible](https://github.com/mattsaund/Crucible). People run local models through the crucible-bench-1 suite and post their results here, each under their own GitHub account. Results that pass the checks are collected in [`board.json`](board.json), which every Crucible reads to compare models.

## How a result gets on the board

1. In Crucible's Bench tab, run a model and post the result. Crucible opens a new issue on this repository with the result filled in, and you submit it.
2. A GitHub Action reads the issue and checks the result.
3. If it passes, the Action saves it as `results/<issue number>.json`, rebuilds `board.json`, comments with your scores and closes the issue.
4. If it does not, the Action comments with the reason, labels the issue `not accepted` and closes it.

Issues without a result are left alone, so questions and bug reports are welcome here. Only newly opened issues are checked: to fix a result, post it again.

A result issue has a line holding only `<!-- crucible-bench -->`, followed by a `` ```json `` block with the result. Crucible writes both. Everything in a result is public, and so is the account that posts it.

Posting the same model file at the same size again replaces your earlier result on the board. The earlier file stays in `results/`.

## What is checked, and what is not

Results are self-reported. The Action checks that a result is well formed: it is for this suite and a known size, each benchmark has exactly the number of items its size calls for, `correct` is between 0 and `n`, and every field has the right type and length (see [the result](#the-result)). It cannot check that a model really scored what was posted.

What it does pin down is the model. `model.sha256` is the SHA-256 of the whole GGUF file, so it says exactly which file was run. Items are chosen the same way everywhere, so anyone with that file can run the same benchmarks and compare.

## The benchmarks

| id | Benchmark | Skill | Dataset | Config | Split | Rows |
|---|---|---|---|---|---|--:|
| `mmlu_pro` | MMLU-Pro | knowledge | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | default | test | 12032 |
| `arc` | ARC-Challenge | reasoning | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | ARC-Challenge | test | 1172 |
| `gsm8k` | GSM8K | math | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | main | test | 1319 |
| `math500` | MATH-500 | math | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | default | test | 500 |
| `humaneval` | HumanEval | code | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | openai_humaneval | test | 164 |
| `mbpp` | MBPP | code | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | sanitized | test | 257 |
| `ifeval` | IFEval | instructions | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | default | train | 541 |
| `hellaswag` | HellaSwag | commonsense | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | default | validation | 10042 |
| `truthfulqa` | TruthfulQA | truthfulness | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | multiple_choice | validation | 817 |

The datasets are not part of this repository: Crucible reads the rows from Hugging Face's datasets server. See each dataset card for its license.

Each item is asked as a chat with one user message, worded as in Crucible's [`bench.py`](https://github.com/mattsaund/Crucible/blob/main/scripts/orchestrator/crucible_orchestrator/bench.py). The model answers in its own words (it may think first) with greedy decoding (temperature 0), and the answer is read from its final text:

- MMLU-Pro, ARC-Challenge and HellaSwag: the letter of a choice (A to J for `options`, one of `choices.label`, A to D for `endings`), compared with `answer`, `answerKey` or `label`.
- TruthfulQA: the letter of one of `mc1_targets.choices`, shown shuffled (see below). The right one is the choice labeled 1.
- GSM8K: the final number, compared with the number after `####` in `answer`.
- MATH-500: the `\boxed{}` answer, compared with `answer` after LaTeX normalization.
- HumanEval: `prompt`, the completion, `test` and `check(entry_point)` are run together.
- MBPP: the code is run with `test_imports` and the asserts in `test_list`.
- IFEval: every instruction must be followed (strict, prompt level).

## Sizes and items

| Size | Items per benchmark |
|---|---|
| `quick` | min(50, rows): 50 for every benchmark |
| `standard` | min(200, rows): 200, except all 164 of HumanEval |
| `full` | every row |

Items are chosen the same way everywhere, so results are comparable. For each row index `i` from 0 to rows - 1, the key is the SHA-256 hex digest of `crucible-bench-1/<id>/<i>`. The indices are sorted by key, the first n are taken, and they run in ascending index order. TruthfulQA's MC1 lists the right answer first, so its choices are shown in the order of the SHA-256 hex digest of `crucible-bench-1/truthfulqa/<i>/<choice text>`. In Python:

```python
import hashlib

def key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def items(bench_id, rows, n):
    picked = sorted(range(rows), key=lambda i: key(f"crucible-bench-1/{bench_id}/{i}"))[:n]
    return sorted(picked)

def truthfulqa_choices(i, choices):
    return sorted(choices, key=lambda text: key(f"crucible-bench-1/truthfulqa/{i}/{text}"))
```

## Scores

Scores are never posted. Whoever reads a result works them out from `benchmarks`, and the board does the same:

- A benchmark's score is `correct / n`.
- A skill is the mean of the scores of its benchmarks that were run. Knowledge is MMLU-Pro, reasoning is ARC-Challenge, math is GSM8K and MATH-500, code is HumanEval and MBPP, instructions is IFEval, commonsense is HellaSwag and truthfulness is TruthfulQA. A skill with none of its benchmarks run is absent, not zero.
- Overall is the mean of the skills present.

The board works these out exactly and rounds to four decimal places only at the end (halves to even). For example, a quick run with 31 of 50 on MMLU-Pro and 44 of 50 on GSM8K has knowledge 0.62, math 0.88 and overall 0.75.

## The result

A result as Crucible posts it:

```json
{
  "schema": 1,
  "suite": "crucible-bench-1",
  "size": "quick",
  "crucible": "0.9.5",
  "llama": "b11846",
  "date": "2026-10-26",
  "model": {
    "file": "flint-Q6_K.gguf",
    "sha256": "<64 lowercase hex: the whole GGUF file>",
    "bytes": 32720880480,
    "name": "Recursive Flint",
    "arch": "flint",
    "params": 39800000000,
    "quant": "Q6_K",
    "repo": "unsloth/flint-GGUF"
  },
  "settings": {"context": 8192, "max_tokens": 4096, "temperature": 0, "effort": "default"},
  "hardware": {"backend": "CUDA", "gpus": ["NVIDIA GeForce RTX 4070"], "memory_gb": 40.2},
  "benchmarks": {
    "mmlu_pro": {"n": 50, "correct": 31},
    "gsm8k": {"n": 50, "correct": 44}
  },
  "speed": {"prompt_tps": 512.3, "gen_tps": 38.1},
  "seconds": 1234
}
```

| Field | Rule |
|---|---|
| `schema` | `1` |
| `suite` | `"crucible-bench-1"` |
| `size` | `"quick"`, `"standard"` or `"full"` |
| `crucible` | Crucible's version, like `"0.9.5"` |
| `llama` | the llama.cpp build, like `"b11846"` |
| `date` | the day of the run, like `"2026-10-26"` |
| `model.file` | required: a bare file name with no slashes, at most 200 characters |
| `model.sha256` | required: 64 lowercase hex characters |
| `model.bytes` | required: the file size, a positive integer |
| `model.name`, `model.arch` | optional: at most 100 characters each |
| `model.params` | optional: the parameter count, a non-negative integer |
| `model.quant` | optional: at most 32 characters |
| `model.repo` | optional: `"owner/name"` |
| `settings` | all four of `context` and `max_tokens` (integers), `temperature` (a number) and `effort` (`"default"`, `"low"`, `"medium"` or `"high"`) |
| `hardware` | optional, as are each of `backend` (at most 20 characters), `gpus` (at most 8 names of at most 80 characters) and `memory_gb` (a number) |
| `speed` | optional, as are each of `prompt_tps` and `gen_tps` (non-negative numbers) |
| `seconds` | optional: the run's length, a non-negative integer |
| `benchmarks` | at least one of the nine ids, each `{"n": ..., "correct": ...}`: `n` is the item count for the size, and `correct` is from 0 to `n` |

Every field not marked optional is required, and an optional field with nothing to say is left out (null is refused). Unknown keys are refused anywhere, and so are `NaN`, `Infinity`, repeated keys, integers beyond 2^53 - 1 and control characters in text.

## board.json

Every Crucible reads <https://raw.githubusercontent.com/mattsaund/crucible-bench/main/board.json>. Its keys are sorted:

```json
{
  "entries": [
    {
      "benchmarks": {"gsm8k": {"correct": 44, "n": 50}, "mmlu_pro": {"correct": 31, "n": 50}},
      "crucible": "0.9.5",
      "date": "2026-10-26",
      "id": 12,
      "llama": "b11846",
      "model": {"bytes": 32720880480, "file": "flint-Q6_K.gguf", "sha256": "9f2c000000000000000000000000000000000000000000000000000000000000"},
      "overall": 0.75,
      "posted": {"at": "2026-10-26T18:03:52Z", "issue": 12, "url": "https://github.com/mattsaund/crucible-bench/issues/12", "user": "octocat"},
      "schema": 1,
      "scores": {"gsm8k": 0.88, "mmlu_pro": 0.62},
      "settings": {"context": 8192, "effort": "default", "max_tokens": 4096, "temperature": 0},
      "size": "quick",
      "skills": {"knowledge": 0.62, "math": 0.88},
      "suite": "crucible-bench-1"
    }
  ],
  "suite": "crucible-bench-1",
  "updated": "2026-10-26T18:04:10Z"
}
```

| Key | What it holds |
|---|---|
| `suite` | `"crucible-bench-1"` |
| `updated` | when the entries last changed, in UTC |
| `entries` | the results on the board: those with all seven skills first, then the rest, each highest `overall` first (on a tie, the lower `id` first) |

Each entry is a stored result, which is the result as posted plus `posted`, with four keys added:

| Key | What it holds |
|---|---|
| `id` | the issue number, so the file is `results/<id>.json` |
| `posted` | `issue`, `user` (the GitHub login), `url` (the issue) and `at` (when the issue was opened) |
| `scores` | each benchmark's `correct / n` |
| `skills` | each skill with at least one benchmark run |
| `overall` | the mean of the skills present |

Only the newest result (the highest issue number) is kept for each person, model file (`model.sha256`) and size. Logins are compared without regard to case, as GitHub does. Text in a result comes from whoever posted it, so show it as plain text, never as HTML or Markdown.

## Removing a result

A maintainer deletes `results/<n>.json` and pushes. The rebuild workflow then rebuilds `board.json` without it (it can also be started by hand from the Actions tab). If that person posted the same model file at the same size before, their earlier result comes back on the board; delete those files too to remove them all.

If an accept run fails or is canceled, the issue stays open with no comment. Re-running that run from the Actions tab checks it again.

## License

[MIT](LICENSE). The benchmark datasets are not part of this repository; see each dataset card for its license.
