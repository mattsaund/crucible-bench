# crucible-bench

The shared results board for the Bench tab in [Crucible](https://github.com/mattsaund/Crucible). People run local models through the crucible-bench-1 suite and post their results here, each under their own GitHub account. Results that pass the checks are collected in [`board.json`](board.json), which every Crucible reads to compare models.

## How a result gets on the board

1. In Crucible's Bench tab, run a model and post the result. Crucible opens a new issue on this repository with the result filled in, and you submit it.
2. A GitHub Action locks the issue, then checks the result and the model file on Hugging Face.
3. If it passes, the Action saves it as `results/<issue number>.json`, rebuilds `board.json`, comments with your scores and closes the issue.
4. If it does not, the Action comments with the reason, labels the issue `not accepted` and closes it.

The board takes results and nothing else. Once the Action has decided, it rewrites the issue's title and body to the result alone, or empties them when there is no result to keep, and the issue stays closed and locked. An edit to it afterwards is undone. An issue that is not a result is closed the same way, with a pointer to [Crucible's issues](https://github.com/mattsaund/Crucible/issues), where questions and bug reports go. Only newly opened issues are checked, so to fix a result, post it again.

A result issue has a line holding only `<!-- crucible-bench -->`, followed by a `` ```json `` block with the result. Crucible writes both. Everything in a result is public, and so is the account that posts it.

Posting the same model file at the same size again replaces your earlier result on the board. The earlier file stays in `results/`.

## What is checked, and what is not

Results are self-reported. The Action checks that a result is well formed: it is for this suite and a known size, each benchmark has exactly the number of items its size calls for, `correct` is between 0 and `n`, and every field has the right type and form (see [the result](#the-result)). A result holds numbers and names of a fixed form, never free text. It cannot check that a model really scored what was posted.

What it does pin down is the model, which must be a file anyone can download from Hugging Face. The Action asks Hugging Face's API whether `model.repo` is a public model holding `model.file` with the SHA-256 `model.sha256` and the size `model.bytes`, and turns the result away with a one-line reason if not. So a result says exactly which file was run, its names come from Hugging Face, and anyone can fetch that file and run the same benchmarks to compare. A split GGUF (`x-00001-of-00003.gguf` and the rest) goes by its first part, which is the part hashed, and `model.bytes` is the size of all its parts.

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

Each item is asked as a chat with one user message, worded as in Crucible's [`bench.py`](https://github.com/mattsaund/Crucible/blob/main/scripts/orchestrator/crucible_orchestrator/bench.py). The model answers in its own words with greedy decoding (temperature 0). A reply may run to 16384 tokens in a 20480-token context, room for a model that thinks first. A model too large to load with that much context gets 4096 tokens in 8192 instead, and its `settings` say so. The answer is read from the reply's final text:

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
    "file": "gpt-oss-20b-MXFP4.gguf",
    "sha256": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
    "bytes": 12109566624,
    "repo": "ggml-org/gpt-oss-20b-GGUF",
    "arch": "gpt-oss",
    "params": 20914757184,
    "quant": "MXFP4"
  },
  "settings": {"context": 20480, "max_tokens": 16384, "temperature": 0, "effort": "default"},
  "hardware": {"backend": "CUDA", "gpu_count": 3, "memory_gb": 40.2},
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
| `model.file` | the file's name on Hugging Face, of letters, digits and `._+-` ending in `.gguf`, at most 200 characters. A split GGUF goes by its first part, like `x-00001-of-00003.gguf` |
| `model.sha256` | the file's SHA-256, 64 lowercase hex characters |
| `model.bytes` | the file's size (all the parts of a split GGUF), a positive integer |
| `model.repo` | the Hugging Face repository holding the file, `"owner/name"` as Hugging Face spells it |
| `model.arch` | optional: the architecture as llama.cpp names it, like `"llama"` or `"gpt-oss"`, of lowercase letters, digits and `_.-`, at most 32 characters |
| `model.params` | optional: the parameter count, a non-negative integer |
| `model.quant` | optional: a quantization like `"Q4_K_M"`, `"IQ2_XXS"`, `"UD-Q4_K_XL"`, `"MXFP4"` or `"BF16"` |
| `settings` | all four of `context` and `max_tokens` (integers), `temperature` (a number) and `effort` (`"default"`, `"low"`, `"medium"` or `"high"`) |
| `hardware` | optional, as are each of `backend` (`"CPU"`, `"CUDA"`, `"Metal"`, `"Vulkan"`, `"ROCm"` or `"SYCL"`), `gpu_count` (an integer from 0 to 64) and `memory_gb` (the GPUs' memory in GB, a non-negative number) |
| `speed` | optional, as are each of `prompt_tps` and `gen_tps` (non-negative numbers) |
| `seconds` | optional: the run's length, a non-negative integer |
| `benchmarks` | at least one of the nine ids, each `{"n": ..., "correct": ...}`: `n` is the item count for the size, and `correct` is from 0 to `n` |

Every field not marked optional is required, and an optional field with nothing to say is left out (null is refused). Unknown keys are refused anywhere, and so are `NaN`, `Infinity`, repeated keys and integers beyond 2^53 - 1. There is no place for words: the name a GGUF gives its model and the names of the GPUs stay on the machine that ran it.

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
      "model": {"bytes": 12109566624, "file": "gpt-oss-20b-MXFP4.gguf", "repo": "ggml-org/gpt-oss-20b-GGUF", "sha256": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"},
      "overall": 0.75,
      "posted": {"at": "2026-10-26T18:03:52Z", "issue": 12, "url": "https://github.com/mattsaund/crucible-bench/issues/12", "user": "octocat"},
      "schema": 1,
      "scores": {"gsm8k": 0.88, "mmlu_pro": 0.62},
      "settings": {"context": 20480, "effort": "default", "max_tokens": 16384, "temperature": 0},
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

Only the newest result (the highest issue number) is kept for each person, model file (`model.sha256`) and size. Logins are compared without regard to case, as GitHub does. Every string in an entry is a name or a code of a fixed form, never free text. Show them as plain text all the same, never as HTML or Markdown.

## Removing a result

A maintainer deletes `results/<n>.json` and pushes. The rebuild workflow then rebuilds `board.json` without it (it can also be started by hand from the Actions tab). If that person posted the same model file at the same size before, their earlier result comes back on the board; delete those files too to remove them all.

If an accept run fails or is canceled, the issue stays open and locked, with no comment. When Hugging Face does not answer, the run fails this way rather than turn a good result away. Re-running that run from the Actions tab checks it again.

## License

[MIT](LICENSE). The benchmark datasets are not part of this repository; see each dataset card for its license.
