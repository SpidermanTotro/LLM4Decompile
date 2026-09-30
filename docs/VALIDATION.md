# Validation record — 2026-09-30

## Scope

The fork began identical to upstream at `85b364bf093eb2ee4f3687cfe38a203fca89f23e`.
The audit indexed all 121 existing file entries and retrieved 78 text files, including source,
configuration, license, documentation, and sample metadata for review/static checks.
Large dataset records and binary/image assets were not downloaded or executed;
remote publishing retains their original Git tree entries.

The baseline Python syntax check found legacy Python-2 print syntax in the Ghidra
script. The updated script parses under Python 3 while retaining Jython-compatible
syntax. A CPython syntax check does not substitute for executing Ghidra's Java APIs.

## Completed locally

Environment: Python 3.12.14, GCC/binutils, Bash, PyYAML; no GPU/ML libraries.

Final regression run: **34 tests: 32 passed, 2 explicitly skipped** (1.088 seconds).
Repository static checks passed for **51 Python files, 9 shell scripts, 3 JSON files,
and 3 YAML files**. `git diff --check` passed after whitespace cleanup.
The standalone synthetic dry run and the five-case host C reference check also passed.

Covered checks include real host compilation and execution, incorrect/invalid
candidate rejection, runtime timeouts, spawn multiprocessing, complete deterministic
JSONL records, filenames with spaces, output preservation on failure, prompt wrapping,
function selection, ELF header checks, context limits, and local-model prerequisites.
The optional real-tensor EOS tests require `train/requirements.txt` and are skipped
when its dependencies are unavailable. Skips are not passes.

## Retest

From any directory, point Python to the repository checker:

```bash
python /path/to/LLM4Decompile/scripts/check_repo.py
```

Or from the repository root:

```bash
python scripts/check_repo.py
python -m n64.run --input n64/fixtures/add.pseudo.c --dry-run
python -m n64.check --candidate n64/fixtures/add.reference.c
```

For training tensor checks, install the training dependencies in an isolated
environment, then rerun the same checker. YAML parsing is reported as skipped if
PyYAML is unavailable; install `PyYAML` to include it.

## Pending runtime validation

- GPU inference of the local 9B-v2 checkpoint, CUDA/quantization compatibility,
  actual VRAM use, throughput, and model-output accuracy.
- Real-tensor EOS regressions and an actual fine-tuning run.
- Ghidra headless execution on real MIPS functions and the cross compiler/toolchain.
- N64 emulator or hardware execution, exact matching, and held-out N64 benchmarks.
- Docker builds and optional vLLM, TGI, IDA, VERL, LLaMA-Factory, and ColossalAI runs.

## Additional upstream gaps retained for follow-up

The legacy ColossalAI preparation script imports
`colossal_llama.dataset.spliced_and_tokenized_dataset`, which upstream does not bundle.
The SK2 R2I evaluator requires an external `metrics/R2I` implementation, also absent
from the tree. These reference workflows are not claimed to run end to end. Keep
their original algorithms rather than inventing replacements for missing research
components. [R2I source](https://github.com/e0mh4/R2I).

The four-bit inference option is not a QLoRA trainer. Full-model 9B training on a
16GB card is not established by this update. Repository fixes do not change existing
local checkpoints or automatically install files on the user's PC.
