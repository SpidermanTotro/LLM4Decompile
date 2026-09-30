# Henric N64 Decompiler

A local decompilation and validation lab for experimenting with N64 MIPS functions,
built on [LLM4Decompile](https://github.com/albertan017/LLM4Decompile).

This fork adds a reproducible path from Ghidra pseudocode to a local model's C
candidate, then checks that candidate against known behaviour. It also repairs
dataset generation, training setup, token decoding, and repository paths.

**Working project name:** Henric N64 Decompiler. The GitHub repository is currently
`SpidermanTotro/LLM4Decompile`; clone commands below use that existing address.
The Python package remains `n64`.

## Current status

| Area | What is available | Validation status |
| --- | --- | --- |
| Repository checks | Python, shell, JSON and YAML checks; regression suite | 32 tests passed, 2 ML tests skipped in the maintenance environment |
| Dataset preparation | Ordered JSONL output, isolated compilation, spawn workers, overwrite protection | Host GCC regression tests passed |
| Local inference | Offline checkpoint loading, optional CUDA four-bit inference, token budgets, output reports | GPU/model execution still needs validation |
| Ghidra preparation | ELF32 big-endian MIPS checks, named-function extraction, headless command | Actual Ghidra/MIPS run still needs validation |
| Candidate checking | Host reference/harness comparison; optional MIPS compilation | Five-case synthetic host fixture passed |
| N64 execution | Emulator/hardware tests and exact binary matching | Pending |
| Training | Corrected standalone data/token handling and local-model launcher | Tensor tests and actual training pending |

Upstream model results measure its published benchmarks. They do not establish
N64 accuracy for this fork. This update contains code and tools; it does not
create or retrain a checkpoint.

## Get the project

For a fresh checkout:

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone https://github.com/SpidermanTotro/LLM4Decompile.git
cd LLM4Decompile
```

For an existing checkout, enter that directory and run `git pull --ff-only`.
Keep existing local changes and checkpoints before updating.

Use Python 3.10+ for the maintenance tools. For the ML stack, an isolated Python
3.11 environment is a practical starting point when newer Python versions lack
compatible wheels. GCC and Bash are required for the lightweight regression suite.

## First: retest without model weights

From the repository root:

```bash
python scripts/check_repo.py
python -m n64.run --input n64/fixtures/add.pseudo.c --dry-run
python -m n64.check --candidate n64/fixtures/add.reference.c
```

The pseudocode fixture is synthetic. Its host harness checks five unsigned-addition
cases, including 32-bit wrapping. The dry run validates prompt formatting and
input hashing; exact token counting requires the model tokenizer.

For a run that fails when any regression or YAML check is skipped:

```bash
python scripts/check_repo.py --strict
```

The checker reports every skip and its reason. Exit codes are 0 for success,
1 for static/test failures, and 2 for incomplete strict validation. Install
`PyYAML` to include YAML validation, and `train/requirements.txt` for tensor tests.

## Run an existing local checkpoint

Create or activate an isolated environment and install a GPU-compatible PyTorch
build first. Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install bitsandbytes

mkdir -p n64/output
python -m n64.run \
  --model "$HOME/models/n64-decompiler/llm4decompile-9b-v2-n64" \
  --input n64/fixtures/add.pseudo.c \
  --output n64/output/add.refined.c \
  --device cuda --load-in-4bit --max-new-tokens 512
```

The model directory must already contain its config, tokenizer and weight files.
The runner loads locally with `local_files_only=True`. A local directory's name
does not prove the weights were trained on N64 data.

Four-bit inference needs working CUDA and bitsandbytes. A 16GB GPU is the intended
local test setting, but fit and performance must be measured with the actual
checkpoint and generation settings. Four-bit inference is not a QLoRA trainer.

The command writes:

- `add.refined.c`: extracted C candidate.
- `add.refined.raw.txt`: raw decoded model output.
- `add.refined.report.json`: input/config hashes, tokens, timing, device and peak allocated VRAM.

Existing output files require an explicit `--overwrite`. Keep separate output
names and versioned checkpoint directories when comparing runs.

## Check the generated candidate

```bash
python -m n64.check \
  --candidate n64/output/add.refined.c \
  --report n64/output/add.validation.json
```

For another function, supply its known source with `--reference` and an observable
test harness with `--harness`. Compilation errors, execution errors, mismatched
outputs and timeouts fail validation.

With a MIPS cross compiler installed, add
`--mips-compiler mips-linux-gnu-gcc` to request target compilation as well.
Host behaviour and MIPS compilation are separate evidence.
The report keeps `n64_execution_tested` false: this checker does not execute
an N64 emulator or console.

## Prepare real N64 inputs

Use a big-endian MIPS ELF and an installed Ghidra headless launcher.
The preparation command selects an exact named function and records provenance.
See [the full N64 workflow](n64/README.md) for compiler flags and complete commands.

A raw `.z64` ROM needs a suitable loader, address mapping and function boundaries.
The current preparation tool accepts ELF input. It does not turn an entire ROM
directly into a rebuilt game.

## Training and datasets

Start with a measured inference baseline and verified pseudocode/source pairs.
Split by original source function or project before creating optimization variants.
Keep compiler versions, ABI, optimization flags and data provenance with the records.

- [Training guide](train/README.md): standalone fine-tuning and corrected data handling.
- [LLaMA-Factory notes](train/llama_factory_llm4decompile/README.md): separate external framework setup.
- [Dataset benchmark guide](decompile-bench/readme.md): upstream datasets and evaluation.
- [Evaluation guide](evaluation/README.md): optional legacy evaluation backends.

The maintained training launcher requires a local model and a fresh output directory.
Full-model 9B training on a 16GB GPU has not been established here. A tested
adapter/QLoRA recipe remains future work.

## Files and folders

| Path | Purpose |
| --- | --- |
| `n64/` | Preparation, local inference, validation and synthetic fixtures |
| `scripts/check_repo.py` | Static checks, regression summary and strict mode |
| `tests/` | Host compilation, dataset, prompt and training regressions |
| `ghidra/` | Ghidra extraction script and example |
| `train/` | Dataset compilation and training entrypoints |
| `evaluation/` | Upstream/legacy model evaluation |
| `decompile-bench/` | Upstream benchmark tooling |
| `sk2decompile/` | Upstream skeleton/identifier recovery workflow |
| `docs/VALIDATION.md` | Measured results and pending runtime checks |
| `docs/UPSTREAM_README.md` | Preserved upstream-facing overview, model tables and citations |

Core dependencies live in `requirements.txt`; training dependencies in
`train/requirements.txt`; optional vLLM/TGI backends in
`evaluation/requirements.txt`. Keep optional research frameworks in separate environments.

## Known gaps and next milestones

1. Execute the local checkpoint and record latency, VRAM, token counts and output quality.
2. Verify Ghidra and cross compilation on real MIPS functions.
3. Add held-out function harnesses, then emulator or hardware evidence.
4. Run the tensor regressions and establish an appropriate training recipe.
5. Validate optional Docker and research-framework workflows separately.

The legacy ColossalAI preparation path requires an unbundled
`colossal_llama.dataset.spliced_and_tokenized_dataset` implementation.
The SK2 R2I evaluator needs an external `metrics/R2I` component.
These remain documented reference-workflow dependencies.

## History, credit and license

See [CHANGELOG.md](CHANGELOG.md) for fork changes and
[docs/VALIDATION.md](docs/VALIDATION.md) for the audit scope.
The original baseline is preserved on
`checkpoint/before-n64-maintenance-20260930`.

This project derives from LLM4Decompile and retains the original
[code license](LICENSE) and [bundled model license](LICENSE-MODEL). Check the terms
of the checkpoint you actually use. The upstream papers, authors, checkpoint links and citations
are preserved in [the upstream overview](docs/UPSTREAM_README.md).
The new project title identifies this fork's work; upstream checkpoints retain
their original names and attribution.
