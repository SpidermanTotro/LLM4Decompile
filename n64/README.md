# Experimental N64 validation

The upstream V2 model refines Ghidra pseudocode. Its published x86 benchmark does
not establish N64 accuracy. This folder supplies a local, reproducible test path
for big-endian MIPS/o32 functions; model quality and on-console execution remain
separate measurements.

Run all commands from the repository root. Python 3.10+ is required for these
maintenance tools. Use an isolated Python 3.11 environment for the optional ML
stack if your current Python has incompatible dependency wheels.

## 1. Retest the code without loading weights

```bash
python scripts/check_repo.py
python -m n64.run --input n64/fixtures/add.pseudo.c --dry-run
python -m n64.check --candidate n64/fixtures/add.reference.c
```

The bundled pseudocode is explicitly synthetic. The C reference/harness checks
five unsigned-addition cases, including wrapping at 32 bits. A host check does
not claim N64 execution or binary matching. The dry run checks prompt formatting
and hashes the input; it checks the exact token budget only after loading a tokenizer.

## 2. Create an actual MIPS input and extract Ghidra pseudocode

With an existing MIPS cross compiler:

```bash
mkdir -p n64/output
mips-linux-gnu-gcc -march=vr4300 -mabi=32 -EB -mno-abicalls -fno-pic \
  -ffreestanding -O0 -c n64/fixtures/add.reference.c -o n64/output/add.mips.o

python -m n64.prepare \
  --elf n64/output/add.mips.o \
  --ghidra /path/to/ghidra/support/analyzeHeadless \
  --function func0 --output n64/output/add.pseudo.c
```

Use your installed compiler and Ghidra paths. The prepare tool validates an ELF32
big-endian MIPS header and selects Ghidra's `MIPS:BE:64:64-32addr` language with the
`o32` compiler specification. The ELF header alone cannot establish VR4300 ISA
compatibility; retain the compiler command and settings. This path needs local
Ghidra/MIPS verification before its first result counts as a measured baseline.

A `.z64` ROM requires an appropriate loader, address mapping, and function
boundaries. The prepare tool accepts an ELF, so extract or reconstruct the target
function into an appropriate MIPS ELF first.

## 3. Run your local 9B-v2 checkpoint

First install a GPU-compatible PyTorch build in an isolated environment, then:

```bash
python -m pip install -r requirements.txt
python -m pip install bitsandbytes

python -m n64.run \
  --model "$HOME/models/n64-decompiler/llm4decompile-9b-v2-n64" \
  --input n64/output/add.pseudo.c \
  --output n64/output/add.refined.c \
  --device cuda --load-in-4bit --max-new-tokens 512
```

Use `n64/fixtures/add.pseudo.c` as input if only testing the synthetic smoke fixture.
The checkpoint directory must already exist and contain `config.json`, tokenizer
files, and weights. The runner uses `local_files_only=True`; it does not download
models. Four-bit CUDA inference is optional and depends on a working bitsandbytes
installation. It is not a guarantee that every checkpoint configuration fits 16GB.

The runner uses deterministic generation, rejects a context-budget overflow,
preserves the final generated token, and writes the C candidate plus `.raw.txt`
and `.report.json` sidecars. Reports include input/config hashes, the model path,
token counts, device, quantization setting, elapsed time, and peak allocated VRAM.
The config hash is not a hash of every weight shard. Keep each checkpoint in a
separate, versioned directory for comparisons. Existing outputs require `--overwrite`.

## 4. Check generated code against known behaviour

```bash
python -m n64.check \
  --candidate n64/output/add.refined.c \
  --report n64/output/add.validation.json
```

An optional target-compilation check is:

```bash
python -m n64.check \
  --candidate n64/output/add.refined.c \
  --mips-compiler mips-linux-gnu-gcc \
  --report n64/output/add.mips-validation.json
```

Use `--reference` and `--harness` for another function. Both the reference and
candidate must compile and execute successfully; the harness must emit observable
results. A failed compiler, failed harness, timeout, or invalid MIPS object fails
the check. `n64_execution_tested` remains false because these tools do not run an
N64 emulator or console. MIPS compilation, host behaviour, and exact binary matching
are distinct checks.

## 5. Prepare training only after the baseline

Build verified `instruction`/`output` records from real Ghidra pseudocode and known
source. Separate training and evaluation by source function/project before making
optimization variants so the same function cannot leak between splits. Preserve
compiler versions, optimization flags, processor/ABI information, and provenance.
The standalone full-model script is described in [training notes](../train/README.md).
A separately configured adapter/QLoRA recipe is still needed for 9B training on 16GB;
this update adds validation tooling and fixes, not newly trained weights.
