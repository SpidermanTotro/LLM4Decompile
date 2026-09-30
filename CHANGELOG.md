# Fork changelog

## 2026-09-30 — Pipeline repairs and experimental N64 validation

Baseline: upstream `85b364bf093eb2ee4f3687cfe38a203fca89f23e`.

- Replace competing worker JSONL writes with one atomic parent-process writer.
  Keep the legacy dataset schema, use spawn-compatible bounded workers, support
  paths with spaces, and compile in temporary directories with timeouts.
- Add missing training dependencies and separate core inference from optional
  vLLM/TGI and other benchmark requirements.
- Keep real EOS tokens attended when EOS shares the padding token ID; handle
  standalone/distributed preprocessing and datasets smaller than three records.
- Decode only generated tokens without unconditionally dropping the final token.
- Bound and isolate the legacy single-GPU evaluator's compilation/execution,
  count actual cases per optimization level, and repair legacy dataset paths.
- Make Ghidra output function boundaries explicit, report decompilation failures,
  dispose resources, and preserve compatibility with its Jython script runtime.
- Add local N64 prepare/run/check tools, synthetic/reference fixtures, provenance
  reports, context-budget checks, and optional MIPS target-compilation validation.
- Fix SK2 example filenames/paths and update setup and retest documentation.
- Give the LLaMA-Factory launcher local checkpoint validation, disabled online
  logging by default, bounded workers, quoted arguments, and fresh output handling.
- Add repository checks and focused GCC/multiprocessing/prompt/behaviour regressions.

See [validation notes](docs/VALIDATION.md) for completed checks, skips, and pending
GPU/toolchain work. Existing datasets, model references, images, and research
attribution are retained. No model weights were trained by this update.
