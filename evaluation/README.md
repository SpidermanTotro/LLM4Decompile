# Evaluation

Run commands from the repository root. V1.5 consumes disassembly; V2 consumes
Ghidra pseudocode. These published datasets test x86-64, not N64.

## Single GPU

```bash
python -m pip install -r requirements.txt
python evaluation/run_evaluation_llm4decompile_singleGPU.py \
  --model_path /path/to/local/v2-checkpoint \
  --data_path legacy-test/decompile-eval-executable-gcc-ghidra.json
```

For V1.5, use `legacy-test/decompile-eval-executable-gcc-obj.json`.
Generated code is compiled and executed in temporary directories with timeouts.

## Optional vLLM backend

Install its dependencies in a separate environment with a GPU-compatible vLLM build.

```bash
python -m pip install -r evaluation/requirements.txt
python evaluation/run_evaluation_llm4decompile_vllm.py \
  --model_path /path/to/local/v2-checkpoint \
  --testset_path legacy-test/decompile-eval-executable-gcc-ghidra.json \
  --gpus 1 --max_total_tokens 4096 --max_new_tokens 512 \
  --repeat 1 --num_workers 4 --gpu_memory_utilization 0.82 --temperature 0
```

The TGI launcher `scripts/run_evaluation_llm4decompile.sh` requires an external
TGI installation and accepts `MODEL_PATH`, `TESTSET_PATH`, `NUM_SHARDS`, and
`NUM_WORKERS`. The default paths now point to the actual `legacy-test` folder.
See [N64 checks](../n64/README.md) for the experimental local workflow.
