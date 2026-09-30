# LLaMA-Factory local training example

Run the maintained launcher from the repository root. Install LLaMA-Factory with
its own documented installation procedure, and point the launcher at that checkout.
The bundled `data/dataset_info.json` names `llm4binary_v1_example.json`; the two
sample records illustrate the schema and are not a sufficient training corpus.

```bash
MODEL_PATH=/path/to/local/checkpoint \
LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory \
OUTPUT_DIR=/path/to/new-experiment \
bash train/run_training.sh
```

The checkpoint must contain `config.json`. The output directory must be empty.
Defaults are full-model fine-tuning, batch size 1, accumulation 16, context 1024,
and at most four configured preprocessing workers. Online logging is disabled by
default, and Hugging Face offline mode is enabled. This is not a measured 9B/16GB
training recipe; configure adapters/quantization separately for that target.

To select your own data and settings:

```bash
MODEL_PATH=/path/to/local/checkpoint \
LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory \
DATASET_DIR=/path/to/data-with-dataset_info \
DATASET=my_dataset_name \
OUTPUT_DIR=/path/to/new-experiment \
BATCH_SIZE=1 NUM_WORKERS=4 CUTOFF_LEN=1024 \
BF16=1 \
bash train/run_training.sh
```

`BF16=1` enables BF16; otherwise the launcher leaves precision selection to the
training backend. Further settings include `GRAD_ACCUM_STEPS`, `LEARNING_RATE`,
`NUM_EPOCHS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, and `HOSTFILE`. Check the installed
LLaMA-Factory version for any extra adapter/quantization arguments. Its installation
and pinned dependency environment should be managed separately from core inference.

The bundled `requirements.txt` is a legacy research dependency snapshot, not the
core installation entrypoint. See [training notes](../README.md),
[N64 validation](../../n64/README.md), and [validation scope](../../docs/VALIDATION.md).
