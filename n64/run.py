"""Refine pseudocode using a local V2 checkpoint; downloads are disabled."""
import argparse
import json
from pathlib import Path
import time
from .common import build_prompt, check_destinations, check_token_budget, extract_c, sha256, write_text


def run(args):
    input_path = Path(args.input).expanduser().resolve()
    prompt = build_prompt(input_path.read_text(encoding="utf-8"))
    if args.max_new_tokens < 1 or args.context_limit < 1:
        raise ValueError("Token limits must be positive")
    if args.dry_run:
        return {"status": "dry_run_no_model_loaded", "input_sha256": sha256(input_path),
                "prompt": prompt, "token_count": None,
                "note": "Exact token budget is checked when a local tokenizer is loaded."}
    if not args.model or not args.output:
        raise ValueError("Generation requires --model LOCAL_DIRECTORY and --output")
    model_path, output = Path(args.model).expanduser().resolve(), Path(args.output)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError("--model must be an existing local checkpoint with config.json")
    report_path, raw_path = output.with_suffix(".report.json"), output.with_suffix(".raw.txt")
    check_destinations([output, report_path, raw_path], [input_path, model_path / "config.json"], args.overwrite)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.load_in_4bit and device != "cuda":
        raise ValueError("This runner's 4-bit mode requires CUDA")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_tokens = inputs["input_ids"].shape[1]
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    context_limit = min(args.context_limit, config.get("max_position_embeddings", args.context_limit))
    check_token_budget(prompt_tokens, args.max_new_tokens, context_limit)
    kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": 0}
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if not args.load_in_4bit:
        model.to(device)
    model.eval()
    inputs = inputs.to(model.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id)
    generated = outputs[0][prompt_tokens:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    code = extract_c(raw)
    report = {"status": "generated_accuracy_unvalidated", "input": str(input_path),
              "input_sha256": sha256(input_path), "model": str(model_path),
              "config_sha256": sha256(model_path / "config.json"), "device": device,
              "load_in_4bit": args.load_in_4bit, "prompt_tokens": prompt_tokens,
              "generated_tokens": len(generated), "context_limit": context_limit,
              "generation_limit_reached": len(generated) >= args.max_new_tokens,
              "load_and_generate_seconds": time.perf_counter() - started,
              "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    write_text(output, code, args.overwrite)
    write_text(raw_path, raw + "\n", args.overwrite)
    write_text(report_path, json.dumps(report, indent=2) + "\n", args.overwrite)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Ghidra pseudocode for one function")
    parser.add_argument("--model", help="Existing local V2 checkpoint directory")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
