"""Single-GPU evaluation with isolated, bounded compilation/execution checks."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]


def evaluate_func(c_func, c_test, c_func_decompile):
    includes = "\n".join(line for text in (c_func, c_test) for line in text.splitlines()
                         if line.lstrip().startswith("#include"))
    tests = "\n".join(line for line in c_test.splitlines() if not line.lstrip().startswith("#include"))
    with tempfile.TemporaryDirectory(prefix="llm4decompile-eval-") as directory:
        function, combined, binary = (Path(directory) / name for name in ("function.c", "combined.c", "test"))
        function.write_text(includes + "\n" + c_func_decompile, encoding="utf-8")
        combined.write_text(includes + "\n" + c_func_decompile + "\n" + tests, encoding="utf-8")
        try:
            subprocess.run(["gcc", "-S", str(function), "-o", str(Path(directory) / "function.s"), "-lm"],
                           check=True, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return 0, 0
        try:
            subprocess.run(["gcc", str(combined), "-o", str(binary), "-lm"],
                           check=True, capture_output=True, timeout=10)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return 1, 0
    return 1, 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', default='LLM4Binary/llm4decompile-6.7b-v1.5')
    parser.add_argument('--data_path', default=str(REPO / 'legacy-test/decompile-eval-executable-gcc-obj.json'))
    parser.add_argument('--output_path', default='results.txt')
    args = parser.parse_args(argv)
    data = json.loads(Path(args.data_path).read_text(encoding='utf-8'))
    if not data:
        parser.error('Evaluation dataset is empty')
    optimizations = ['O0', 'O1', 'O2', 'O3']
    if any(case['type'] not in optimizations for case in data):
        parser.error('Unknown optimization level in dataset')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    counts = {opt: [0, 0, 0] for opt in optimizations}
    for case in tqdm(data):
        prompt = case['input_asm_prompt'].strip()
        if not prompt.startswith('# This is'):
            prompt = '# This is the assembly code with {} optimization:\n{}\n# What is the source code?\n'.format(case['type'], prompt)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        compiled, executed = evaluate_func(case['c_func'], case['c_test'], code)
        counts[case['type']][0] += compiled
        counts[case['type']][1] += executed
        counts[case['type']][2] += 1
    with open(args.output_path, 'a', encoding='utf-8') as stream:
        for opt, (compiled, executed, total) in counts.items():
            if total:
                stream.write('model:{},opt:{},compile rate:{:.4f},run_rate:{:.4f}\n'.format(
                    args.model_path, opt, compiled / total, executed / total))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
