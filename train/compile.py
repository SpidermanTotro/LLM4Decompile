"""Build source/assembly JSONL with one atomic parent-process writer."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

OPT = ["O0", "O1", "O2", "O3"]


def compile_file(input_file, compiler="gcc", objdump="objdump", timeout=30):
    source = Path(input_file).read_text(encoding="utf-8")
    processed = source
    if "/* Variables and functions */" in processed:
        processed = processed.split("/* Variables and functions */")[-1]
        processed = "\n\n".join(processed.split("\n\n")[1:])
        processed = processed.replace("__attribute__((used)) ", "")
    assembly_by_opt = {}
    with tempfile.TemporaryDirectory(prefix="llm4decompile-") as directory:
        for opt in OPT:
            obj = str(Path(directory) / (opt + ".o"))
            subprocess.run([compiler, "-c", "-o", obj, str(input_file), "-" + opt],
                           check=True, capture_output=True, text=True, timeout=timeout)
            result = subprocess.run([objdump, "-d", obj], check=True,
                                    capture_output=True, text=True, timeout=timeout)
            lines = result.stdout.split("Disassembly of section .text:")[-1].strip().splitlines()
            assembly = "\n".join(re.sub(r"^0+\s", "", line.split("\t")[-1].split("#", 1)[0].strip()) for line in lines) + "\n"
            if len(assembly.splitlines()) < 4:
                raise ValueError("No usable text-section assembly: " + str(input_file))
            assembly_by_opt["opt-state-" + opt] = assembly
    return {"name": str(input_file), "input": processed, "input_ori": source,
            "output": assembly_by_opt}


def _compile_task(task):
    try:
        return compile_file(*task), None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return None, str(task[0]) + ": " + (getattr(error, "stderr", "") or str(error)).strip()


def build_dataset(root, output, jobs=None, compiler="gcc", objdump="objdump",
                  timeout=30, overwrite=False):
    root, output = Path(root), Path(output)
    jobs = min(4, os.cpu_count() or 1) if jobs is None else jobs
    if jobs < 1 or timeout <= 0:
        raise ValueError("jobs and timeout must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(str(output) + " exists; pass --overwrite to replace it")
    sources = sorted(root.rglob("*.c"))
    if not sources:
        raise ValueError("No C files found under " + str(root))
    if output.resolve() in [path.resolve() for path in sources]:
        raise ValueError("Output must not replace a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(str(path), compiler, objdump, timeout) for path in sources]
    pool, temporary = None, None
    succeeded, failed = 0, 0
    try:
        if jobs == 1:
            results = map(_compile_task, tasks)
        else:
            pool = multiprocessing.get_context("spawn").Pool(min(jobs, len(tasks)))
            results = pool.imap(_compile_task, tasks)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            for sample, error in results:
                if error:
                    failed += 1
                    print(error, file=sys.stderr)
                else:
                    stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    succeeded += 1
        if not succeeded:
            raise ValueError("All files failed compilation; existing output was preserved")
        os.replace(temporary, output)
        temporary = None
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"compiled": succeeded, "failed": failed, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_dataset(args.root, args.output, args.jobs, args.compiler,
                               args.objdump, args.timeout, args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
