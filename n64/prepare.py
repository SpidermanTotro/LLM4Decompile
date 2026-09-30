"""Extract a function from a big-endian MIPS ELF using Ghidra headless."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from .common import check_destinations, extract_function, sha256, validate_mips_elf, write_text


def prepare(elf, ghidra, function, output, timeout=300, overwrite=False):
    elf, ghidra, output = Path(elf).resolve(), Path(ghidra).resolve(), Path(output)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    validate_mips_elf(elf)
    report_path = output.with_suffix(".prepare.json")
    check_destinations([output, report_path], [elf], overwrite)
    if not ghidra.is_file():
        raise ValueError("--ghidra must point to support/analyzeHeadless")
    scripts = Path(__file__).resolve().parents[1] / "ghidra"
    with tempfile.TemporaryDirectory(prefix="n64-ghidra-") as directory:
        decompiled = Path(directory) / "all_functions.c"
        command = [str(ghidra), directory, "n64_validation", "-import", str(elf),
                   "-processor", "MIPS:BE:64:64-32addr", "-cspec", "o32",
                   "-scriptPath", str(scripts), "-postScript", "decompile.py",
                   str(decompiled), "-deleteProject"]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        pseudo = extract_function(decompiled.read_text(encoding="utf-8"), function)
    report = {"input": str(elf), "input_sha256": sha256(elf), "function": function,
              "processor": "MIPS:BE:64:64-32addr", "compiler_spec": "o32",
              "ghidra": str(ghidra), "command": command,
              "status": "pseudocode_prepared_accuracy_unvalidated",
              "ghidra_output": result.stdout[-4000:]}
    write_text(output, pseudo, overwrite)
    write_text(report_path, json.dumps(report, indent=2) + "\n", overwrite)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", required=True)
    parser.add_argument("--ghidra", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare(args.elf, args.ghidra, args.function, args.output, args.timeout, args.overwrite)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
