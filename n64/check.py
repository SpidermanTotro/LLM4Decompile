"""Compare generated C with a reference using a host test harness."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from .common import MIPS_FLAGS, check_destinations, sha256, validate_mips_elf, write_text

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def compile_and_run(source, harness, compiler, directory, name, timeout):
    combined, binary = Path(directory) / (name + ".c"), Path(directory) / name
    combined.write_text(Path(source).read_text(encoding="utf-8") + "\n" +
                        Path(harness).read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run([compiler, "-std=c99", "-O0", "-Wall", "-Wextra", str(combined),
                    "-o", str(binary), "-lm"], check=True, capture_output=True, text=True, timeout=timeout)
    return subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=timeout).stdout


def check(candidate, reference=FIXTURES / "add.reference.c", harness=FIXTURES / "add.harness.c",
          compiler="gcc", timeout=5, mips_compiler=None):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    report = {"status": "failed", "candidate_sha256": sha256(candidate),
              "reference_sha256": sha256(reference), "harness_sha256": sha256(harness),
              "compiler": compiler, "host_behaviour_matches": False,
              "mips_compilation_passed": None, "n64_execution_tested": False}
    with tempfile.TemporaryDirectory(prefix="n64-check-") as directory:
        try:
            expected = compile_and_run(reference, harness, compiler, directory, "reference", timeout)
            actual = compile_and_run(candidate, harness, compiler, directory, "candidate", timeout)
            report.update(expected_stdout=expected, candidate_stdout=actual, host_behaviour_matches=expected == actual)
            if not expected.strip():
                raise ValueError("Harness must emit observable results")
            if mips_compiler:
                obj = Path(directory) / "candidate.mips.o"
                command = [mips_compiler, *MIPS_FLAGS, "-ffreestanding", "-c",
                           str(Path(candidate).resolve()), "-o", str(obj)]
                report.update(mips_compiler=mips_compiler, mips_flags=MIPS_FLAGS, mips_compilation_passed=False)
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
                validate_mips_elf(obj)
                report["mips_compilation_passed"] = True
            if report["host_behaviour_matches"]:
                report["status"] = "host_behaviour_matches"
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["error"] = (getattr(error, "stderr", "") or str(error)).strip()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", default=str(FIXTURES / "add.reference.c"))
    parser.add_argument("--harness", default=str(FIXTURES / "add.harness.c"))
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--mips-compiler")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.report:
            check_destinations([args.report], [args.candidate, args.reference, args.harness], args.overwrite)
        report = check(args.candidate, args.reference, args.harness, args.compiler, args.timeout, args.mips_compiler)
        if args.report:
            write_text(args.report, json.dumps(report, indent=2) + "\n", args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "host_behaviour_matches" else 1


if __name__ == "__main__":
    raise SystemExit(main())
