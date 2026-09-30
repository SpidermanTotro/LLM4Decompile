import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from n64.common import (PREFIX, SUFFIX, build_prompt, check_destinations, check_token_budget,
                        extract_c, extract_function, validate_mips_elf, write_text)
from n64.check import check, FIXTURES
from n64.run import run

ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def test_existing_prompt_is_not_duplicated(self):
        prompt = PREFIX + 'int func0(void) { return 1; }' + SUFFIX
        self.assertEqual(build_prompt(prompt), prompt)

    def test_empty_pseudocode_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt('  \n ')

    def test_incomplete_prompt_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt(PREFIX + 'int x;')

    def test_final_source_character_is_preserved(self):
        self.assertEqual(extract_c('int func0(void) { return 1; }'), 'int func0(void) { return 1; }\n')

    def test_one_fenced_c_block(self):
        self.assertEqual(extract_c('Here it is:\n```c\nint x;\n```'), 'int x;\n')

    def test_multiple_c_blocks_rejected(self):
        with self.assertRaises(ValueError):
            extract_c('```c\nint x;\n```\n```c\nint y;\n```')

    def test_exact_function_name_selection(self):
        text = '// Function: func01\nint func01(void) { return 0; }\n\n// Function: func0\nint func0(void) { return 1; }\n'
        self.assertEqual(extract_function(text, 'func0'), 'int func0(void) { return 1; }\n')
        with self.assertRaises(ValueError):
            extract_function(text, 'missing')

    def test_context_boundary_and_overflow(self):
        check_token_budget(3584, 512, 4096)
        with self.assertRaises(ValueError):
            check_token_budget(3585, 512, 4096)

    def test_zero_generation_budget_rejected(self):
        with self.assertRaises(ValueError):
            check_token_budget(1, 0, 4096)

    def test_elf_header_and_endianness_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.o'
            header = bytearray(52)
            header[:6] = b'\x7fELF\x01\x02'
            header[18:20] = (8).to_bytes(2, 'big')
            path.write_bytes(header)
            validate_mips_elf(path)
            header[5] = 1
            path.write_bytes(header)
            with self.assertRaises(ValueError):
                validate_mips_elf(path)
            path.write_bytes(b'not ELF')
            with self.assertRaises(ValueError):
                validate_mips_elf(path)

    def test_output_collision_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.c'
            write_text(path, 'checkpoint')
            with self.assertRaises(ValueError):
                check_destinations([path], [path], overwrite=True)
            with self.assertRaises(ValueError):
                check_destinations([path, path], overwrite=True)
            with self.assertRaises(FileExistsError):
                write_text(path, 'replacement')
            self.assertEqual(path.read_text(), 'checkpoint')

    def test_dry_run_works_without_ml_dependencies(self):
        result = subprocess.run([sys.executable, '-m', 'n64.run', '--input',
                                 str(FIXTURES / 'add.pseudo.c'), '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'dry_run_no_model_loaded')
        self.assertEqual(len(report['input_sha256']), 64)
        self.assertIsNone(report['token_count'])

    def test_generation_requires_existing_local_checkpoint(self):
        args = argparse.Namespace(input=str(FIXTURES / 'add.pseudo.c'), max_new_tokens=512,
                                  context_limit=4096, dry_run=False, model=None, output=None)
        with self.assertRaises(ValueError):
            run(args)


@unittest.skipUnless(shutil.which('gcc'), 'GCC required')
class BehaviourTests(unittest.TestCase):
    def test_reference_passes_five_host_cases(self):
        report = check(FIXTURES / 'add.reference.c')
        self.assertEqual(report['status'], 'host_behaviour_matches')
        self.assertEqual(len(report['candidate_stdout'].splitlines()), 5)
        self.assertFalse(report['n64_execution_tested'])
        self.assertIsNone(report['mips_compilation_passed'])

    def test_incorrect_function_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'wrong.c'
            candidate.write_text('#include <stdint.h>\nuint32_t func0(uint32_t a,uint32_t b) { return a-b; }')
            report = check(candidate)
            self.assertEqual(report['status'], 'failed')
            self.assertIn('case', report['error'])

    def test_invalid_function_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'invalid.c'
            candidate.write_text('not valid C')
            self.assertEqual(check(candidate)['status'], 'failed')

    def test_nonterminating_function_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'loop.c'
            candidate.write_text('#include <stdint.h>\nuint32_t func0(uint32_t a,uint32_t b) { for (;;) {} }')
            self.assertEqual(check(candidate, timeout=0.2)['status'], 'failed')

    def test_missing_cross_compiler_is_not_reported_as_success(self):
        report = check(FIXTURES / 'add.reference.c', mips_compiler='/nonexistent/mips-gcc')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['mips_compilation_passed'])

    def test_cli_bad_candidate_returns_nonzero(self):
        result = subprocess.run([sys.executable, '-m', 'n64.check', '--candidate', str(FIXTURES / 'add.pseudo.c')],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['status'], 'failed')
