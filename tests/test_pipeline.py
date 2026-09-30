import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from train.compile import build_dataset
from evaluation.run_evaluation_llm4decompile_singleGPU import evaluate_func

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("gcc") and shutil.which("objdump"), "GCC/binutils required")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="corpus with spaces ")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def source(self, name="input.c", text="int func0(int a, int b) { return a + b; }\n"):
        path = self.root / name
        path.write_text(text)
        return path

    def test_spawn_workers_emit_complete_sorted_jsonl(self):
        for name in ["b space.c", "a.c", "z.c", "middle.c"]:
            self.source(name)
        output = self.root / "nested" / "data.jsonl"
        result = build_dataset(self.root, output, jobs=2)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(result["compiled"], 4)
        self.assertEqual([row["name"] for row in rows], sorted(row["name"] for row in rows))
        self.assertTrue(all(set(row["output"]) == {"opt-state-O0", "opt-state-O1", "opt-state-O2", "opt-state-O3"} for row in rows))
        self.assertFalse(list(self.root.rglob("*.o")) + list(self.root.rglob("*.s")))

    def test_existing_output_requires_explicit_overwrite(self):
        self.source()
        output = self.root / "data.jsonl"
        output.write_text("checkpoint\n")
        with self.assertRaises(FileExistsError):
            build_dataset(self.root, output, jobs=1)
        self.assertEqual(output.read_text(), "checkpoint\n")

    def test_total_failure_preserves_previous_output(self):
        self.source(text="this is invalid C")
        output = self.root / "data.jsonl"
        output.write_text("checkpoint\n")
        with self.assertRaises(ValueError):
            build_dataset(self.root, output, jobs=1, overwrite=True)
        self.assertEqual(output.read_text(), "checkpoint\n")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["data.jsonl", "input.c"])

    def test_mixed_failures_are_counted(self):
        self.source()
        self.source("broken.c", "this is invalid C")
        output = self.root / "data.jsonl"
        result = build_dataset(self.root, output, jobs=1)
        self.assertEqual((result["compiled"], result["failed"]), (1, 1))
        self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_empty_corpus_does_not_create_output(self):
        output = self.root / "data.jsonl"
        with self.assertRaises(ValueError):
            build_dataset(self.root, output)
        self.assertFalse(output.exists())

    def test_invalid_worker_count(self):
        with self.assertRaises(ValueError):
            build_dataset(self.root, self.root / "data.jsonl", jobs=0)

    def test_output_cannot_replace_source(self):
        source = self.source()
        before = source.read_bytes()
        with self.assertRaises(ValueError):
            build_dataset(self.root, source, jobs=1, overwrite=True)
        self.assertEqual(source.read_bytes(), before)


@unittest.skipUnless(shutil.which("gcc"), "GCC required")
class LegacyEvaluationTests(unittest.TestCase):
    def test_correct_function_compiles_and_executes(self):
        self.assertEqual(evaluate_func('', 'int main(void) { return func0(2,3) != 5; }',
                                       'int func0(int a,int b) { return a+b; }'), (1, 1))

    def test_semantic_failure_is_not_a_pass(self):
        self.assertEqual(evaluate_func('', 'int main(void) { return func0(2,3) != 5; }',
                                       'int func0(int a,int b) { return a-b; }'), (1, 0))

    def test_invalid_generated_source_fails_compilation(self):
        self.assertEqual(evaluate_func('', 'int main(void) { return 0; }', 'not valid C'), (0, 0))


class LauncherTests(unittest.TestCase):
    def test_training_requires_local_model_before_launch(self):
        import os
        env = os.environ.copy()
        env.pop('MODEL_PATH', None)
        result = subprocess.run(['bash', str(ROOT / 'train/run_training.sh')], env=env,
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MODEL_PATH', result.stderr)


class ReferenceWorkflowTests(unittest.TestCase):
    def test_optional_reference_clis_show_help_without_ml_packages(self):
        for relative in ['train/colossalai_llm4decompile/prepare_pretrain_dataset.py',
                         'sk2decompile/evaluation/evaluate_r2i.py']:
            with self.subTest(relative=relative):
                result = subprocess.run([sys.executable, str(ROOT / relative), '--help'],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_bundled_dataset_metadata_points_to_a_real_sample(self):
        directory = ROOT / 'train/llama_factory_llm4decompile/data'
        metadata = json.loads((directory / 'dataset_info.json').read_text())
        rows = json.loads((directory / metadata['llm4binary_v1']['file_name']).read_text())
        self.assertTrue(rows)
        self.assertTrue(all({'instruction', 'input', 'output'} <= set(row) for row in rows))
