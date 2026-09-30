"""Run source/config/shell checks and lightweight regression tests from any cwd."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true',
                        help='fail if any regression test or YAML validation is skipped')
    args = parser.parse_args()
    paths = [path for path in ROOT.rglob('*') if path.is_file()
             and not any(part.startswith('.') or part in {'__pycache__', 'node_modules', 'output_models', 'models'}
                         for part in path.relative_to(ROOT).parts)
             and not path.is_relative_to(ROOT / 'n64/output')]
    counts = {'python': 0, 'shell': 0, 'json': 0, 'yaml': 0}
    errors = []
    try:
        import yaml
    except ImportError:
        yaml = None
    for path in paths:
        try:
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                counts['python'] += 1
            elif path.suffix == '.sh':
                if not shutil.which('bash'):
                    raise RuntimeError('Bash required for shell syntax checks')
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True, timeout=10)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
                counts['shell'] += 1
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
                counts['json'] += 1
            elif path.suffix in {'.yaml', '.yml'} and yaml is not None:
                yaml.safe_load(path.read_text(encoding='utf-8'))
                counts['yaml'] += 1
        except (OSError, SyntaxError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append(str(path.relative_to(ROOT)) + ': ' + str(error))
        except Exception as error:
            # PyYAML errors do not inherit the built-in exception types above.
            errors.append(str(path.relative_to(ROOT)) + ': ' + str(error))
    print(json.dumps({'static_checks': counts, 'errors': errors,
                      'yaml_note': 'checked' if yaml is not None else 'skipped: install PyYAML'}, indent=2), flush=True)
    if errors:
        return 1
    # Discovery imports repository modules even when launched outside the checkout.
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'regression_tests': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'skipped': [{'test': str(test), 'reason': reason}
                                  for test, reason in result.skipped],
                      'strict': args.strict}, indent=2), flush=True)
    if not result.wasSuccessful():
        return 1
    if args.strict and (result.skipped or yaml is None):
        print('Strict validation incomplete: resolve the reported skips and rerun.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
