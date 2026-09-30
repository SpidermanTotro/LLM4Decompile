"""Run source/config/shell checks and lightweight regression tests from any cwd."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
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
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
