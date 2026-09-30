"""Dependency-free prompt, provenance, and output helpers."""
import hashlib
import os
from pathlib import Path
import re
import tempfile

PREFIX = "# This is the assembly code:\n"
SUFFIX = "\n# What is the source code?\n"
MIPS_FLAGS = ["-march=vr4300", "-mabi=32", "-EB", "-mno-abicalls", "-fno-pic"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_prompt(pseudo):
    pseudo = pseudo.strip()
    if not pseudo:
        raise ValueError("Pseudocode input is empty")
    if pseudo.startswith(PREFIX.strip()):
        if not pseudo.endswith(SUFFIX.strip()):
            raise ValueError("Existing prompt is missing its source-code question")
        return pseudo + "\n"
    return PREFIX + pseudo + SUFFIX


def extract_function(decompiled, name):
    sections = re.split(r"^// Function:\s*([^\r\n]+)\r?$", decompiled, flags=re.M)
    for index in range(1, len(sections), 2):
        if sections[index].strip() == name and sections[index + 1].strip():
            return sections[index + 1].strip() + "\n"
    raise ValueError("Ghidra output has no completed function named " + name)


def extract_c(text):
    blocks = re.findall(r"```(?:c|C)?[ \t]*\n(.*?)```", text, flags=re.S)
    if len(blocks) > 1:
        raise ValueError("Model returned multiple C blocks; select the intended function")
    code = (blocks[0] if blocks else text).strip()
    if not code:
        raise ValueError("Model produced no source code")
    return code + "\n"


def check_token_budget(prompt_tokens, new_tokens, context_limit):
    if min(prompt_tokens, new_tokens, context_limit) < 1:
        raise ValueError("Token counts and context limit must be positive")
    if prompt_tokens + new_tokens > context_limit:
        raise ValueError("Prompt plus generation exceeds context; split the function or reduce --max-new-tokens")


def validate_mips_elf(path):
    with Path(path).open("rb") as stream:
        header = stream.read(52)
    if (len(header) < 52 or header[:4] != b"\x7fELF" or header[4:6] != b"\x01\x02"
            or int.from_bytes(header[18:20], "big") != 8):
        raise ValueError("Expected ELF32 big-endian MIPS input")


def check_destinations(destinations, inputs=(), overwrite=False):
    paths = [Path(path).resolve() for path in destinations]
    protected = {Path(path).resolve() for path in inputs}
    if len(set(paths)) != len(paths) or protected.intersection(paths):
        raise ValueError("Outputs must be distinct and must not replace input files")
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(str(path) + " exists; pass --overwrite")


def write_text(path, content, overwrite=False):
    path = Path(path)
    check_destinations([path], overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
