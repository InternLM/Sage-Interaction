"""Apply the demo's two version-checked StepAudio2 compatibility adjustments."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

PATCHES = Path(__file__).resolve().parents[1] / "patches"


def apply_patch(original, patch):
    source = original.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    output, cursor, index = [], 0, 2
    while index < len(lines):
        match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", lines[index])
        if not match:
            raise ValueError("Malformed dependency patch")
        start = int(match.group(1)) - 1
        output.extend(source[cursor:start])
        cursor = start
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            if line[0] in " -":
                if cursor >= len(source) or source[cursor] != line[1:]:
                    raise ValueError("Dependency patch context does not match")
                cursor += 1
            if line[0] in " +":
                output.append(line[1:])
            index += 1
    output.extend(source[cursor:])
    return "".join(output)


def prepare(check_only=False):
    distribution = importlib.metadata.distribution("stepaudio2-minicpmo")
    if distribution.version != "0.1.1":
        raise RuntimeError("These adjustments require stepaudio2-minicpmo==0.1.1")
    manifest = json.loads((PATCHES / "manifest.json").read_text())
    pending = []
    for name, entry in manifest.items():
        path = Path(distribution.locate_file(name))
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in [entry["patched_sha256"], *entry.get("compatible_sha256", [])]:
            continue
        if digest != entry["original_sha256"]:
            raise RuntimeError("Dependency has unrecognized local changes: " + name)
        updated = apply_patch(
            content.decode(), (PATCHES / entry["patch"]).read_text()
        ).encode()
        if hashlib.sha256(updated).hexdigest() != entry["patched_sha256"]:
            raise RuntimeError("Dependency patch checksum mismatch: " + name)
        pending.append((path, updated))
    if check_only and pending:
        raise RuntimeError(
            "Run scripts/prepare_environment.py to apply the required adjustments"
        )
    for path, content in pending:
        temporary = path.with_name(path.name + ".sage-tmp")
        try:
            temporary.write_bytes(content)
            temporary.chmod(path.stat().st_mode)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print("StepAudio2 demo compatibility verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify without modifying the environment"
    )
    args = parser.parse_args()
    prepare(args.check)
