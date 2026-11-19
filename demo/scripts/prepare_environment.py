"""Prepare the pinned StepAudio2 dependency for native streaming speech."""

import argparse
import importlib.metadata
from pathlib import Path
import shutil
import subprocess

PATCHES = Path(__file__).resolve().parents[1] / "patches"


def prepare(check_only=False):
    distribution = importlib.metadata.distribution("stepaudio2-minicpmo")
    if distribution.version != "0.1.1":
        raise RuntimeError("Install stepaudio2-minicpmo==0.1.1 first")
    patch = shutil.which("patch")
    if patch is None:
        raise RuntimeError("Install the system 'patch' utility first")
    pending = []
    for path in sorted(PATCHES.glob("*.patch")):
        command = [
            patch,
            "--batch",
            "--silent",
            "-p1",
            "-d",
            str(distribution.locate_file("")),
            "-i",
            str(path),
        ]
        forward = subprocess.run(
            [*command, "--forward", "--dry-run"], capture_output=True
        )
        if forward.returncode == 0:
            pending.append(command)
        elif subprocess.run(
            [*command, "--reverse", "--force", "--dry-run"], capture_output=True
        ).returncode:
            raise RuntimeError(
                "Dependency differs from the supported version: " + path.name
            )
    if check_only and pending:
        raise RuntimeError("Run python scripts/prepare_environment.py first")
    for command in pending:
        subprocess.run([*command, "--forward", "--no-backup-if-mismatch"], check=True)
    print("Speech dependencies ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    prepare(parser.parse_args().check)
