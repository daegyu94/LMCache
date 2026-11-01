# SPDX-License-Identifier: Apache-2.0
"""Exercise native store completion after an injected close failure."""

# Standard
from pathlib import Path
import shutil
import subprocess
import sys

# Third Party
import pytest


def test_write_close_error_prevents_publication(tmp_path: Path) -> None:
    """Failed close rejects the store, cleans staging, and permits a retry."""
    compiler = shutil.which("g++")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("requires Linux and a compiler with GNU linker wrapping")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "fs_close_error"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-I",
            str(root),
            "-Wl,--wrap=close",
            str(root / "tests/native/storage_backends/test_fs_close_error.cpp"),
            str(root / "csrc/storage_backends/fs/connector.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(binary), str(tmp_path / "store")], check=True, timeout=20)
