# SPDX-License-Identifier: Apache-2.0
"""Run the native filename contract checks without a Python extension build."""

# Standard
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestNativeFilenameEncoding(unittest.TestCase):
    """Compile and execute the public C++ filename helper's contract tests."""

    def test_filename_contract(self) -> None:
        """Filename bytes, malformed input and returned ownership stay stable."""
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a C++17 compiler is required")
        root = Path(__file__).resolve().parents[3]
        source = root / "tests/native/storage_backends/test_filename_encoding.cpp"
        with tempfile.TemporaryDirectory(prefix="lmcache-filename-test-") as temp:
            binary = Path(temp) / "test_filename_encoding"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{root}",
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
                timeout=60,
            )
            subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
