# SPDX-License-Identifier: Apache-2.0
"""Compile and run storage connector regressions without Python extensions.

Requires a C++17 compiler, the hf3fs_usrbio.h SDK header, and Abseil headers.
HF3FS SDK functions are supplied by hf3fs_mock.cpp; no 3FS service is used.
"""

# Standard
from pathlib import Path
import argparse
import os
import subprocess
import tempfile


def main() -> None:
    """Build selected regressions in a temporary directory and run each one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include", action="append", default=[], help="SDK/header path"
    )
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--only",
        choices=[
            "fs_size", "hf3fs_ownership", "hf3fs_shutdown",
            "hf3fs_config", "hf3fs_paths",
        ],
    )
    parser.add_argument("--link", action="append", default=[], help="extra linker flag")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    tests = Path(__file__).resolve().parent
    names = [args.only] if args.only else [
        path.stem.removeprefix("test_") for path in sorted(tests.glob("test_*.cpp"))
    ]
    with tempfile.TemporaryDirectory(prefix="lmcache-native-tests-") as temp:
        build = Path(temp)
        for name in names:
            backend = "fs" if name == "fs_size" else "hf3fs"
            command = [args.cxx, "-std=c++17", "-O1", "-g", "-pthread", f"-I{repo}"]
            command.extend(f"-I{path}" for path in args.include)
            if name == "hf3fs_ownership":
                command.append("-fno-elide-constructors")
            if name == "hf3fs_shutdown":
                command.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
            command.extend([
                str(repo / "csrc/storage_backends" / backend / "connector.cpp"),
                str(tests / f"test_{name}.cpp"),
            ])
            if backend == "hf3fs":
                command.append(str(tests / "hf3fs_mock.cpp"))
            command.extend(args.link)
            binary = build / name
            command.extend(["-o", str(binary)])
            subprocess.run(command, check=True, timeout=90)
            scratch = build / f"{name}-data"
            scratch.mkdir()
            environment = os.environ.copy()
            if name == "hf3fs_shutdown":
                # LeakSanitizer cannot run under the managed executor's ptrace.
                # AddressSanitizer remains enabled for use-after-free detection.
                environment["ASAN_OPTIONS"] = "detect_leaks=0"
            subprocess.run([str(binary), str(scratch)], check=True, timeout=15,
                           env=environment)
            print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
