#!/usr/bin/env python3
"""Run the full suite, every numbered block separately, or one prototype test."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="run the full repository test suite")
    selection.add_argument(
        "--matrix",
        action="store_true",
        help="run each of the 38 section and 13 algorithm tests separately",
    )
    selection.add_argument("--section", type=int, metavar="N", help="run formula section N tests (1-38)")
    selection.add_argument("--algorithm", type=int, metavar="N", help="run algorithm N tests (1-13)")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    if args.all:
        targets = [ROOT / "tests"]
    elif args.matrix:
        targets = [
            ROOT / "tests" / "prototype_sections" / f"test_section_{number:02d}.py"
            for number in range(1, 39)
        ] + [
            ROOT / "tests" / "prototype_algorithms" / f"test_algorithm_{number:02d}.py"
            for number in range(1, 14)
        ]
    elif args.section is not None:
        if not 1 <= args.section <= 38:
            parser.error("--section must be between 1 and 38")
        targets = [
            ROOT / "tests" / "prototype_sections" / f"test_section_{args.section:02d}.py"
        ]
    else:
        if not 1 <= args.algorithm <= 13:
            parser.error("--algorithm must be between 1 and 13")
        targets = [
            ROOT / "tests" / "prototype_algorithms" / f"test_algorithm_{args.algorithm:02d}.py"
        ]

    missing = [path for path in targets if not path.exists()]
    if missing:
        parser.error("test file does not exist: " + ", ".join(str(path.relative_to(ROOT)) for path in missing))

    if args.matrix:
        failures = []
        for path in targets:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(path)],
                cwd=ROOT,
                check=False,
            )
            if result.returncode:
                failures.append(path.relative_to(ROOT))
        if failures:
            print("Prototype test failures:")
            for path in failures:
                print(f"  {path}")
            return 1
        print("All 38 section and 13 algorithm test files passed individually.")
        return 0

    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *(str(path) for path in targets)],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
