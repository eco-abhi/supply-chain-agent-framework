"""Run evaluation experiments. Default: exp1–exp5 in order."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        nargs="+",
        choices=["exp1", "exp2", "exp3", "exp4", "exp5"],
        default=["exp1", "exp2", "exp3", "exp4", "exp5"],
    )
    args = p.parse_args(argv)

    if "exp1" in args.only:
        from src.experiments.exp1 import run_exp1

        run_exp1()
    if "exp2" in args.only:
        from src.experiments.exp2 import run_exp2

        run_exp2()
    if "exp3" in args.only:
        from src.experiments.exp3 import run_exp3

        run_exp3()
    if "exp4" in args.only:
        from src.experiments.exp4 import run_exp4

        run_exp4()
    if "exp5" in args.only:
        from src.experiments.exp5 import run_exp5

        run_exp5()


if __name__ == "__main__":
    main()
