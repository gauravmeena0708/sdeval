from __future__ import annotations

import sys
from textwrap import dedent


ROOT_HELP = dedent(
    """\
    Usage:
      sdeval evaluate [options]
      sdeval evaluate_bulk [options]

    Commands:
      evaluate         Run the standard sdeval evaluator (same flags as `python -m sdeval.main`).
      evaluate_bulk    Evaluate every CSV in a folder and produce Excel/visual summaries.

    Use `sdeval <command> --help` to view command-specific options.
    """
)


def _print_help() -> None:
    print(ROOT_HELP)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0

    command = args[0]
    sub_args = args[1:]

    if command == "evaluate":
        from . import main as eval_main

        return eval_main.main(sub_args)

    if command == "evaluate_bulk":
        from . import evaluate_bulk as bulk_cli

        bulk_cli.main(sub_args)
        return 0

    print(f"Unknown command '{command}'.", file=sys.stderr)
    _print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
