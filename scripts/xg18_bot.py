"""XG18 football 1X2 prediction bot entry point.

The production predictor is deliberately wired to the frozen XG18 ensemble.
Do not train on validation/test rows or tune against the held-out validation set.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="XG18 frozen 1X2 prediction bot")
    parser.add_argument("--data", type=Path, required=True, help="CSV with chronological match history")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Input CSV not found: {args.data}")
    raise SystemExit(
        "Frozen predictor integration is pending source archive inspection; "
        "no predictions have been generated."
    )


if __name__ == "__main__":
    raise SystemExit(main())
