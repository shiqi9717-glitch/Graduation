"""Compatibility wrapper for legacy pair-case report imports.

Prefer importing from `src.stats.pair_case_report`.
"""

from src.stats.pair_case_report import (
    PairRecord,
    SideRecord,
    build_pairs,
    generate_report,
    load_side_records,
    main,
    parse_args,
)

__all__ = [
    "SideRecord",
    "PairRecord",
    "load_side_records",
    "build_pairs",
    "generate_report",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()
