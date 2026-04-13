"""Compatibility wrapper for legacy objective case-extractor imports.

Prefer importing `ObjectiveCaseStudyExtractor` from
`src.analyzer.objective_case_extractor`.
"""

from src.analyzer.objective_case_extractor import (
    HighValueCaseExtractor,
    ObjectiveCaseStudyExtractor,
    main,
    parse_args,
)

__all__ = [
    "HighValueCaseExtractor",
    "ObjectiveCaseStudyExtractor",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
