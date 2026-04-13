"""Compatibility wrapper for legacy imports.

Prefer importing `StatsAnalyzer` from `src.stats` or `src.analyzer.stats_analyzer`.
"""

from src.analyzer.stats_analyzer import StatsAnalyzer

__all__ = ["StatsAnalyzer"]
