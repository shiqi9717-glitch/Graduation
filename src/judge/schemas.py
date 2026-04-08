"""Data schemas for the Phase 2 judge scoring pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class JudgeInput:
    """Single input item for judge scoring."""

    record_id: str
    question: str
    answer: str
    question_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Single judge output item."""

    record_id: str
    question: str
    answer: str
    question_type: str
    score: Optional[float]
    reason: str
    success: bool
    error_message: Optional[str] = None
    raw_judgment: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "score": self.score,
            "reason": self.reason,
            "success": self.success,
            "error_message": self.error_message,
            "raw_judgment": self.raw_judgment,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class JudgeStatistics:
    """Aggregate statistics for judge results."""

    total_records: int
    successful_judgments: int
    failed_judgments: int
    success_rate: float
    average_score: Optional[float]
    score_std: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    by_question_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_records": self.total_records,
            "successful_judgments": self.successful_judgments,
            "failed_judgments": self.failed_judgments,
            "success_rate": self.success_rate,
            "average_score": self.average_score,
            "score_std": self.score_std,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "by_question_type": self.by_question_type,
        }

