"""
Track whether retrying research actually helps.

When the validator sends results back for another try, does the
second (or third) attempt actually produce better results? This
module tracks that so we can learn what's working.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from statistics import mean, stdev

logger = logging.getLogger(__name__)


@dataclass
class RetryAttempt:
    """Info about one research attempt."""

    attempt_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence_score: float = 0.0
    validation_result: str = "pending"
    gaps_identified: List[str] = field(default_factory=list)
    gaps_from_previous: List[str] = field(default_factory=list)  # what we were supposed to fix
    gaps_addressed: List[str] = field(default_factory=list)  # what we actually fixed
    feedback_received: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp,
            "confidence_score": self.confidence_score,
            "validation_result": self.validation_result,
            "gaps_identified": self.gaps_identified,
            "gaps_from_previous": self.gaps_from_previous,
            "gaps_addressed": self.gaps_addressed,
            "feedback_received": self.feedback_received,
        }


@dataclass
class RetryEffectivenessReport:
    """Summary of how well retries worked for a conversation."""

    thread_id: str
    company: str
    total_attempts: int
    initial_confidence: float
    final_confidence: float
    confidence_improvement: float
    total_gaps_identified: int
    gaps_resolved: int
    gap_resolution_rate: float
    retry_was_worthwhile: bool
    attempts: List[RetryAttempt] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "company": self.company,
            "total_attempts": self.total_attempts,
            "initial_confidence": round(self.initial_confidence, 2),
            "final_confidence": round(self.final_confidence, 2),
            "confidence_improvement": round(self.confidence_improvement, 2),
            "total_gaps_identified": self.total_gaps_identified,
            "gaps_resolved": self.gaps_resolved,
            "gap_resolution_rate": round(self.gap_resolution_rate, 2),
            "retry_was_worthwhile": self.retry_was_worthwhile,
            "attempts": [a.to_dict() for a in self.attempts],
            "recommendations": self.recommendations,
        }


class RetryEffectivenessTracker:
    """
    Keeps track of retry attempts to see if they actually help.

    The idea: if retries aren't improving confidence or fixing gaps,
    maybe we should tweak our retry strategy or stop earlier.
    """

    # when is a retry "worth it"?
    MIN_CONFIDENCE_IMPROVEMENT = 0.5  # need at least this much gain
    MIN_GAP_RESOLUTION_RATE = 0.3     # or fix at least 30% of gaps

    def __init__(self):
        self._sessions: Dict[str, List[RetryAttempt]] = {}  # active sessions
        self._historical_reports: List[RetryEffectivenessReport] = []  # for analysis

    def start_session(self, thread_id: str) -> None:
        """Start tracking retries for a conversation."""
        self._sessions[thread_id] = []
        logger.debug(f"Started retry tracking session for thread {thread_id}")

    def record_attempt(
        self,
        thread_id: str,
        attempt_number: int,
        confidence_score: float,
        validation_result: str = "pending",
        gaps_identified: Optional[List[str]] = None,
        feedback_received: Optional[str] = None
    ) -> RetryAttempt:
        """Log a research attempt so we can analyze later."""
        if thread_id not in self._sessions:
            self.start_session(thread_id)

        gaps_identified = gaps_identified or []

        # figure out what gaps we were supposed to fix vs what we fixed
        gaps_from_previous = []
        gaps_addressed = []

        if attempt_number > 1 and self._sessions[thread_id]:
            prev_attempt = self._sessions[thread_id][-1]
            gaps_from_previous = prev_attempt.gaps_identified

            # if a gap from before isn't in current gaps, we fixed it
            current_gap_text = " ".join(gaps_identified).lower()
            for prev_gap in gaps_from_previous:
                if prev_gap.lower() not in current_gap_text:
                    gaps_addressed.append(prev_gap)

        attempt = RetryAttempt(
            attempt_number=attempt_number,
            confidence_score=confidence_score,
            validation_result=validation_result,
            gaps_identified=gaps_identified,
            gaps_from_previous=gaps_from_previous,
            gaps_addressed=gaps_addressed,
            feedback_received=feedback_received,
        )

        self._sessions[thread_id].append(attempt)

        logger.info(f"Recorded attempt {attempt_number} for thread {thread_id}: "
                   f"confidence={confidence_score:.2f}, gaps={len(gaps_identified)}")

        return attempt

    def generate_report(
        self,
        thread_id: str,
        company: str
    ) -> RetryEffectivenessReport:
        """Create a summary of how the retries went."""
        attempts = self._sessions.get(thread_id, [])

        if not attempts:
            return RetryEffectivenessReport(
                thread_id=thread_id,
                company=company,
                total_attempts=0,
                initial_confidence=0.0,
                final_confidence=0.0,
                confidence_improvement=0.0,
                total_gaps_identified=0,
                gaps_resolved=0,
                gap_resolution_rate=0.0,
                retry_was_worthwhile=False,
                recommendations=["No retry attempts recorded"]
            )

        # crunch the numbers
        initial_confidence = attempts[0].confidence_score
        final_confidence = attempts[-1].confidence_score
        confidence_improvement = final_confidence - initial_confidence

        # count unique gaps and how many got fixed
        all_gaps = set()
        resolved_gaps = set()

        for attempt in attempts:
            all_gaps.update(attempt.gaps_identified)
            resolved_gaps.update(attempt.gaps_addressed)

        total_gaps = len(all_gaps)
        gaps_resolved = len(resolved_gaps)
        gap_resolution_rate = gaps_resolved / total_gaps if total_gaps > 0 else 1.0

        # was it worth retrying?
        retry_worthwhile = (
            confidence_improvement >= self.MIN_CONFIDENCE_IMPROVEMENT or
            gap_resolution_rate >= self.MIN_GAP_RESOLUTION_RATE
        )

        recommendations = self._generate_recommendations(
            attempts, confidence_improvement, gap_resolution_rate
        )

        report = RetryEffectivenessReport(
            thread_id=thread_id,
            company=company,
            total_attempts=len(attempts),
            initial_confidence=initial_confidence,
            final_confidence=final_confidence,
            confidence_improvement=confidence_improvement,
            total_gaps_identified=total_gaps,
            gaps_resolved=gaps_resolved,
            gap_resolution_rate=gap_resolution_rate,
            retry_was_worthwhile=retry_worthwhile,
            attempts=attempts,
            recommendations=recommendations,
        )

        self._historical_reports.append(report)

        logger.info(f"Generated retry effectiveness report for thread {thread_id}: "
                   f"worthwhile={retry_worthwhile}, improvement={confidence_improvement:.2f}")

        return report

    def _generate_recommendations(
        self,
        attempts: List[RetryAttempt],
        confidence_improvement: float,
        gap_resolution_rate: float
    ) -> List[str]:
        """Figure out what to suggest based on the numbers."""
        recommendations = []

        if len(attempts) == 1:
            recommendations.append("Single attempt - no retry analysis available")
            return recommendations

        if confidence_improvement < 0:
            recommendations.append(
                "Confidence decreased across retries - consider stopping earlier"
            )
        elif confidence_improvement < self.MIN_CONFIDENCE_IMPROVEMENT:
            recommendations.append(
                "Minimal confidence improvement - retry may not be cost-effective"
            )

        if gap_resolution_rate < self.MIN_GAP_RESOLUTION_RATE:
            recommendations.append(
                "Low gap resolution rate - feedback may not be actionable"
            )

        # watch for up-down-up patterns
        if len(attempts) >= 3:
            confidences = [a.confidence_score for a in attempts]
            if confidences[1] > confidences[0] and confidences[2] < confidences[1]:
                recommendations.append(
                    "Confidence oscillated - retries may be introducing noise"
                )

        # is the feedback actually helping?
        feedback_attempts = [a for a in attempts if a.feedback_received]
        if feedback_attempts and gap_resolution_rate < 0.5:
            recommendations.append(
                "Validation feedback not effectively addressing gaps"
            )

        if not recommendations:
            recommendations.append("Retry strategy appears effective")

        return recommendations

    def get_historical_stats(self) -> Dict[str, Any]:
        """Aggregate stats from all sessions we've tracked."""
        if not self._historical_reports:
            return {"message": "No historical data available"}

        reports = self._historical_reports

        worthwhile_count = sum(1 for r in reports if r.retry_was_worthwhile)
        improvements = [r.confidence_improvement for r in reports]
        resolution_rates = [r.gap_resolution_rate for r in reports]

        return {
            "total_sessions": len(reports),
            "worthwhile_retries": worthwhile_count,
            "worthwhile_rate": worthwhile_count / len(reports),
            "avg_confidence_improvement": mean(improvements),
            "std_confidence_improvement": stdev(improvements) if len(improvements) > 1 else 0,
            "avg_gap_resolution_rate": mean(resolution_rates),
            "avg_attempts_per_session": mean(r.total_attempts for r in reports),
        }

    def clear_session(self, thread_id: str) -> None:
        """Forget about a session."""
        if thread_id in self._sessions:
            del self._sessions[thread_id]


# shared instance
_tracker_instance: Optional[RetryEffectivenessTracker] = None


def get_retry_tracker() -> RetryEffectivenessTracker:
    """Grab the shared tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = RetryEffectivenessTracker()
    return _tracker_instance


def record_retry_attempt(
    thread_id: str,
    attempt_number: int,
    confidence_score: float,
    validation_result: str = "pending",
    gaps_identified: Optional[List[str]] = None,
    feedback_received: Optional[str] = None
) -> RetryAttempt:
    """Quick way to log an attempt."""
    tracker = get_retry_tracker()
    return tracker.record_attempt(
        thread_id, attempt_number, confidence_score,
        validation_result, gaps_identified, feedback_received
    )


def get_retry_report(thread_id: str, company: str) -> RetryEffectivenessReport:
    """Quick way to get a report."""
    tracker = get_retry_tracker()
    return tracker.generate_report(thread_id, company)
