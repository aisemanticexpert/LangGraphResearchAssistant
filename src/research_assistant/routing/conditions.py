"""
Conditional routing functions for the LangGraph workflow.

These functions determine the next node in the graph based on
the current state, implementing the multi-agent routing logic.
"""

from typing import Any, Dict, Literal

from ..config import settings


def route_after_clarity(
    state: Dict[str, Any]
) -> Literal["human_clarification", "research"]:
    """
    Route after Clarity Agent based on clarity_status.

    This function determines whether to:
    - Request human clarification (if query is unclear)
    - Proceed to research (if query is clear)

    Args:
        state: Current workflow state

    Returns:
        "human_clarification" if query needs clarification
        "research" if query is clear and ready for research

    Routing Logic:
        clarity_status == "needs_clarification" -> human_clarification
        clarity_status == "clear" -> research
    """
    clarity_status = state.get("clarity_status", "pending")

    if clarity_status == "needs_clarification":
        return "human_clarification"

    return "research"


def route_after_research(
    state: Dict[str, Any]
) -> Literal["validator", "synthesis"]:
    """
    Route after Research Agent based on confidence_score.

    This function determines whether to:
    - Send to validator for quality check (if low confidence)
    - Proceed directly to synthesis (if high confidence)

    Args:
        state: Current workflow state

    Returns:
        "validator" if confidence < threshold (default 6)
        "synthesis" if confidence >= threshold

    Routing Logic:
        confidence_score < 6.0 -> validator
        confidence_score >= 6.0 -> synthesis
    """
    confidence_score = state.get("confidence_score", 0.0)
    threshold = settings.confidence_threshold

    if confidence_score < threshold:
        return "validator"

    return "synthesis"


def route_after_validation(
    state: Dict[str, Any]
) -> Literal["research", "synthesis"]:
    """
    Route after Validator Agent based on validation result and attempts.

    This function implements the feedback loop:
    - Retry research if insufficient AND attempts < max
    - Proceed to synthesis if sufficient OR max attempts reached

    Args:
        state: Current workflow state

    Returns:
        "research" if insufficient and can retry
        "synthesis" if sufficient or max attempts reached

    Routing Logic:
        validation_result == "insufficient" AND research_attempts < 3 -> research (retry)
        validation_result == "sufficient" OR research_attempts >= 3 -> synthesis
    """
    validation_result = state.get("validation_result", "pending")
    research_attempts = state.get("research_attempts", 0)
    max_attempts = settings.max_research_attempts

    # If insufficient and haven't reached max attempts, retry research
    if validation_result == "insufficient" and research_attempts < max_attempts:
        return "research"

    # Otherwise proceed to synthesis (either sufficient or max attempts)
    return "synthesis"
