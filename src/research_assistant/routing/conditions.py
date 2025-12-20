"""
conditions.py - Routing logic between agents

These functions decide where to go next based on what each agent returned.
LangGraph calls these after each node to figure out the next step.

Author: Rajesh Gupta
"""

from typing import Any, Dict, Literal
from ..config import settings


def route_after_clarity(state: Dict[str, Any]) -> Literal["human_clarification", "research"]:
    """
    After the Clarity Agent runs, where do we go?
    - If query is confusing -> ask the user
    - If query is clear -> start researching
    """
    if state.get("clarity_status") == "needs_clarification":
        return "human_clarification"
    return "research"


def route_after_research(state: Dict[str, Any]) -> Literal["validator", "synthesis"]:
    """
    After Research Agent, check the confidence score.
    - High confidence (>= 6) -> skip validation, go straight to synthesis
    - Low confidence -> have the validator check it
    """
    confidence = state.get("confidence_score", 0.0)
    if confidence < settings.confidence_threshold:
        return "validator"
    return "synthesis"


def route_after_validation(state: Dict[str, Any]) -> Literal["research", "synthesis"]:
    """
    After Validator checks the research:
    - If it's not good enough AND we haven't hit 3 tries -> retry
    - Otherwise -> synthesize what we have
    """
    result = state.get("validation_result", "pending")
    attempts = state.get("research_attempts", 0)

    if result == "insufficient" and attempts < settings.max_research_attempts:
        return "research"  # try again
    return "synthesis"  # good enough or out of retries
