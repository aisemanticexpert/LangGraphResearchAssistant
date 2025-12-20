"""
Unit tests for routing conditions.

Tests the conditional routing functions that determine
workflow paths between agents.
"""

import pytest
from src.research_assistant.routing.conditions import (
    route_after_clarity,
    route_after_research,
    route_after_validation,
)


class TestRouteAfterClarity:
    """Tests for clarity routing function."""

    def test_routes_to_human_when_needs_clarification(self):
        """Should route to human_clarification when query is unclear."""
        state = {"clarity_status": "needs_clarification"}
        assert route_after_clarity(state) == "human_clarification"

    def test_routes_to_research_when_clear(self):
        """Should route to research when query is clear."""
        state = {"clarity_status": "clear"}
        assert route_after_clarity(state) == "research"

    def test_routes_to_research_when_pending(self):
        """Should route to research when status is pending (default behavior)."""
        state = {"clarity_status": "pending"}
        assert route_after_clarity(state) == "research"

    def test_routes_to_research_when_status_missing(self):
        """Should default to research when clarity_status is not present."""
        state = {}
        assert route_after_clarity(state) == "research"


class TestRouteAfterResearch:
    """Tests for research routing function."""

    def test_routes_to_validator_when_low_confidence(self):
        """Should route to validator when confidence < 6."""
        state = {"confidence_score": 4.0}
        assert route_after_research(state) == "validator"

    def test_routes_to_validator_when_confidence_is_5(self):
        """Should route to validator when confidence is exactly 5."""
        state = {"confidence_score": 5.0}
        assert route_after_research(state) == "validator"

    def test_routes_to_validator_when_confidence_is_5_9(self):
        """Should route to validator when confidence is 5.9."""
        state = {"confidence_score": 5.9}
        assert route_after_research(state) == "validator"

    def test_routes_to_synthesis_when_confidence_is_6(self):
        """Should route to synthesis when confidence is exactly 6."""
        state = {"confidence_score": 6.0}
        assert route_after_research(state) == "synthesis"

    def test_routes_to_synthesis_when_high_confidence(self):
        """Should route to synthesis when confidence >= 6."""
        state = {"confidence_score": 8.0}
        assert route_after_research(state) == "synthesis"

    def test_routes_to_synthesis_when_confidence_is_10(self):
        """Should route to synthesis when confidence is maximum."""
        state = {"confidence_score": 10.0}
        assert route_after_research(state) == "synthesis"

    def test_routes_to_validator_when_confidence_missing(self):
        """Should route to validator when confidence_score is missing (defaults to 0)."""
        state = {}
        assert route_after_research(state) == "validator"


class TestRouteAfterValidation:
    """Tests for validation routing function."""

    def test_routes_to_research_when_insufficient_and_retries_available(self):
        """Should retry research when insufficient and attempts < 3."""
        state = {
            "validation_result": "insufficient",
            "research_attempts": 1
        }
        assert route_after_validation(state) == "research"

    def test_routes_to_research_when_insufficient_attempt_2(self):
        """Should retry research on second attempt."""
        state = {
            "validation_result": "insufficient",
            "research_attempts": 2
        }
        assert route_after_validation(state) == "research"

    def test_routes_to_synthesis_when_sufficient(self):
        """Should proceed to synthesis when validation is sufficient."""
        state = {
            "validation_result": "sufficient",
            "research_attempts": 1
        }
        assert route_after_validation(state) == "synthesis"

    def test_routes_to_synthesis_when_max_attempts_reached(self):
        """Should proceed to synthesis when max attempts (3) reached."""
        state = {
            "validation_result": "insufficient",
            "research_attempts": 3
        }
        assert route_after_validation(state) == "synthesis"

    def test_routes_to_synthesis_when_attempts_exceed_max(self):
        """Should proceed to synthesis when attempts exceed max."""
        state = {
            "validation_result": "insufficient",
            "research_attempts": 5
        }
        assert route_after_validation(state) == "synthesis"

    def test_routes_to_synthesis_when_sufficient_regardless_of_attempts(self):
        """Should proceed to synthesis when sufficient even with low attempts."""
        state = {
            "validation_result": "sufficient",
            "research_attempts": 0
        }
        assert route_after_validation(state) == "synthesis"

    def test_routes_to_synthesis_when_pending(self):
        """Should proceed to synthesis when validation is pending."""
        state = {
            "validation_result": "pending",
            "research_attempts": 1
        }
        assert route_after_validation(state) == "synthesis"

    def test_routes_to_synthesis_when_missing_values(self):
        """Should handle missing values gracefully."""
        state = {}
        assert route_after_validation(state) == "synthesis"
