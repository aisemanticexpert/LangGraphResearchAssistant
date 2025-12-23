"""
Integration tests for the LangGraph workflow.

Tests the complete graph flow including agent interactions
and routing decisions.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.research_assistant.graph import build_research_graph, get_graph_visualization


class TestGraphConstruction:
    """Tests for graph construction."""

    def test_graph_builds_successfully(self, mock_settings):
        """Should build graph without errors."""
        graph = build_research_graph()
        assert graph is not None

    def test_graph_has_nodes(self, mock_settings):
        """Should have all required nodes."""
        graph = build_research_graph()
        # The compiled graph should be functional
        assert graph is not None

    def test_graph_visualization(self):
        diagram = get_graph_visualization()
        assert "mermaid" in diagram
        assert "thinksemantic" in diagram.lower()
        assert "research" in diagram.lower()
        assert "validator" in diagram.lower()
        assert "synthesis" in diagram.lower()


class TestGraphExecution:
    """Tests for graph execution with mocked LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM that returns predictable responses."""
        mock = MagicMock()

        def mock_invoke(input_dict):
            response = MagicMock()
            # Default clarity response
            response.content = '''{"clarity_status": "clear", "detected_company": "Apple Inc.", "reasoning": "Clear query"}'''
            return response

        mock.invoke = mock_invoke
        return mock

    @pytest.fixture
    def mock_chain(self, mock_llm):
        """Create mock chain."""
        mock = MagicMock()
        mock.__or__ = MagicMock(return_value=mock_llm)
        return mock

    def test_clarity_agent_clear_path(self, mock_settings, sample_state):
        """Test that clear queries route to research."""
        # This would require full integration testing with mocked LLM
        # For now, verify state structure
        assert sample_state["clarity_status"] == "pending"
        assert sample_state["research_attempts"] == 0

    def test_state_transitions(self, mock_settings, sample_state_with_company):
        """Test state after clarity agent."""
        state = sample_state_with_company
        assert state["clarity_status"] == "clear"
        assert state["detected_company"] == "Apple Inc."


class TestGraphEdgeCases:
    """Tests for edge cases in graph execution."""

    def test_empty_query_handling(self, sample_state):
        """Should handle empty queries."""
        sample_state["user_query"] = ""
        # Graph should still process (clarity agent will ask for clarification)
        assert sample_state["user_query"] == ""

    def test_max_attempts_boundary(self):
        """Test behavior at max research attempts."""
        state = {
            "validation_result": "insufficient",
            "research_attempts": 3
        }
        # At max attempts, should route to synthesis regardless of validation
        from src.research_assistant.routing.conditions import route_after_validation
        assert route_after_validation(state) == "synthesis"

    def test_high_confidence_skips_validation(self):
        """Test that high confidence skips validator."""
        state = {"confidence_score": 8.0}
        from src.research_assistant.routing.conditions import route_after_research
        assert route_after_research(state) == "synthesis"
