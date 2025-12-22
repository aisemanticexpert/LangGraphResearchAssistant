"""
Tests for the Research Agent.
Checks initialization, prompts, and utility methods.
"""

import pytest
from unittest.mock import MagicMock

from src.research_assistant.agents.research_agent import ResearchAgent
from src.research_assistant.state import ResearchFindings


class TestResearchAgentInitialization:
    """Tests for ResearchAgent initialization."""

    def test_agent_name(self):
        """Should have correct agent name."""
        agent = ResearchAgent()
        assert agent.name == "ResearchAgent"

    def test_has_research_tool(self):
        """Should initialize with research tool."""
        agent = ResearchAgent()
        assert hasattr(agent, 'research_tool')
        assert agent.research_tool is not None

    def test_system_prompt_exists(self):
        """Should have a non-empty system prompt."""
        agent = ResearchAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_mentions_research(self):
        """Should mention research in system prompt."""
        agent = ResearchAgent()
        assert "research" in agent.system_prompt.lower()

    def test_system_prompt_has_confidence_scale(self):
        """Should have confidence scoring instructions."""
        agent = ResearchAgent()
        assert "confidence" in agent.system_prompt.lower()
        assert "0-10" in agent.system_prompt or "0" in agent.system_prompt

    def test_system_prompt_defines_output_schema(self):
        """Should define expected output fields."""
        agent = ResearchAgent()
        assert "confidence_score" in agent.system_prompt
        assert "analysis" in agent.system_prompt


class TestResearchAgentJsonParsing:
    """Tests for JSON response parsing."""

    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        agent = ResearchAgent()
        response = '{"confidence_score": 7, "analysis": {}, "gaps_identified": []}'
        result = agent._parse_json_response(response)
        assert result["confidence_score"] == 7

    def test_handles_markdown_wrapped_json(self):
        """Should handle JSON in markdown code blocks."""
        agent = ResearchAgent()
        response = '''```json
{"confidence_score": 8, "analysis": {"summary": "Good data"}}
```'''
        result = agent._parse_json_response(response)
        assert result["confidence_score"] == 8

    def test_handles_invalid_json(self):
        """Should return empty dict for invalid JSON."""
        agent = ResearchAgent()
        result = agent._parse_json_response("Invalid JSON")
        assert result == {}


class TestResearchAgentConfidenceBreakdownFormatting:
    """Tests for confidence breakdown formatting."""

    def test_formats_high_scores(self):
        """Should format high scoring components."""
        agent = ResearchAgent()
        from src.research_assistant.utils.confidence import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            data_completeness_score=8.0,
            data_freshness_score=7.5,
            query_relevance_score=6.0,
            data_specificity_score=5.0,
            source_quality_score=7.0
        )

        summary = agent._format_confidence_breakdown(breakdown)
        assert "Strong" in summary or "Scoring complete" in summary

    def test_formats_low_scores(self):
        """Should format low scoring components."""
        agent = ResearchAgent()
        from src.research_assistant.utils.confidence import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            data_completeness_score=3.0,
            data_freshness_score=2.0,
            query_relevance_score=4.0,
            gaps=["missing stock info"]
        )

        summary = agent._format_confidence_breakdown(breakdown)
        assert "Weak" in summary or "Gaps" in summary

    def test_formats_gaps(self):
        """Should include gaps in summary."""
        agent = ResearchAgent()
        from src.research_assistant.utils.confidence import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            gaps=["missing news", "no stock info"]
        )

        summary = agent._format_confidence_breakdown(breakdown)
        assert "Gaps" in summary or "Scoring complete" in summary


class TestResearchAgentPromptGeneration:
    """Tests for prompt generation."""

    def test_creates_prompt_template(self):
        """Should create a valid prompt template."""
        agent = ResearchAgent()
        prompt = agent._create_prompt("Company: {company}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        """Should include system prompt in template."""
        agent = ResearchAgent()
        prompt = agent._create_prompt("Data: {data}")
        messages = prompt.format_messages(data="test")
        assert len(messages) >= 2


class TestResearchAgentLogging:
    """Tests for logging functionality."""

    def test_log_execution_without_details(self):
        """Should log without details."""
        agent = ResearchAgent()
        agent._log_execution("Test action")

    def test_log_execution_with_details(self):
        """Should log with details dict."""
        agent = ResearchAgent()
        agent._log_execution("Research completed", {"company": "Apple", "confidence": 7.5})
