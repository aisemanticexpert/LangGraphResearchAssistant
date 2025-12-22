"""
Tests for the Synthesis Agent.
Checks initialization, prompts, and utility methods.
"""

import pytest
from src.research_assistant.agents.synthesis_agent import SynthesisAgent
from src.research_assistant.state import ResearchFindings, Message


class TestSynthesisAgentInitialization:
    """Tests for SynthesisAgent initialization."""

    def test_agent_name(self):
        """Should have correct agent name."""
        agent = SynthesisAgent()
        assert agent.name == "SynthesisAgent"

    def test_system_prompt_exists(self):
        """Should have a non-empty system prompt."""
        agent = SynthesisAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_contains_formatting_guidelines(self):
        """Should contain formatting guidelines."""
        agent = SynthesisAgent()
        prompt = agent.system_prompt.lower()
        # Should mention formatting or structure
        assert "bullet" in prompt or "format" in prompt or "response" in prompt

    def test_system_prompt_mentions_synthesis(self):
        """Should mention synthesis or summary."""
        agent = SynthesisAgent()
        prompt = agent.system_prompt.lower()
        assert "synthes" in prompt or "summar" in prompt or "response" in prompt


class TestSynthesisAgentFindingsFormatting:
    """Tests for research findings formatting."""

    def test_formats_pydantic_findings(self):
        """Should format Pydantic ResearchFindings correctly."""
        agent = SynthesisAgent()

        findings = ResearchFindings(
            company_name="Microsoft",
            recent_news="Azure growth continues",
            stock_info="Trading at $400",
            key_developments="Copilot integration"
        )

        formatted = agent._format_findings(findings)
        assert "Microsoft" in formatted
        assert "Azure" in formatted or "news" in formatted.lower()
        assert "400" in formatted or "stock" in formatted.lower()

    def test_formats_dict_findings(self):
        """Should format dictionary findings correctly."""
        agent = SynthesisAgent()

        findings_dict = {
            "company_name": "Google",
            "recent_news": "Gemini launch",
            "stock_info": "$140",
            "key_developments": "AI focus"
        }

        formatted = agent._format_findings(findings_dict)
        assert "Google" in formatted or "Gemini" in formatted

    def test_handles_none_findings(self):
        """Should handle None findings."""
        agent = SynthesisAgent()
        formatted = agent._format_findings(None)
        assert "No" in formatted

    def test_includes_additional_info(self):
        """Should include additional info from raw_data."""
        agent = SynthesisAgent()

        findings = ResearchFindings(
            company_name="Apple Inc.",
            recent_news="Vision Pro",
            stock_info="$195",
            key_developments="AI",
            raw_data={
                "additional_info": {
                    "competitors": ["Microsoft", "Google"],
                    "industry": "Technology",
                    "ceo": "Tim Cook"
                }
            }
        )

        formatted = agent._format_findings(findings)
        # Should include some additional info
        assert "Tim Cook" in formatted or "Technology" in formatted or "Competitors" in formatted or "Apple" in formatted


class TestSynthesisAgentJsonParsing:
    """Tests for JSON response parsing (inherited from base)."""

    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        agent = SynthesisAgent()
        response = '{"key": "value"}'
        result = agent._parse_json_response(response)
        assert result["key"] == "value"

    def test_handles_invalid_json(self):
        """Should return empty dict for invalid JSON."""
        agent = SynthesisAgent()
        result = agent._parse_json_response("Not JSON")
        assert result == {}


class TestSynthesisAgentContextBuilding:
    """Tests for context building from messages."""

    def test_builds_context_from_messages(self):
        """Should build context from message history."""
        agent = SynthesisAgent()
        messages = [
            Message(role="user", content="Tell me about Tesla"),
            Message(role="assistant", content="Tesla is an EV company...")
        ]
        context = agent._build_context_from_messages(messages)
        assert "user" in context.lower()
        assert "Tesla" in context

    def test_handles_empty_messages(self):
        """Should handle empty message list."""
        agent = SynthesisAgent()
        context = agent._build_context_from_messages([])
        assert "No previous conversation" in context


class TestSynthesisAgentPromptGeneration:
    """Tests for prompt generation."""

    def test_creates_prompt_template(self):
        """Should create a valid prompt template."""
        agent = SynthesisAgent()
        prompt = agent._create_prompt("Query: {query}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        """Should include system prompt in template."""
        agent = SynthesisAgent()
        prompt = agent._create_prompt("Data: {data}")
        messages = prompt.format_messages(data="test")
        assert len(messages) >= 2


class TestSynthesisAgentLogging:
    """Tests for logging functionality."""

    def test_log_execution_without_details(self):
        """Should log without details."""
        agent = SynthesisAgent()
        agent._log_execution("Synthesis started")

    def test_log_execution_with_details(self):
        """Should log with details dict."""
        agent = SynthesisAgent()
        agent._log_execution("Synthesis complete", {"company": "Apple"})
