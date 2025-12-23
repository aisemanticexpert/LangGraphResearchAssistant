"""Tests for the Synthesis Agent."""

import pytest
from src.research_assistant.agents.synthesis_agent import SynthesisAgent
from src.research_assistant.state import ResearchFindings, Message, NewsItem, StockInfo


class TestSynthesisAgentInitialization:
    """Tests for SynthesisAgent initialization."""

    def test_agent_name(self):
        agent = SynthesisAgent()
        assert agent.name == "SynthesisAgent"

    def test_system_prompt_exists(self):
        agent = SynthesisAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_contains_formatting_guidelines(self):
        agent = SynthesisAgent()
        prompt = agent.system_prompt.lower()
        assert "bullet" in prompt or "format" in prompt or "response" in prompt

    def test_system_prompt_mentions_synthesis(self):
        agent = SynthesisAgent()
        prompt = agent.system_prompt.lower()
        assert "synthes" in prompt or "summar" in prompt or "response" in prompt


class TestSynthesisAgentFindingsFormatting:
    """Tests for research findings formatting."""

    def test_formats_pydantic_findings(self):
        agent = SynthesisAgent()

        findings = ResearchFindings(
            company_name="Microsoft",
            recent_news=[NewsItem(title="Azure growth continues")],
            stock_info=StockInfo(price=400.0),
            key_developments=["Copilot integration"]
        )

        formatted = agent._format_findings(findings, intent="general")
        assert "Microsoft" in formatted

    def test_formats_dict_findings(self):
        agent = SynthesisAgent()

        findings_dict = {
            "company_name": "Google",
            "recent_news": "Gemini launch",
            "stock_info": "$140",
            "key_developments": "AI focus"
        }

        formatted = agent._format_findings(findings_dict, intent="general")
        assert "Google" in formatted or "Gemini" in formatted

    def test_handles_none_findings(self):
        agent = SynthesisAgent()
        formatted = agent._format_findings(None, intent="general")
        assert "No" in formatted or formatted != ""

    def test_includes_additional_info(self):
        agent = SynthesisAgent()

        findings = ResearchFindings(
            company_name="Apple Inc.",
            recent_news=[NewsItem(title="Vision Pro")],
            stock_info=StockInfo(price=195.0),
            key_developments=["AI"],
            raw_data={
                "additional_info": {
                    "competitors": ["Microsoft", "Google"],
                    "industry": "Technology",
                    "ceo": "Tim Cook"
                }
            }
        )

        formatted = agent._format_findings(findings, intent="general")
        assert "Apple" in formatted


class TestSynthesisAgentJsonParsing:
    """Tests for JSON response parsing."""

    def test_parses_valid_json(self):
        agent = SynthesisAgent()
        response = '{"key": "value"}'
        result = agent._parse_json_response(response)
        assert result["key"] == "value"

    def test_handles_invalid_json(self):
        agent = SynthesisAgent()
        result = agent._parse_json_response("Not JSON")
        assert result == {}


class TestSynthesisAgentContextBuilding:
    """Tests for context building from messages."""

    def test_builds_context_from_messages(self):
        agent = SynthesisAgent()
        messages = [
            Message(role="user", content="Tell me about Tesla"),
            Message(role="assistant", content="Tesla is an EV company...")
        ]
        context = agent._build_context_from_messages(messages)
        assert "user" in context.lower()
        assert "Tesla" in context

    def test_handles_empty_messages(self):
        agent = SynthesisAgent()
        context = agent._build_context_from_messages([])
        assert "No previous conversation" in context


class TestSynthesisAgentPromptGeneration:
    """Tests for prompt generation."""

    def test_creates_prompt_template(self):
        agent = SynthesisAgent()
        prompt = agent._create_prompt("Query: {query}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        agent = SynthesisAgent()
        prompt = agent._create_prompt("Data: {data}")
        messages = prompt.format_messages(data="test")
        assert len(messages) >= 2


class TestSynthesisAgentLogging:
    """Tests for logging functionality."""

    def test_log_execution_without_details(self):
        agent = SynthesisAgent()
        agent._log_execution("Synthesis started")

    def test_log_execution_with_details(self):
        agent = SynthesisAgent()
        agent._log_execution("Synthesis complete", {"company": "Apple"})
