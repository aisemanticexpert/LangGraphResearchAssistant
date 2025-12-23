"""Tests for the Research Agent."""

import pytest
from unittest.mock import MagicMock

from src.research_assistant.agents.research_agent import ResearchAgent
from src.research_assistant.state import ResearchFindings


class TestResearchAgentInitialization:

    def test_agent_name(self):
        agent = ResearchAgent()
        assert agent.name == "ResearchAgent"

    def test_has_research_tool(self):
        agent = ResearchAgent()
        assert hasattr(agent, 'research_tool')
        assert agent.research_tool is not None

    def test_system_prompt_exists(self):
        agent = ResearchAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_mentions_research(self):
        agent = ResearchAgent()
        assert "research" in agent.system_prompt.lower()

    def test_system_prompt_has_confidence_reference(self):
        agent = ResearchAgent()
        assert "confidence" in agent.system_prompt.lower()


class TestResearchAgentJsonParsing:

    def test_parses_valid_json(self):
        agent = ResearchAgent()
        response = '{"confidence_score": 7, "analysis": {}, "gaps_identified": []}'
        result = agent._parse_json_response(response)
        assert result["confidence_score"] == 7

    def test_handles_markdown_wrapped_json(self):
        agent = ResearchAgent()
        response = '''```json
{"confidence_score": 8, "analysis": {"summary": "Good data"}}
```'''
        result = agent._parse_json_response(response)
        assert result["confidence_score"] == 8

    def test_handles_invalid_json(self):
        agent = ResearchAgent()
        result = agent._parse_json_response("Invalid JSON")
        assert result == {}


class TestResearchAgentPromptGeneration:

    def test_creates_prompt_template(self):
        agent = ResearchAgent()
        prompt = agent._create_prompt("Company: {company}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        agent = ResearchAgent()
        prompt = agent._create_prompt("Data: {data}")
        messages = prompt.format_messages(data="test")
        assert len(messages) >= 2


class TestResearchAgentLogging:

    def test_log_execution_without_details(self):
        agent = ResearchAgent()
        agent._log_execution("Test action")

    def test_log_execution_with_details(self):
        agent = ResearchAgent()
        agent._log_execution("Research completed", {"company": "Apple", "confidence": 7.5})
