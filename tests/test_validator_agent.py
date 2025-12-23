"""Tests for the Validator Agent."""

import pytest
from src.research_assistant.agents.validator_agent import ValidatorAgent
from src.research_assistant.state import ResearchFindings, NewsItem, StockInfo


class TestValidatorAgentInitialization:

    def test_agent_name(self):
        agent = ValidatorAgent()
        assert agent.name == "ValidatorAgent"

    def test_system_prompt_exists(self):
        agent = ValidatorAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_contains_evaluation_criteria(self):
        agent = ValidatorAgent()
        prompt = agent.system_prompt.lower()
        assert "relevance" in prompt or "quality" in prompt
        assert "sufficient" in prompt or "insufficient" in prompt

    def test_system_prompt_defines_output_schema(self):
        agent = ValidatorAgent()
        assert "validation_result" in agent.system_prompt


class TestValidatorAgentJsonParsing:

    def test_parses_valid_json(self):
        agent = ValidatorAgent()
        response = '{"validation_result": "sufficient", "reasoning": "Good data"}'
        result = agent._parse_json_response(response)
        assert result["validation_result"] == "sufficient"

    def test_handles_markdown_wrapped_json(self):
        agent = ValidatorAgent()
        response = '''```json
{"validation_result": "insufficient", "validation_feedback": "Need more data"}
```'''
        result = agent._parse_json_response(response)
        assert result["validation_result"] == "insufficient"

    def test_handles_invalid_json(self):
        agent = ValidatorAgent()
        result = agent._parse_json_response("Invalid JSON")
        assert result == {}


class TestValidatorAgentPromptGeneration:

    def test_creates_prompt_template(self):
        agent = ValidatorAgent()
        prompt = agent._create_prompt("Query: {query}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        agent = ValidatorAgent()
        prompt = agent._create_prompt("Findings: {findings}")
        messages = prompt.format_messages(findings="test")
        assert len(messages) >= 2


class TestValidatorAgentLogging:

    def test_log_execution_without_details(self):
        agent = ValidatorAgent()
        agent._log_execution("Validation started")

    def test_log_execution_with_details(self):
        agent = ValidatorAgent()
        agent._log_execution("Validation complete", {"result": "sufficient"})
