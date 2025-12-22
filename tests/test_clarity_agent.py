"""
Tests for the Clarity Agent.
Checks initialization, prompts, and utility methods.
"""

import pytest
from unittest.mock import MagicMock

from src.research_assistant.agents.clarity_agent import ClarityAgent
from src.research_assistant.state import Message


class TestClarityAgentInitialization:
    """Tests for ClarityAgent initialization."""

    def test_agent_name(self):
        """Should have correct agent name."""
        agent = ClarityAgent()
        assert agent.name == "ClarityAgent"

    def test_system_prompt_exists(self):
        """Should have a non-empty system prompt."""
        agent = ClarityAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_contains_key_instructions(self):
        """Should have key instructions in system prompt."""
        agent = ClarityAgent()
        prompt = agent.system_prompt.lower()
        assert "company" in prompt
        assert "clear" in prompt
        assert "clarification" in prompt

    def test_system_prompt_has_json_format(self):
        """Should instruct LLM to return JSON."""
        agent = ClarityAgent()
        assert "json" in agent.system_prompt.lower()

    def test_system_prompt_defines_output_schema(self):
        """Should define expected output fields."""
        agent = ClarityAgent()
        assert "clarity_status" in agent.system_prompt
        assert "detected_company" in agent.system_prompt
        assert "clarification_request" in agent.system_prompt


class TestClarityAgentPromptGeneration:
    """Tests for prompt generation."""

    def test_creates_prompt_template(self):
        """Should create a valid prompt template."""
        agent = ClarityAgent()
        prompt = agent._create_prompt("Test: {query}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        """Should include system prompt in template."""
        agent = ClarityAgent()
        prompt = agent._create_prompt("Query: {query}")
        messages = prompt.format_messages(query="test")
        assert len(messages) >= 2  # System + Human


class TestClarityAgentContextBuilding:
    """Tests for context building from messages."""

    def test_builds_empty_context(self):
        """Should handle empty message list."""
        agent = ClarityAgent()
        context = agent._build_context_from_messages([])
        assert "No previous conversation" in context

    def test_builds_context_from_messages(self):
        """Should build context from messages."""
        agent = ClarityAgent()
        messages = [
            Message(role="user", content="Tell me about Apple"),
            Message(role="assistant", content="Apple is a tech company")
        ]
        context = agent._build_context_from_messages(messages)
        assert "user" in context.lower()
        assert "Apple" in context

    def test_truncates_long_messages(self):
        """Should truncate long messages."""
        agent = ClarityAgent()
        long_content = "A" * 1000
        messages = [Message(role="user", content=long_content)]
        context = agent._build_context_from_messages(messages)
        assert len(context) < 600  # Truncated to ~500 + formatting

    def test_limits_message_count(self):
        """Should limit number of messages in context."""
        agent = ClarityAgent()
        messages = [Message(role="user", content=f"Message {i}") for i in range(20)]
        context = agent._build_context_from_messages(messages, max_messages=5)
        # Should only include last 5 messages
        assert "Message 15" in context
        assert "Message 0" not in context


class TestClarityAgentJsonParsing:
    """Tests for JSON response parsing."""

    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        agent = ClarityAgent()
        response = '{"clarity_status": "clear", "detected_company": "Apple"}'
        result = agent._parse_json_response(response)
        assert result["clarity_status"] == "clear"
        assert result["detected_company"] == "Apple"

    def test_handles_markdown_wrapped_json(self):
        """Should handle JSON in markdown code blocks."""
        agent = ClarityAgent()
        response = '''```json
{"clarity_status": "clear", "detected_company": "Tesla"}
```'''
        result = agent._parse_json_response(response)
        assert result["clarity_status"] == "clear"
        assert result["detected_company"] == "Tesla"

    def test_handles_plain_code_block(self):
        """Should handle JSON in plain code blocks."""
        agent = ClarityAgent()
        response = '''```
{"clarity_status": "needs_clarification"}
```'''
        result = agent._parse_json_response(response)
        assert result["clarity_status"] == "needs_clarification"

    def test_handles_invalid_json(self):
        """Should return empty dict for invalid JSON."""
        agent = ClarityAgent()
        result = agent._parse_json_response("This is not JSON")
        assert result == {}

    def test_handles_empty_response(self):
        """Should handle empty response."""
        agent = ClarityAgent()
        result = agent._parse_json_response("")
        assert result == {}


class TestClarityAgentLogging:
    """Tests for logging functionality."""

    def test_log_execution_without_details(self):
        """Should log without details."""
        agent = ClarityAgent()
        # Should not raise
        agent._log_execution("Test action")

    def test_log_execution_with_details(self):
        """Should log with details dict."""
        agent = ClarityAgent()
        # Should not raise
        agent._log_execution("Test action", {"key": "value"})
