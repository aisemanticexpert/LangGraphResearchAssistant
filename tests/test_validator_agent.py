"""
Tests for the Validator Agent.
Checks initialization, prompts, and utility methods.
"""

import pytest
from src.research_assistant.agents.validator_agent import ValidatorAgent
from src.research_assistant.state import ResearchFindings


class TestValidatorAgentInitialization:
    """Tests for ValidatorAgent initialization."""

    def test_agent_name(self):
        """Should have correct agent name."""
        agent = ValidatorAgent()
        assert agent.name == "ValidatorAgent"

    def test_system_prompt_exists(self):
        """Should have a non-empty system prompt."""
        agent = ValidatorAgent()
        assert agent.system_prompt
        assert len(agent.system_prompt) > 100

    def test_system_prompt_contains_evaluation_criteria(self):
        """Should contain evaluation criteria in prompt."""
        agent = ValidatorAgent()
        prompt = agent.system_prompt.lower()
        assert "relevance" in prompt or "quality" in prompt
        assert "sufficient" in prompt or "insufficient" in prompt

    def test_system_prompt_defines_output_schema(self):
        """Should define expected output fields."""
        agent = ValidatorAgent()
        assert "validation_result" in agent.system_prompt


class TestValidatorAgentFindingsFormatting:
    """Tests for research findings formatting."""

    def test_formats_dict_findings(self):
        """Should format dictionary findings correctly."""
        agent = ValidatorAgent()

        findings_dict = {
            "company_name": "Google",
            "recent_news": "AI updates",
            "stock_info": "$140",
            "key_developments": "Gemini launch"
        }

        formatted = agent._format_findings(findings_dict)
        assert "Google" in formatted or "company_name" in formatted
        assert "AI updates" in formatted or "recent_news" in formatted

    def test_formats_pydantic_findings(self):
        """Should format Pydantic ResearchFindings correctly."""
        agent = ValidatorAgent()

        findings = ResearchFindings(
            company_name="Amazon",
            recent_news="AWS growth",
            stock_info="$180",
            key_developments="AI services expansion"
        )

        formatted = agent._format_findings(findings)
        assert "Amazon" in formatted or "AWS" in formatted

    def test_handles_none_findings(self):
        """Should handle None findings gracefully."""
        agent = ValidatorAgent()
        formatted = agent._format_findings(None)
        assert "No findings" in formatted or "available" in formatted.lower()

    def test_handles_empty_findings(self):
        """Should handle empty findings."""
        agent = ValidatorAgent()
        formatted = agent._format_findings({})
        assert len(formatted) > 0


class TestValidatorAgentJsonParsing:
    """Tests for JSON response parsing."""

    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        agent = ValidatorAgent()
        response = '{"validation_result": "sufficient", "reasoning": "Good data"}'
        result = agent._parse_json_response(response)
        assert result["validation_result"] == "sufficient"

    def test_handles_markdown_wrapped_json(self):
        """Should handle JSON in markdown code blocks."""
        agent = ValidatorAgent()
        response = '''```json
{"validation_result": "insufficient", "validation_feedback": "Need more data"}
```'''
        result = agent._parse_json_response(response)
        assert result["validation_result"] == "insufficient"

    def test_handles_invalid_json(self):
        """Should return empty dict for invalid JSON."""
        agent = ValidatorAgent()
        result = agent._parse_json_response("Invalid JSON")
        assert result == {}


class TestValidatorAgentPromptGeneration:
    """Tests for prompt generation."""

    def test_creates_prompt_template(self):
        """Should create a valid prompt template."""
        agent = ValidatorAgent()
        prompt = agent._create_prompt("Query: {query}")
        assert prompt is not None

    def test_prompt_includes_system_message(self):
        """Should include system prompt in template."""
        agent = ValidatorAgent()
        prompt = agent._create_prompt("Findings: {findings}")
        messages = prompt.format_messages(findings="test")
        assert len(messages) >= 2


class TestValidatorAgentLogging:
    """Tests for logging functionality."""

    def test_log_execution_without_details(self):
        """Should log without details."""
        agent = ValidatorAgent()
        agent._log_execution("Validation started")

    def test_log_execution_with_details(self):
        """Should log with details dict."""
        agent = ValidatorAgent()
        agent._log_execution("Validation complete", {"result": "sufficient"})
