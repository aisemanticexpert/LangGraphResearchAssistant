"""
Unit tests for state schema definitions.

Tests the Pydantic models used for state management
in the LangGraph workflow.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.research_assistant.state import (
    Message,
    ResearchFindings,
    ResearchAssistantState,
    add_messages,
)


class TestMessage:
    """Tests for the Message model."""

    def test_create_user_message(self):
        """Should create a valid user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_create_assistant_message(self):
        """Should create a valid assistant message."""
        msg = Message(role="assistant", content="Hi there", agent="ClarityAgent")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
        assert msg.agent == "ClarityAgent"

    def test_create_system_message(self):
        """Should create a valid system message."""
        msg = Message(role="system", content="System prompt")
        assert msg.role == "system"

    def test_invalid_role_raises_error(self):
        """Should raise error for invalid role."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")

    def test_timestamp_auto_generated(self):
        """Should auto-generate timestamp."""
        msg = Message(role="user", content="test")
        # Timestamp should be a valid ISO format
        datetime.fromisoformat(msg.timestamp)


class TestResearchFindings:
    """Tests for the ResearchFindings model."""

    def test_create_empty_findings(self):
        """Should create findings with default values."""
        findings = ResearchFindings()
        assert findings.company_name is None
        assert findings.recent_news is None
        assert findings.sources == []

    def test_create_full_findings(self):
        """Should create findings with all fields."""
        findings = ResearchFindings(
            company_name="Apple Inc.",
            recent_news="Launched Vision Pro",
            stock_info="Trading at $195",
            key_developments="AI integration",
            sources=["mock_data"]
        )
        assert findings.company_name == "Apple Inc."
        assert findings.recent_news == "Launched Vision Pro"
        assert "mock_data" in findings.sources

    def test_raw_data_storage(self):
        """Should store raw data dictionary."""
        raw = {"key": "value", "nested": {"a": 1}}
        findings = ResearchFindings(raw_data=raw)
        assert findings.raw_data == raw


class TestResearchAssistantState:
    """Tests for the main state schema."""

    def test_create_default_state(self):
        """Should create state with default values."""
        state = ResearchAssistantState()
        assert state.user_query == ""
        assert state.messages == []
        assert state.clarity_status == "pending"
        assert state.confidence_score == 0.0
        assert state.research_attempts == 0
        assert state.awaiting_human_input is False

    def test_create_state_with_query(self):
        """Should create state with user query."""
        state = ResearchAssistantState(user_query="Tell me about Apple")
        assert state.user_query == "Tell me about Apple"

    def test_clarity_status_validation(self):
        """Should validate clarity_status literal values."""
        state = ResearchAssistantState(clarity_status="clear")
        assert state.clarity_status == "clear"

        state = ResearchAssistantState(clarity_status="needs_clarification")
        assert state.clarity_status == "needs_clarification"

    def test_confidence_score_bounds(self):
        """Should enforce confidence score bounds."""
        # Valid scores
        state = ResearchAssistantState(confidence_score=0.0)
        assert state.confidence_score == 0.0

        state = ResearchAssistantState(confidence_score=10.0)
        assert state.confidence_score == 10.0

        state = ResearchAssistantState(confidence_score=5.5)
        assert state.confidence_score == 5.5

    def test_confidence_score_below_minimum_raises_error(self):
        """Should raise error for confidence below 0."""
        with pytest.raises(ValidationError):
            ResearchAssistantState(confidence_score=-1.0)

    def test_confidence_score_above_maximum_raises_error(self):
        """Should raise error for confidence above 10."""
        with pytest.raises(ValidationError):
            ResearchAssistantState(confidence_score=11.0)

    def test_research_attempts_non_negative(self):
        """Should enforce non-negative research attempts."""
        state = ResearchAssistantState(research_attempts=0)
        assert state.research_attempts == 0

        with pytest.raises(ValidationError):
            ResearchAssistantState(research_attempts=-1)

    def test_validation_result_literals(self):
        """Should validate validation_result literal values."""
        for status in ["sufficient", "insufficient", "pending"]:
            state = ResearchAssistantState(validation_result=status)
            assert state.validation_result == status


class TestAddMessages:
    """Tests for the message reducer function."""

    def test_add_messages_empty_lists(self):
        """Should handle empty lists."""
        result = add_messages([], [])
        assert result == []

    def test_add_messages_to_empty(self):
        """Should add messages to empty list."""
        msg = Message(role="user", content="Hello")
        result = add_messages([], [msg])
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_add_messages_append(self):
        """Should append messages to existing list."""
        msg1 = Message(role="user", content="First")
        msg2 = Message(role="assistant", content="Second")
        result = add_messages([msg1], [msg2])
        assert len(result) == 2
        assert result[0].content == "First"
        assert result[1].content == "Second"

    def test_add_multiple_messages(self):
        """Should append multiple messages."""
        existing = [Message(role="user", content="One")]
        new = [
            Message(role="assistant", content="Two"),
            Message(role="user", content="Three")
        ]
        result = add_messages(existing, new)
        assert len(result) == 3
