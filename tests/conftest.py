"""
Pytest configuration and fixtures for Research Assistant tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_settings():
    """Override settings for testing."""
    with patch("src.research_assistant.config.settings") as mock:
        mock.anthropic_api_key = "test-api-key"
        mock.default_model = "claude-3-5-sonnet-20241022"
        mock.temperature = 0.0
        mock.use_mock_data = True
        mock.max_research_attempts = 3
        mock.confidence_threshold = 6.0
        mock.validate_api_key.return_value = True
        yield mock


@pytest.fixture
def sample_state():
    """Create a sample state dictionary for testing."""
    return {
        "user_query": "Tell me about Apple's recent developments",
        "messages": [],
        "clarity_status": "pending",
        "clarification_request": None,
        "detected_company": None,
        "research_findings": None,
        "confidence_score": 0.0,
        "validation_result": "pending",
        "validation_feedback": None,
        "research_attempts": 0,
        "final_response": None,
        "current_agent": None,
        "error_message": None,
        "awaiting_human_input": False,
        "human_response": None,
    }


@pytest.fixture
def sample_state_with_company():
    """Create a sample state with detected company."""
    return {
        "user_query": "Tell me about Apple's recent developments",
        "messages": [],
        "clarity_status": "clear",
        "clarification_request": None,
        "detected_company": "Apple Inc.",
        "research_findings": None,
        "confidence_score": 0.0,
        "validation_result": "pending",
        "validation_feedback": None,
        "research_attempts": 0,
        "final_response": None,
        "current_agent": "clarity",
        "error_message": None,
        "awaiting_human_input": False,
        "human_response": None,
    }


@pytest.fixture
def sample_research_findings():
    """Create sample research findings dictionary."""
    return {
        "company_name": "Apple Inc.",
        "recent_news": "Launched Vision Pro, expanding services revenue",
        "stock_info": "Trading at $195, up 45% YTD",
        "key_developments": "AI integration across product line",
        "raw_data": {
            "additional_info": {
                "competitors": ["Microsoft", "Google", "Samsung"],
                "industry": "Technology",
                "ceo": "Tim Cook"
            }
        },
        "sources": ["mock_data"]
    }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    mock = MagicMock()
    mock.content = '{"clarity_status": "clear", "detected_company": "Apple Inc.", "reasoning": "Clear query about Apple"}'
    return mock
