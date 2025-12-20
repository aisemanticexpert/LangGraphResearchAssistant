"""
Tests for the FastAPI REST API.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_app():
    """Create a mock ResearchAssistantApp."""
    mock = MagicMock()
    mock.start_conversation.return_value = {
        "thread_id": "test-thread-123",
        "final_response": "Apple is a technology company...",
        "interrupted": False,
    }
    mock.continue_conversation.return_value = {
        "thread_id": "test-thread-123",
        "final_response": "Apple's competitors include...",
        "interrupted": False,
    }
    mock.get_conversation_state.return_value = {
        "detected_company": "Apple Inc.",
        "clarity_status": "clear",
        "research_attempts": 1,
        "confidence_score": 8.0,
        "validation_result": "sufficient",
        "final_response": "Apple is a technology company...",
        "messages": [],
    }
    return mock


@pytest.fixture
def client(mock_app):
    """Create a test client with mocked dependencies."""
    with patch("src.research_assistant.api.get_app", return_value=mock_app):
        with patch("src.research_assistant.api.query_cache") as mock_cache:
            mock_cache.get.return_value = None
            mock_cache.get_stats.return_value = {
                "total_entries": 0,
                "valid_entries": 0,
                "max_size": 100,
                "ttl_seconds": 3600,
                "enabled": True,
            }
            from src.research_assistant.api import app
            yield TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Research Assistant API"
        assert "version" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cache_stats" in data

    def test_list_companies(self, client):
        """Test listing available companies."""
        response = client.get("/companies")
        assert response.status_code == 200
        data = response.json()
        assert "companies" in data
        assert "total" in data
        assert data["total"] > 0
        assert "Apple Inc." in data["companies"]

    def test_process_query(self, client, mock_app):
        """Test processing a new query."""
        response = client.post(
            "/query",
            json={"query": "Tell me about Apple", "use_cache": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert data["final_response"] is not None
        assert data["interrupted"] == False

    def test_process_query_validation_error(self, client):
        """Test query validation error."""
        response = client.post(
            "/query",
            json={"query": "", "use_cache": True}
        )
        assert response.status_code == 422  # Validation error

    def test_continue_conversation(self, client, mock_app):
        """Test continuing a conversation."""
        response = client.post(
            "/continue",
            json={
                "thread_id": "test-thread-123",
                "query": "What about their competitors?"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-thread-123"

    def test_get_conversation_state(self, client, mock_app):
        """Test getting conversation state."""
        response = client.get("/conversation/test-thread-123")
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-thread-123"
        assert "state" in data
        assert data["state"]["detected_company"] == "Apple Inc."

    def test_get_nonexistent_conversation(self, client, mock_app):
        """Test getting non-existent conversation."""
        mock_app.get_conversation_state.return_value = None
        response = client.get("/conversation/nonexistent")
        assert response.status_code == 404

    def test_cache_stats(self, client):
        """Test getting cache statistics."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "max_size" in data

    def test_clear_cache(self, client):
        """Test clearing the cache."""
        with patch("src.research_assistant.api.query_cache") as mock_cache:
            response = client.post("/cache/clear")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            mock_cache.clear.assert_called_once()


class TestAPIQueryCaching:
    """Tests for API query caching behavior."""

    def test_cached_response(self, mock_app):
        """Test that cached responses are returned."""
        cached_result = {
            "thread_id": "cached-thread",
            "final_response": "Cached response about Apple",
        }

        with patch("src.research_assistant.api.get_app", return_value=mock_app):
            with patch("src.research_assistant.api.query_cache") as mock_cache:
                mock_cache.get.return_value = cached_result
                mock_cache.get_stats.return_value = {"enabled": True}

                from src.research_assistant.api import app
                client = TestClient(app)

                response = client.post(
                    "/query",
                    json={"query": "Tell me about Apple", "use_cache": True}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["cached"] == True
                assert data["final_response"] == "Cached response about Apple"

    def test_skip_cache_when_disabled(self, mock_app):
        """Test that cache is skipped when use_cache is False."""
        with patch("src.research_assistant.api.get_app", return_value=mock_app):
            with patch("src.research_assistant.api.query_cache") as mock_cache:
                mock_cache.get.return_value = {"cached": "data"}
                mock_cache.get_stats.return_value = {"enabled": True}

                from src.research_assistant.api import app
                client = TestClient(app)

                response = client.post(
                    "/query",
                    json={"query": "Tell me about Apple", "use_cache": False}
                )

                assert response.status_code == 200
                # When use_cache=False, we should call start_conversation
                mock_app.start_conversation.assert_called_once()
