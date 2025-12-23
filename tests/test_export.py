"""
Tests for the conversation export functionality.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch


class TestConversationExporter:
    """Tests for ConversationExporter functionality."""

    @pytest.fixture
    def temp_export_dir(self):
        """Create a temporary directory for exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_state(self):
        """Sample conversation state for testing."""
        return {
            "detected_company": "Apple Inc.",
            "research_attempts": 2,
            "confidence_score": 8.5,
            "clarity_status": "clear",
            "validation_result": "sufficient",
            "final_response": "Apple Inc. is a technology company...",
            "research_findings": {
                "company_name": "Apple Inc.",
                "recent_news": "Launched Vision Pro",
                "stock_info": "Trading at $195",
                "key_developments": "AI integration",
            },
        }

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            {"role": "user", "content": "Tell me about Apple", "timestamp": "2024-01-01T10:00:00", "agent": None},
            {"role": "assistant", "content": "Apple Inc. is...", "timestamp": "2024-01-01T10:00:05", "agent": "synthesis"},
        ]

    def test_export_to_json(self, temp_export_dir, sample_state, sample_messages):
        """Test exporting conversation to JSON."""
        with patch("src.research_assistant.utils.export.settings") as mock_settings:
            mock_settings.export_dir = temp_export_dir

            from src.research_assistant.utils.export import ConversationExporter
            exporter = ConversationExporter(export_dir=temp_export_dir)

            filepath = exporter.export_to_json("thread-123", sample_state, sample_messages)

            assert os.path.exists(filepath)
            assert filepath.endswith(".json")

            # Verify JSON content
            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["thread_id"] == "thread-123"
            assert "export_timestamp" in data
            assert data["metadata"]["detected_company"] == "Apple Inc."
            assert len(data["messages"]) == 2
            assert data["final_response"] == "Apple Inc. is a technology company..."

    def test_export_to_markdown(self, temp_export_dir, sample_state, sample_messages):
        """Test exporting conversation to Markdown."""
        with patch("src.research_assistant.utils.export.settings") as mock_settings:
            mock_settings.export_dir = temp_export_dir

            from src.research_assistant.utils.export import ConversationExporter
            exporter = ConversationExporter(export_dir=temp_export_dir)

            filepath = exporter.export_to_markdown("thread-456", sample_state, sample_messages)

            assert os.path.exists(filepath)
            assert filepath.endswith(".md")

            # Verify Markdown content
            with open(filepath, "r") as f:
                content = f.read()

            assert "# Research Assistant Conversation" in content
            assert "thread-456" in content
            assert "Apple Inc." in content
            assert "## Metadata" in content
            assert "## Conversation" in content

    def test_list_exports(self, temp_export_dir, sample_state, sample_messages):
        """Test listing exported conversations."""
        with patch("src.research_assistant.utils.export.settings") as mock_settings:
            mock_settings.export_dir = temp_export_dir

            from src.research_assistant.utils.export import ConversationExporter
            exporter = ConversationExporter(export_dir=temp_export_dir)

            # Create some exports
            exporter.export_to_json("thread-1", sample_state, sample_messages)
            exporter.export_to_markdown("thread-2", sample_state, sample_messages)

            exports = exporter.list_exports()

            assert len(exports) == 2
            assert all("filename" in e for e in exports)
            assert all("path" in e for e in exports)
            assert all("format" in e for e in exports)

    def test_export_with_pydantic_findings(self, temp_export_dir, sample_messages):
        from src.research_assistant.state import ResearchFindings, NewsItem, StockInfo

        findings = ResearchFindings(
            company_name="Tesla",
            recent_news=[NewsItem(title="Cybertruck deliveries")],
            stock_info=StockInfo(price=242.0),
            key_developments=["FSD v12 rollout"],
            sources=["mock_data"]
        )

        state = {
            "detected_company": "Tesla",
            "research_findings": findings,
            "final_response": "Tesla is an EV company...",
            "confidence_score": 7.5,
            "research_attempts": 1,
            "clarity_status": "clear",
            "validation_result": "sufficient",
        }

        with patch("src.research_assistant.utils.export.settings") as mock_settings:
            mock_settings.export_dir = temp_export_dir

            from src.research_assistant.utils.export import ConversationExporter
            exporter = ConversationExporter(export_dir=temp_export_dir)

            filepath = exporter.export_to_json("thread-pydantic", state, sample_messages)

            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["research_findings"]["company_name"] == "Tesla"
            assert len(data["research_findings"]["recent_news"]) > 0

    def test_export_creates_directory(self, temp_export_dir):
        """Test that export creates directory if it doesn't exist."""
        nested_dir = os.path.join(temp_export_dir, "nested", "exports")

        with patch("src.research_assistant.utils.export.settings") as mock_settings:
            mock_settings.export_dir = nested_dir

            from src.research_assistant.utils.export import ConversationExporter
            exporter = ConversationExporter(export_dir=nested_dir)

            assert os.path.exists(nested_dir)

    def test_export_empty_messages(self, temp_export_dir, sample_state):
        """Test exporting conversation with no messages."""
        with patch("src.research_assistant.utils.export.settings") as mock_settings:
            mock_settings.export_dir = temp_export_dir

            from src.research_assistant.utils.export import ConversationExporter
            exporter = ConversationExporter(export_dir=temp_export_dir)

            filepath = exporter.export_to_json("thread-empty", sample_state, [])

            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["messages"] == []
