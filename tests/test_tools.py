"""Tests for research tools."""

import pytest
from src.research_assistant.tools.mock_data import (
    MOCK_RESEARCH_DATA,
    COMPANY_ALIASES,
    get_company_data,
    list_available_companies,
)
from src.research_assistant.tools.research_tool import ResearchTool


class TestMockData:
    """Tests for mock data module."""

    def test_mock_data_has_required_companies(self):
        """Should include Apple and Tesla as per SRS."""
        assert "Apple Inc." in MOCK_RESEARCH_DATA
        assert "Tesla" in MOCK_RESEARCH_DATA

    def test_mock_data_has_required_fields(self):
        """Should have required fields for each company."""
        for company, data in MOCK_RESEARCH_DATA.items():
            assert "recent_news" in data, f"Missing recent_news for {company}"
            assert "stock_info" in data, f"Missing stock_info for {company}"
            assert "key_developments" in data, f"Missing key_developments for {company}"

    def test_company_aliases_map_correctly(self):
        """Should map aliases to canonical names."""
        assert COMPANY_ALIASES["apple"] == "Apple Inc."
        assert COMPANY_ALIASES["aapl"] == "Apple Inc."
        assert COMPANY_ALIASES["tesla"] == "Tesla"
        assert COMPANY_ALIASES["tsla"] == "Tesla"

    def test_get_company_data_by_name(self):
        """Should return data for exact company name."""
        data = get_company_data("Apple Inc.")
        assert data["recent_news"] is not None
        assert "Vision Pro" in data["recent_news"] or "AI" in data["recent_news"]

    def test_get_company_data_by_alias(self):
        """Should return data for company alias."""
        data = get_company_data("apple")
        assert data["stock_info"] is not None

    def test_get_company_data_case_insensitive(self):
        """Should handle case-insensitive lookups."""
        data1 = get_company_data("APPLE")
        data2 = get_company_data("Apple")
        data3 = get_company_data("apple")
        # All should return Apple data
        assert data1.get("stock_info") == data2.get("stock_info") == data3.get("stock_info")

    def test_get_company_data_unknown_company(self):
        """Should return default data for unknown companies."""
        data = get_company_data("Unknown Corp XYZ")
        assert "not available" in data["recent_news"].lower() or "no" in data["recent_news"].lower()

    def test_list_available_companies(self):
        """Should list all available companies."""
        companies = list_available_companies()
        assert len(companies) >= 2  # At least Apple and Tesla
        assert "Apple Inc." in companies
        assert "Tesla" in companies


class TestResearchTool:
    """Tests for ResearchTool class."""

    @pytest.fixture
    def tool(self):
        """Create a ResearchTool instance."""
        return ResearchTool()

    def test_search_returns_dict(self, tool):
        """Should return a dictionary of findings."""
        result = tool.search(
            company_name="Apple Inc.",
            query="Tell me about Apple"
        )
        assert isinstance(result, dict)

    def test_search_has_required_fields(self, tool):
        """Should return results with required fields."""
        result = tool.search(
            company_name="Tesla",
            query="Tesla stock info"
        )
        assert "company_name" in result
        assert "recent_news" in result
        assert "stock_info" in result
        assert "key_developments" in result

    def test_search_resolves_aliases(self, tool):
        result = tool.search(
            company_name="Apple Inc.",
            query="Apple news"
        )
        assert "Apple" in result["company_name"]

    def test_search_with_validation_feedback(self, tool):
        """Should accept validation feedback parameter."""
        result = tool.search(
            company_name="Microsoft",
            query="Microsoft cloud",
            validation_feedback="Need more financial details"
        )
        assert result is not None

    def test_get_available_companies(self, tool):
        """Should return list of available companies."""
        companies = tool.get_available_companies()
        assert len(companies) > 0
        assert "Apple Inc." in companies

    def test_has_data_for_known_company(self, tool):
        """Should return True for known companies."""
        assert tool.has_data_for("Apple Inc.") is True
        assert tool.has_data_for("apple") is True
        assert tool.has_data_for("TSLA") is True

    def test_has_data_for_unknown_company(self, tool):
        """Should return False for unknown companies."""
        assert tool.has_data_for("Unknown Corp") is False
        assert tool.has_data_for("Random Company") is False
