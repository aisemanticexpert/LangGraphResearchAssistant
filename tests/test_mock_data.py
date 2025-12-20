"""
Tests for the mock data module.
"""

import pytest
from src.research_assistant.tools.mock_data import (
    MOCK_RESEARCH_DATA,
    COMPANY_ALIASES,
    get_company_data,
    list_available_companies,
)


class TestMockData:
    """Tests for mock data functionality."""

    def test_mock_data_has_minimum_companies(self):
        """Test that mock data has at least 25 companies."""
        companies = list_available_companies()
        assert len(companies) >= 25, f"Expected at least 25 companies, got {len(companies)}"

    def test_mock_data_company_structure(self):
        """Test that each company has required fields."""
        required_fields = ["recent_news", "stock_info", "key_developments"]

        for company, data in MOCK_RESEARCH_DATA.items():
            for field in required_fields:
                assert field in data, f"Company {company} missing field: {field}"

    def test_get_company_data_by_name(self):
        """Test getting company data by exact name."""
        data = get_company_data("Apple Inc.")
        assert data["ceo"] == "Tim Cook"
        assert "recent_news" in data

    def test_get_company_data_by_alias(self):
        """Test getting company data by alias."""
        # Test various aliases
        test_cases = [
            ("apple", "Apple Inc."),
            ("AAPL", "Apple Inc."),
            ("tesla", "Tesla"),
            ("TSLA", "Tesla"),
            ("google", "Google"),
            ("alphabet", "Google"),
            ("facebook", "Meta"),
            ("meta", "Meta"),
            ("nvidia", "NVIDIA"),
            ("nvda", "NVIDIA"),
        ]

        for alias, expected_company in test_cases:
            data = get_company_data(alias)
            # Check that we got valid data (not the default "not found" response)
            assert "note" not in data, f"Alias '{alias}' should find {expected_company}"

    def test_get_company_data_case_insensitive(self):
        """Test that company lookup is case-insensitive."""
        lower = get_company_data("apple")
        upper = get_company_data("APPLE")
        mixed = get_company_data("ApPlE")

        # All should return the same data
        assert lower == upper == mixed
        assert lower["ceo"] == "Tim Cook"

    def test_get_company_data_unknown(self):
        """Test getting data for unknown company."""
        data = get_company_data("Unknown Corp XYZ")

        assert "note" in data
        assert "not in our mock database" in data["note"]
        assert "Unknown Corp XYZ" in data["recent_news"]

    def test_company_aliases_coverage(self):
        """Test that all major companies have aliases."""
        major_companies = [
            "Apple Inc.", "Tesla", "Microsoft", "Google", "Amazon",
            "Meta", "NVIDIA", "Netflix", "AMD", "Intel"
        ]

        for company in major_companies:
            # Find at least one alias for each major company
            found_alias = False
            for alias, target in COMPANY_ALIASES.items():
                if target == company:
                    found_alias = True
                    break
            assert found_alias, f"Company {company} has no aliases"

    def test_list_available_companies(self):
        """Test listing all available companies."""
        companies = list_available_companies()

        assert isinstance(companies, list)
        assert len(companies) > 0
        assert "Apple Inc." in companies
        assert "Tesla" in companies
        assert "Microsoft" in companies

    def test_new_companies_added(self):
        """Test that newly added companies are present."""
        new_companies = [
            "AMD", "Intel", "Salesforce", "Oracle", "Adobe",
            "JPMorgan Chase", "Visa", "PayPal", "Square (Block)",
            "Pfizer", "Johnson & Johnson", "UnitedHealth",
            "Walmart", "Costco", "Nike", "Starbucks",
            "Toyota", "Ford", "Disney", "Spotify"
        ]

        available = list_available_companies()

        for company in new_companies:
            assert company in available, f"Company {company} should be in mock data"

    def test_company_data_fields_extended(self):
        """Test that companies have extended fields like headquarters and employees."""
        data = get_company_data("Apple Inc.")

        assert "headquarters" in data
        assert "employees" in data
        assert "Cupertino" in data["headquarters"]

    def test_ticker_symbol_aliases(self):
        """Test that ticker symbols work as aliases."""
        ticker_tests = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft"),
            ("GOOGL", "Google"),
            ("AMZN", "Amazon"),
            ("NVDA", "NVIDIA"),
            ("NFLX", "Netflix"),
            ("AMD", "AMD"),
            ("INTC", "Intel"),
            ("CRM", "Salesforce"),
            ("JPM", "JPMorgan Chase"),
        ]

        for ticker, expected in ticker_tests:
            data = get_company_data(ticker)
            assert "note" not in data, f"Ticker {ticker} should find {expected}"
