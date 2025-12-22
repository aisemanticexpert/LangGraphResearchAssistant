"""
Research tool for gathering company information.

Provides a unified interface for company research that can use
Tavily Search API (preferred) or mock data for development/testing.
"""

import logging
from typing import Any, Dict, Optional

from ..config import settings
from .mock_data import get_company_data, COMPANY_ALIASES, MOCK_RESEARCH_DATA


class ResearchTool:
    """
    Research tool for gathering company information.

    Uses Tavily Search API when available, with fallback to mock data.
    Tavily is the preferred search tool as specified in requirements.
    """

    def __init__(self):
        """Initialize the research tool."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tavily_client = None
        self._init_tavily()

    def _init_tavily(self):
        """
        Initialize Tavily client if API key is available.

        Priority:
        1. If Tavily API key is configured -> use Tavily (preferred)
        2. If use_mock_data is explicitly True AND no Tavily key -> use mock data
        3. If no Tavily key available -> fall back to mock data
        """
        # Check if Tavily API key is properly configured
        if settings.validate_tavily_key():
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=settings.tavily_api_key)
                self.logger.info("ResearchTool initialized with Tavily Search API (PREFERRED)")
            except ImportError:
                self.logger.warning(
                    "Tavily package not installed. Install with: pip install tavily-python. "
                    "Falling back to mock data."
                )
                self._tavily_client = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize Tavily client: {e}. Falling back to mock data.")
                self._tavily_client = None
        elif settings.use_mock_data:
            self.logger.info("ResearchTool initialized with mock data (use_mock_data=True, no Tavily key)")
        else:
            self.logger.warning(
                "No Tavily API key configured. Set TAVILY_API_KEY in .env file. "
                "Using mock data as fallback."
            )

    @property
    def is_tavily_enabled(self) -> bool:
        """Check if Tavily is enabled and available."""
        return self._tavily_client is not None

    def search(
        self,
        company_name: str,
        query: str,
        validation_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for company information using Tavily or mock data.

        Args:
            company_name: Name of the company to research
            query: Original user query for context
            validation_feedback: Feedback from validator for retry attempts

        Returns:
            Dictionary containing research findings
        """
        self.logger.info(f"Searching for company: {company_name}")
        self.logger.debug(f"Query: {query}")

        if validation_feedback:
            self.logger.info(f"Retry with feedback: {validation_feedback}")

        # Use Tavily if available, otherwise fall back to mock data
        if self.is_tavily_enabled:
            return self._search_tavily(company_name, query, validation_feedback)
        else:
            return self._search_mock(company_name, query)

    def _search_tavily(
        self,
        company_name: str,
        query: str,
        validation_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search using Tavily Search API.

        Args:
            company_name: Name of the company to research
            query: Original user query for context
            validation_feedback: Feedback from validator for retry attempts

        Returns:
            Dictionary containing research findings from Tavily
        """
        self.logger.info(f"Using Tavily Search API for {company_name}")

        try:
            # Build search queries for different aspects
            searches = {}

            # Search for recent news
            news_query = f"{company_name} latest news developments 2024"
            if validation_feedback and "news" in validation_feedback.lower():
                news_query = f"{company_name} breaking news recent updates"

            news_results = self._tavily_client.search(
                query=news_query,
                search_depth="advanced",
                max_results=3
            )
            searches["news"] = news_results

            # Search for stock/financial info
            stock_query = f"{company_name} stock price financial performance"
            stock_results = self._tavily_client.search(
                query=stock_query,
                search_depth="basic",
                max_results=2
            )
            searches["stock"] = stock_results

            # Search for key developments based on user query
            dev_query = f"{company_name} {query}"
            if validation_feedback:
                dev_query = f"{company_name} {validation_feedback}"

            dev_results = self._tavily_client.search(
                query=dev_query,
                search_depth="advanced",
                max_results=3
            )
            searches["developments"] = dev_results

            # Process and structure the results
            result = self._process_tavily_results(company_name, query, searches)
            return result

        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}. Falling back to mock data.")
            return self._search_mock(company_name, query)

    def _process_tavily_results(
        self,
        company_name: str,
        query: str,
        searches: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process Tavily search results into structured format.

        Args:
            company_name: Name of the company
            query: Original user query
            searches: Dictionary of Tavily search results

        Returns:
            Structured research findings
        """
        # Extract news content
        news_content = []
        news_sources = []
        if "news" in searches and searches["news"].get("results"):
            for r in searches["news"]["results"][:3]:
                news_content.append(r.get("content", "")[:200])
                news_sources.append(r.get("url", ""))

        # Extract stock content
        stock_content = []
        if "stock" in searches and searches["stock"].get("results"):
            for r in searches["stock"]["results"][:2]:
                stock_content.append(r.get("content", "")[:200])

        # Extract development content
        dev_content = []
        if "developments" in searches and searches["developments"].get("results"):
            for r in searches["developments"]["results"][:3]:
                dev_content.append(r.get("content", "")[:200])

        result = {
            "company_name": company_name,
            "recent_news": " | ".join(news_content) if news_content else "No recent news found",
            "stock_info": " | ".join(stock_content) if stock_content else "No stock info found",
            "key_developments": " | ".join(dev_content) if dev_content else "No developments found",
            "additional_info": {
                "sources": news_sources,
                "search_engine": "tavily",
            },
            "source": "tavily_search",
            "query_context": query,
        }

        self.logger.info(f"Tavily search completed for {company_name}")
        return result

    def _search_mock(self, company_name: str, query: str) -> Dict[str, Any]:
        """
        Search using mock data.

        Args:
            company_name: Name of the company to research
            query: Original user query for context

        Returns:
            Dictionary containing mock research findings
        """
        self.logger.info(f"Using mock data for {company_name}")

        # Get data from mock database
        data = get_company_data(company_name)

        # Resolve canonical company name
        canonical_name = self._resolve_company_name(company_name)

        result = {
            "company_name": canonical_name or company_name,
            "recent_news": data.get("recent_news", "No news available"),
            "stock_info": data.get("stock_info", "No stock info available"),
            "key_developments": data.get("key_developments", "No developments found"),
            "additional_info": {
                "competitors": data.get("competitors", []),
                "industry": data.get("industry", "Unknown"),
                "ceo": data.get("ceo", "Unknown"),
            },
            "source": "mock_data",
            "query_context": query,
        }

        self.logger.info(f"Found mock data for {canonical_name or company_name}")
        return result

    def _resolve_company_name(self, name: str) -> Optional[str]:
        """Resolve a company name or alias to its canonical form."""
        normalized = name.lower().strip()

        # Remove common suffixes for matching
        normalized_clean = normalized.replace(" inc.", "").replace(" inc", "").replace(" corp.", "").replace(" corp", "")

        # Check aliases first
        if normalized in COMPANY_ALIASES:
            return COMPANY_ALIASES[normalized]

        # Try without suffixes
        if normalized_clean in COMPANY_ALIASES:
            return COMPANY_ALIASES[normalized_clean]

        # Check direct matches
        for key in MOCK_RESEARCH_DATA:
            if key.lower() == normalized:
                return key
            # Also check without common suffixes
            key_clean = key.lower().replace(" inc.", "").replace(" inc", "")
            if key_clean == normalized_clean:
                return key

        return None

    def get_available_companies(self) -> list:
        """Return list of companies with available mock data."""
        return list(MOCK_RESEARCH_DATA.keys())

    def has_data_for(self, company_name: str) -> bool:
        """Check if mock data exists for a company."""
        normalized = company_name.lower().strip()

        if normalized in COMPANY_ALIASES:
            return True

        for key in MOCK_RESEARCH_DATA:
            if key.lower() == normalized:
                return True

        return False
