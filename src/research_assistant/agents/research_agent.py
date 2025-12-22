"""
Research Agent for the Research Assistant
==========================================

The Research Agent is responsible for:
    1. Gathering comprehensive company information based on query intent
    2. Computing RAGHEAT-inspired confidence scores
    3. Building structured research findings
    4. Routing based on confidence threshold

The query_intent from Clarity Agent determines what data to prioritize:
    - "leadership": Focus on CEO, executives, management
    - "stock_price": Focus on stock data, trading info
    - "financial_analysis": Focus on financial metrics
    - "news_developments": Focus on recent news
    - "company_overview": Provide comprehensive overview

Confidence Score Calculation (RAGHEAT-Inspired):
    Uses weighted factor aggregation: confidence = Σ(wi × fi)
    where Σwi = 1.0 (guaranteed normalization)

Routing Logic:
    - confidence_score >= 6.0 -> Synthesis Agent
    - confidence_score < 6.0 -> Validator Agent

Author: Rajesh Gupta
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent
from ..state import (
    Message,
    ResearchFindings,
    NewsItem,
    StockInfo,
    FinancialData,
    MarketRegime,
    ConfidenceBreakdown,
    FactorScore,
    calculate_ragheat_confidence,
    DEFAULT_FACTOR_WEIGHTS
)
from ..tools.research_tool import ResearchTool
from ..guardrails import CompanyNameValidator


class ResearchAgent(BaseAgent):
    """
    Conducts comprehensive company research with RAGHEAT-inspired scoring.

    This agent is the core data gathering component that:
        - Searches for company information based on query intent
        - Prioritizes data relevant to the user's specific question
        - Computes multi-factor confidence scores
        - Detects market regime (bull/bear/sideways)
        - Structures findings for downstream processing

    Intent-Aware Research:
        - "leadership": Includes CEO, founder, executives info
        - "stock_price": Focuses on stock and trading data
        - "financial_analysis": Focuses on financial metrics
        - "news_developments": Focuses on recent news
        - "company_overview": Comprehensive company information

    Features:
        - Mock data provider for development/testing
        - Tavily Search API integration (when configured)
        - RAGHEAT 6-factor confidence scoring
        - Market regime detection
        - Retry-aware research with feedback

    Outputs:
        - research_findings: Structured ResearchFindings object
        - confidence_score: 0-10 score using RAGHEAT methodology
        - confidence_breakdown: Detailed factor-level scoring
        - factor_scores: Individual factor contributions
    """

    # RAGHEAT-inspired factor weights
    FACTOR_WEIGHTS = DEFAULT_FACTOR_WEIGHTS

    # Confidence threshold for routing
    CONFIDENCE_THRESHOLD = 6.0

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        confidence_threshold: float = 6.0
    ):
        """
        Initialize the Research Agent.

        Args:
            model_name: LLM model to use
            temperature: LLM temperature
            confidence_threshold: Threshold for routing (default 6.0)
        """
        super().__init__(model_name=model_name, temperature=temperature)
        self.research_tool = ResearchTool()
        self.confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "ResearchAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Agent specializing in gathering and analyzing company information.

Your task is to:
1. Analyze the provided research data about a company
2. Focus on the specific intent (leadership, stock price, news, etc.)
3. Assess the quality, completeness, and reliability of findings
4. Identify any gaps that may affect research confidence

Consider these factors when analyzing:
- Data completeness: Are all key data fields present for the intent?
- Source diversity: How many independent sources?
- Relevance: Does the data address the specific question?
- Data recency: How fresh is the information?

If this is a retry after validation feedback, focus on addressing the identified gaps.

Respond with analysis of the research quality and any concerns."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research for the specified company.

        This is the main entry point called by LangGraph.

        Steps:
            1. Get company data from research tool
            2. Build structured research findings based on intent
            3. Calculate RAGHEAT confidence score
            4. Detect market regime
            5. Prepare state updates

        The query_intent from Clarity Agent determines what data to prioritize:
            - "leadership": Focus on CEO, executives, management
            - "stock_price": Focus on stock data, trading info
            - "financial_analysis": Focus on financials
            - etc.

        Args:
            state: Current workflow state

        Returns:
            State updates with research findings
        """
        start_time = datetime.now()

        company = state.get("detected_company", "Unknown Company")
        query = state.get("user_query", "")
        query_intent = state.get("query_intent", "company_overview")
        current_attempts = state.get("research_attempts", 0)
        validation_feedback = state.get("validation_feedback")

        attempt = current_attempts + 1

        self._log_execution("Starting research", {
            "company": company,
            "intent": query_intent,
            "attempt": f"{attempt}/3"
        })

        # Get research data from tool
        raw_data = self.research_tool.search(
            company_name=company,
            query=query,
            validation_feedback=validation_feedback
        )

        # Build structured findings with intent awareness
        findings = self._build_research_findings(company, raw_data, query, query_intent)

        # Calculate RAGHEAT confidence score
        confidence_score, confidence_breakdown = calculate_ragheat_confidence(
            findings, query
        )

        # Boost confidence if we have the specific data for the intent
        confidence_score = self._adjust_confidence_for_intent(
            confidence_score, findings, query_intent, raw_data
        )

        # Convert breakdown to serializable dict
        breakdown_dict = confidence_breakdown.to_dict()

        # Extract factor scores for state
        factor_scores = {
            name: factor.score
            for name, factor in confidence_breakdown.factors.items()
        }

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        self._log_execution("Research completed", {
            "company": company,
            "intent": query_intent,
            "confidence": round(confidence_score, 2),
            "attempt": attempt,
            "processing_time_ms": round(processing_time, 2)
        })

        # Build response message
        response_summary = self._build_research_summary(
            findings, confidence_score, breakdown_dict, attempt, query_intent
        )

        return {
            "research_findings": findings,
            "confidence_score": confidence_score,
            "confidence_breakdown": breakdown_dict,
            "factor_scores": factor_scores,
            "research_attempts": attempt,
            "current_agent": self.name,
            "validation_result": "pending",  # Reset for validator
            "messages": [Message(
                role="assistant",
                content=response_summary,
                agent=self.name,
                metadata={
                    "company": company,
                    "ticker": findings.ticker,
                    "confidence_score": confidence_score,
                    "query_intent": query_intent,
                    "attempt": attempt,
                    "processing_time_ms": processing_time
                }
            )]
        }

    def _build_research_findings(
        self,
        company: str,
        raw_data: Dict[str, Any],
        query: str,
        intent: str
    ) -> ResearchFindings:
        """
        Build structured ResearchFindings from raw data.

        Converts raw research data into Pydantic models for type safety
        and consistent structure. Includes leadership info for relevant intents.

        Args:
            company: Company name
            raw_data: Raw data from research tool
            query: User's original query
            intent: Query intent for data prioritization

        Returns:
            Structured ResearchFindings object
        """
        # Parse news items
        news_items = self._parse_news_items(raw_data.get("recent_news", ""))

        # Parse stock info
        stock_info = self._parse_stock_info(raw_data.get("stock_info", ""))

        # Parse financial data
        financials = self._parse_financial_data(raw_data.get("additional_info", {}))

        # Parse key developments
        developments = self._parse_developments(raw_data.get("key_developments", ""))

        # Build factor data for sentiment analysis
        factor_data = self._build_factor_data(news_items, raw_data)

        # Add leadership info to factor_data for leadership intent
        leadership_info = self._extract_leadership_info(raw_data)
        if leadership_info:
            factor_data["leadership"] = leadership_info

        # Determine market regime
        market_regime = self._detect_market_regime(stock_info, factor_data)

        # Determine sources
        sources = ["mock_data"] if "mock" in str(raw_data.get("source", "")).lower() else ["tavily_api"]

        # Get sector from additional info
        additional_info = raw_data.get("additional_info", {})
        sector = additional_info.get("industry")

        return ResearchFindings(
            company_name=company,
            ticker=self._extract_ticker(company),
            sector=sector,
            recent_news=news_items,
            stock_info=stock_info,
            financials=financials,
            key_developments=developments,
            factor_data=factor_data,
            sources=sources,
            market_regime=market_regime,
            research_timestamp=datetime.now().isoformat(),
            data_freshness_hours=0.0,  # Mock data is "fresh"
            raw_data=raw_data
        )

    def _extract_leadership_info(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract leadership information (CEO, founder, etc.) from raw data.

        Args:
            raw_data: Raw research data

        Returns:
            Leadership info dict or None
        """
        additional_info = raw_data.get("additional_info", {})
        leadership = {}

        # Check for CEO
        if additional_info.get("ceo"):
            leadership["ceo"] = additional_info["ceo"]

        # Check for founder
        if additional_info.get("founder"):
            leadership["founder"] = additional_info["founder"]

        # Check for founded year
        if additional_info.get("founded"):
            leadership["founded"] = additional_info["founded"]

        # Check for headquarters
        if additional_info.get("headquarters"):
            leadership["headquarters"] = additional_info["headquarters"]

        # Check for employees
        if additional_info.get("employees"):
            leadership["employees"] = additional_info["employees"]

        return leadership if leadership else None

    def _adjust_confidence_for_intent(
        self,
        base_confidence: float,
        findings: ResearchFindings,
        intent: str,
        raw_data: Dict[str, Any]
    ) -> float:
        """
        Adjust confidence score based on whether we have data for the specific intent.

        If the user asks about CEO and we have CEO info, boost confidence.
        If user asks about stock and we have stock info, boost confidence.

        Args:
            base_confidence: Base RAGHEAT confidence score
            findings: Research findings
            intent: Query intent
            raw_data: Raw data from research tool

        Returns:
            Adjusted confidence score (0-10)
        """
        additional_info = raw_data.get("additional_info", {})
        adjustment = 0.0

        if intent == "leadership":
            # Check if we have CEO/leadership info
            if additional_info.get("ceo"):
                adjustment += 2.0  # Significant boost for having CEO info
            if additional_info.get("founder"):
                adjustment += 0.5
            if additional_info.get("founded"):
                adjustment += 0.3

        elif intent == "stock_price":
            # Check if we have stock info
            if findings.stock_info and findings.stock_info.price:
                adjustment += 2.0

        elif intent == "financial_analysis":
            # Check if we have financial data
            if findings.financials:
                completeness = findings.financials.get_completeness_score()
                adjustment += completeness * 2.0

        elif intent == "news_developments":
            # Check news coverage
            if findings.recent_news and len(findings.recent_news) >= 2:
                adjustment += 1.5

        # Apply adjustment but cap at 10
        return min(10.0, base_confidence + adjustment)

    def _parse_news_items(self, news_data: Any) -> List[NewsItem]:
        """
        Parse news data into structured NewsItem list.

        Handles both string format (from Tavily) and dict format (from mock).

        Args:
            news_data: Raw news data

        Returns:
            List of NewsItem objects
        """
        if not news_data:
            return []

        news_items = []

        # Handle string format (news summary text)
        if isinstance(news_data, str):
            # Split by common delimiters and create items
            parts = news_data.split(". ")
            for i, part in enumerate(parts[:5]):  # Max 5 items
                if part.strip():
                    news_items.append(NewsItem(
                        title=part.strip()[:200],
                        sentiment=0.5,  # Neutral default
                        source="research"
                    ))

        # Handle list format (structured news)
        elif isinstance(news_data, list):
            for item in news_data[:5]:
                if isinstance(item, dict):
                    news_items.append(NewsItem(
                        title=item.get("title", ""),
                        date=item.get("date"),
                        sentiment=item.get("sentiment", 0.5),
                        source=item.get("source"),
                        url=item.get("url"),
                        summary=item.get("summary")
                    ))
                elif isinstance(item, str):
                    news_items.append(NewsItem(
                        title=item[:200],
                        sentiment=0.5
                    ))

        return news_items

    def _parse_stock_info(self, stock_data: Any) -> Optional[StockInfo]:
        """
        Parse stock data into StockInfo model.

        Args:
            stock_data: Raw stock data

        Returns:
            StockInfo object or None
        """
        if not stock_data:
            return None

        # Handle string format
        if isinstance(stock_data, str):
            # Try to extract numbers from string
            import re
            price_match = re.search(r'\$?([\d,]+\.?\d*)', stock_data)
            if price_match:
                try:
                    price = float(price_match.group(1).replace(',', ''))
                    return StockInfo(price=price)
                except ValueError:
                    pass
            return None

        # Handle dict format
        if isinstance(stock_data, dict):
            return StockInfo(
                price=stock_data.get("price"),
                change=stock_data.get("change"),
                change_percent=stock_data.get("change_percent"),
                volume=stock_data.get("volume"),
                avg_volume=stock_data.get("avg_volume"),
                high_52w=stock_data.get("52w_high") or stock_data.get("high_52w"),
                low_52w=stock_data.get("52w_low") or stock_data.get("low_52w"),
                market_cap=stock_data.get("market_cap")
            )

        return None

    def _parse_financial_data(self, additional_info: Dict) -> Optional[FinancialData]:
        """
        Parse financial data from additional info.

        Args:
            additional_info: Additional info dict

        Returns:
            FinancialData object or None
        """
        if not additional_info:
            return None

        # Check if we have any financial fields
        financial_fields = [
            "revenue", "net_income", "eps", "pe_ratio",
            "profit_margin", "roe", "debt_to_equity"
        ]

        has_financial = any(
            additional_info.get(f) is not None
            for f in financial_fields
        )

        if not has_financial:
            return None

        return FinancialData(
            revenue=additional_info.get("revenue"),
            net_income=additional_info.get("net_income"),
            eps=additional_info.get("eps"),
            pe_ratio=additional_info.get("pe_ratio"),
            profit_margin=additional_info.get("profit_margin"),
            roe=additional_info.get("roe"),
            debt_to_equity=additional_info.get("debt_to_equity")
        )

    def _parse_developments(self, dev_data: Any) -> List[str]:
        """
        Parse key developments into list of strings.

        Args:
            dev_data: Raw developments data

        Returns:
            List of development strings
        """
        if not dev_data:
            return []

        if isinstance(dev_data, list):
            return [str(d) for d in dev_data[:10]]

        if isinstance(dev_data, str):
            # Split by common delimiters
            parts = dev_data.split(". ")
            return [p.strip() for p in parts if p.strip()][:10]

        return []

    def _build_factor_data(
        self,
        news_items: List[NewsItem],
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build RAGHEAT-style factor data for confidence scoring.

        Creates structured factor data for microeconomic, sentiment,
        and technical analysis.

        Args:
            news_items: Parsed news items
            raw_data: Raw research data

        Returns:
            Factor data dictionary
        """
        factor_data = {}

        # Sentiment factors from news
        if news_items:
            avg_sentiment = sum(n.sentiment for n in news_items) / len(news_items)
            factor_data["sentiment"] = {
                "news_sentiment": avg_sentiment,
                "social_sentiment": avg_sentiment,  # Approximation
                "news_count": len(news_items)
            }

        # Add any existing factor data from mock
        additional = raw_data.get("additional_info", {})
        if additional.get("factor_data"):
            factor_data.update(additional["factor_data"])

        return factor_data

    def _detect_market_regime(
        self,
        stock_info: Optional[StockInfo],
        factor_data: Dict[str, Any]
    ) -> MarketRegime:
        """
        Detect market regime using RAGHEAT HMM-inspired approach.

        Simplified version using available indicators:
            - Price change direction
            - Sentiment signals
            - Technical indicators (if available)

        Args:
            stock_info: Stock price information
            factor_data: Factor analysis data

        Returns:
            MarketRegime enum value
        """
        bullish_signals = 0
        bearish_signals = 0

        # Check price change
        if stock_info and stock_info.change_percent is not None:
            if stock_info.change_percent > 1.0:
                bullish_signals += 1
            elif stock_info.change_percent < -1.0:
                bearish_signals += 1

        # Check sentiment
        if "sentiment" in factor_data:
            sentiment = factor_data["sentiment"]
            news_sent = sentiment.get("news_sentiment", 0.5)
            if news_sent > 0.6:
                bullish_signals += 1
            elif news_sent < 0.4:
                bearish_signals += 1

        # Check technical indicators if available
        if "technical" in factor_data:
            tech = factor_data["technical"]
            if tech.get("macd_signal") == "bullish":
                bullish_signals += 1
            elif tech.get("macd_signal") == "bearish":
                bearish_signals += 1

            if tech.get("above_50dma"):
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Determine regime
        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            return MarketRegime.BULL
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            return MarketRegime.BEAR
        elif bullish_signals > 0 or bearish_signals > 0:
            return MarketRegime.SIDEWAYS

        return MarketRegime.UNKNOWN

    def _extract_ticker(self, company: str) -> Optional[str]:
        """
        Extract ticker symbol for company.

        Args:
            company: Company name

        Returns:
            Ticker symbol or None
        """
        _, ticker = CompanyNameValidator.normalize_company_name(company)
        return ticker

    def _build_research_summary(
        self,
        findings: ResearchFindings,
        confidence_score: float,
        breakdown: Dict[str, Any],
        attempt: int,
        intent: str
    ) -> str:
        """
        Build a summary message for research results.

        Args:
            findings: Research findings
            confidence_score: Confidence score
            breakdown: Confidence breakdown
            attempt: Current attempt number
            intent: Query intent

        Returns:
            Summary message string
        """
        parts = [
            f"[Research Agent] Completed research for {findings.company_name}",
        ]

        if findings.ticker:
            parts.append(f"({findings.ticker})")

        parts.append(f"- Intent: {intent}")
        parts.append(f"- Attempt: {attempt}/3")
        parts.append(f"- Confidence: {confidence_score:.1f}/10")

        # Add intent-specific info
        if intent == "leadership" and findings.factor_data.get("leadership"):
            leadership = findings.factor_data["leadership"]
            if leadership.get("ceo"):
                parts.append(f"- CEO: {leadership['ceo']}")

        # Add key metrics
        if findings.recent_news:
            parts.append(f"- News items: {len(findings.recent_news)}")

        if findings.key_developments:
            parts.append(f"- Developments: {len(findings.key_developments)}")

        # Add strengths/gaps from breakdown
        if breakdown.get("strengths"):
            parts.append(f"- Strengths: {', '.join(breakdown['strengths'][:2])}")

        if breakdown.get("gaps"):
            parts.append(f"- Gaps: {', '.join(breakdown['gaps'][:2])}")

        return " | ".join(parts)
