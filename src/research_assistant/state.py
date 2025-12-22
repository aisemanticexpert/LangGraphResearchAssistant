"""
State Schema for LangGraph Research Assistant
==============================================

This module defines comprehensive Pydantic models for the multi-agent research workflow.
The schema integrates concepts from the RAGHEAT research paper's weighted factor taxonomy
for stock research, providing production-ready type safety and validation.

Key Design Principles:
    1. Pydantic models for type safety and validation
    2. RAGHEAT-inspired confidence scoring with normalized weights (sum to 1.0)
    3. Comprehensive state tracking for all 5 agents
    4. Human-in-the-loop interrupt support
    5. Audit trail for compliance and debugging

RAGHEAT Confidence Formula:
    confidence = Σ(wi × fi) where Σwi = 1.0

    Factors:
    - data_completeness: 30% - Presence of key data fields
    - source_diversity: 20% - Number of independent sources
    - news_coverage: 15% - News quantity and sentiment quality
    - financial_data: 15% - Financial metrics completeness
    - recency: 10% - Time decay (exponential)
    - sentiment_consistency: 10% - Alignment of sentiment signals

Usage:
    from state import ResearchAssistantState, Message, ResearchFindings
    state = ResearchAssistantState(user_query="Tell me about Apple")

Developer: Rajesh Gupta
Copyright (c) 2024 Rajesh Gupta. All rights reserved.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# ENUMS - Clear status definitions for workflow routing
# ============================================================================

class ClarityStatus(str, Enum):
    """
    Status from Clarity Agent analysis.

    Routing Logic:
        - CLEAR -> Research Agent
        - NEEDS_CLARIFICATION -> Human-in-the-Loop interrupt
        - PENDING -> Initial state before analysis
    """
    CLEAR = "clear"
    NEEDS_CLARIFICATION = "needs_clarification"
    PENDING = "pending"


class ValidationResult(str, Enum):
    """
    Result from Validator Agent assessment.

    Routing Logic:
        - SUFFICIENT -> Synthesis Agent
        - INSUFFICIENT + attempts < 3 -> Research Agent (retry)
        - INSUFFICIENT + attempts >= 3 -> Synthesis Agent (best effort)
        - PENDING -> Initial state before validation
    """
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    PENDING = "pending"


class MarketRegime(str, Enum):
    """
    Market regime classification (inspired by RAGHEAT HMM regime detection).

    Used for context-aware response generation:
        - BULL: Generally positive momentum, risk-on sentiment
        - BEAR: Negative momentum, risk-off sentiment
        - SIDEWAYS: Range-bound, neutral sentiment
        - UNKNOWN: Insufficient data for classification
    """
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class QueryIntent(str, Enum):
    """
    Classified intent of user query for focused research.

    Helps Research Agent prioritize data gathering.
    """
    COMPANY_OVERVIEW = "company_overview"
    STOCK_PRICE = "stock_price"
    FINANCIAL_ANALYSIS = "financial_analysis"
    NEWS_DEVELOPMENTS = "news_developments"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    INVESTMENT_RESEARCH = "investment_research"
    FOLLOW_UP = "follow_up"
    GENERAL = "general"


# ============================================================================
# MESSAGE HANDLING - Conversation history with reducer pattern
# ============================================================================

def add_messages(left: List["Message"], right: List["Message"]) -> List["Message"]:
    """
    LangGraph reducer for message accumulation.

    Appends new messages to existing conversation history,
    maintaining chronological order for context awareness.

    Args:
        left: Existing messages in conversation
        right: New messages to append

    Returns:
        Combined message list
    """
    return left + right


class Message(BaseModel):
    """
    Single message in conversation history.

    Tracks role, content, timing, and metadata for:
        - Multi-turn conversation context
        - Agent attribution
        - Debugging and audit trails

    Attributes:
        role: Who sent the message (user/assistant/system)
        content: The actual message text
        timestamp: When message was created (ISO format)
        agent: Which agent generated this message (if assistant)
        metadata: Additional context (confidence scores, company info, etc.)
    """
    role: Literal["user", "assistant", "system"] = Field(
        description="Message sender role"
    )
    content: str = Field(
        description="Message content text"
    )
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO format timestamp of message creation"
    )
    agent: Optional[str] = Field(
        default=None,
        description="Agent name if role is assistant"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context like confidence scores, company info"
    )


# ============================================================================
# RAGHEAT-INSPIRED FACTOR SCORING
# ============================================================================

class FactorScore(BaseModel):
    """
    Individual factor score following RAGHEAT weighted taxonomy.

    The RAGHEAT paper defines 10 factor categories with baseline weights.
    This model tracks individual factor contributions to overall confidence.

    Attributes:
        name: Human-readable factor name
        weight: Factor weight in weighted sum (0-1, all weights sum to 1.0)
        score: Raw factor score (0-1 scale)
        weighted_score: weight * score contribution
        description: Explanation of what this factor measures
    """
    name: str = Field(description="Factor category name")
    weight: float = Field(ge=0.0, le=1.0, description="Weight in confidence calculation")
    score: float = Field(ge=0.0, le=1.0, description="Raw factor score")
    weighted_score: float = Field(default=0.0, description="Contribution to total confidence")
    description: Optional[str] = Field(default=None, description="What this factor measures")

    @model_validator(mode='after')
    def compute_weighted_score(self) -> 'FactorScore':
        """Auto-compute weighted score from weight and score."""
        object.__setattr__(self, 'weighted_score', self.weight * self.score)
        return self


class ConfidenceBreakdown(BaseModel):
    """
    Detailed breakdown of confidence score calculation.

    Implements RAGHEAT formula: confidence = Σ(wi × fi) where Σwi = 1.0

    Default factor weights based on RAGHEAT Table II (adapted for research context):
        - data_completeness: 30% (company-specific data quality)
        - source_diversity: 20% (multiple independent sources)
        - news_coverage: 15% (news quantity and sentiment quality)
        - financial_data: 15% (financial metrics completeness)
        - recency: 10% (time decay factor)
        - sentiment_consistency: 10% (alignment of sentiment signals)

    Attributes:
        factors: Individual factor scores with weights
        total_score: Final confidence on 0-10 scale
        gaps: List of identified data gaps
        strengths: List of data strengths
    """
    factors: Dict[str, FactorScore] = Field(
        default_factory=dict,
        description="Individual factor scores"
    )
    total_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Final confidence score (0-10 scale)"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified data gaps that lower confidence"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Data strengths that boost confidence"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/display."""
        return {
            "total_score": self.total_score,
            "components": {
                name: factor.score * 10  # Scale to 0-10 for display
                for name, factor in self.factors.items()
            },
            "gaps": self.gaps,
            "strengths": self.strengths
        }


# ============================================================================
# RESEARCH FINDINGS - Structured data from Research Agent
# ============================================================================

class StockInfo(BaseModel):
    """
    Stock price and trading information.

    Captures real-time and historical price data for stock analysis.
    """
    price: Optional[float] = Field(default=None, description="Current stock price")
    change: Optional[float] = Field(default=None, description="Price change (absolute)")
    change_percent: Optional[float] = Field(default=None, description="Price change (percentage)")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    avg_volume: Optional[int] = Field(default=None, description="Average trading volume")
    high_52w: Optional[float] = Field(default=None, description="52-week high")
    low_52w: Optional[float] = Field(default=None, description="52-week low")
    market_cap: Optional[str] = Field(default=None, description="Market capitalization")

    def to_display_string(self) -> str:
        """Format stock info for display."""
        parts = []
        if self.price is not None:
            parts.append(f"Price: ${self.price:.2f}")
        if self.change_percent is not None:
            sign = "+" if self.change_percent >= 0 else ""
            parts.append(f"Change: {sign}{self.change_percent:.2f}%")
        if self.volume is not None:
            parts.append(f"Volume: {self.volume:,}")
        if self.high_52w is not None and self.low_52w is not None:
            parts.append(f"52W Range: ${self.low_52w:.2f} - ${self.high_52w:.2f}")
        return " | ".join(parts) if parts else "No stock data"


class FinancialData(BaseModel):
    """
    Company financial metrics for fundamental analysis.

    Key metrics used in financial health assessment.
    """
    revenue: Optional[str] = Field(default=None, description="Total revenue")
    net_income: Optional[str] = Field(default=None, description="Net income")
    eps: Optional[float] = Field(default=None, description="Earnings per share")
    pe_ratio: Optional[float] = Field(default=None, description="Price-to-earnings ratio")
    profit_margin: Optional[float] = Field(default=None, description="Profit margin percentage")
    roe: Optional[float] = Field(default=None, description="Return on equity")
    debt_to_equity: Optional[float] = Field(default=None, description="Debt-to-equity ratio")

    def get_completeness_score(self) -> float:
        """Calculate how complete the financial data is (0-1)."""
        fields = ["revenue", "net_income", "eps", "pe_ratio", "profit_margin"]
        filled = sum(1 for f in fields if getattr(self, f) is not None)
        return filled / len(fields)


class NewsItem(BaseModel):
    """
    Single news item with sentiment analysis.

    Includes source attribution and sentiment scoring for news analysis.
    """
    title: str = Field(description="News headline")
    date: Optional[str] = Field(default=None, description="Publication date")
    sentiment: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sentiment score (0=negative, 0.5=neutral, 1=positive)"
    )
    source: Optional[str] = Field(default=None, description="News source")
    url: Optional[str] = Field(default=None, description="Article URL")
    summary: Optional[str] = Field(default=None, description="Brief summary")


class ResearchFindings(BaseModel):
    """
    Comprehensive research findings from Research Agent.

    Structured container for all research data gathered about a company,
    including RAGHEAT-inspired factor analysis for confidence scoring.

    Attributes:
        company_name: Official company name
        ticker: Stock ticker symbol
        sector: Industry sector
        recent_news: List of news items with sentiment
        stock_info: Current stock price and trading data
        financials: Financial metrics
        key_developments: List of significant company developments
        factor_data: RAGHEAT-style factor analysis data
        sources: Data sources used
        market_regime: Detected market regime (bull/bear/sideways)
        research_timestamp: When research was conducted
    """
    company_name: Optional[str] = Field(default=None, description="Company name")
    ticker: Optional[str] = Field(default=None, description="Stock ticker")
    sector: Optional[str] = Field(default=None, description="Industry sector")

    # Core research data
    recent_news: List[NewsItem] = Field(
        default_factory=list,
        description="Recent news with sentiment"
    )
    stock_info: Optional[StockInfo] = Field(
        default=None,
        description="Stock price and trading info"
    )
    financials: Optional[FinancialData] = Field(
        default=None,
        description="Financial metrics"
    )
    key_developments: List[str] = Field(
        default_factory=list,
        description="Key company developments"
    )

    # RAGHEAT-inspired factor data
    factor_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Factor analysis data (microeconomic, sentiment, technical)"
    )

    # Metadata
    sources: List[str] = Field(
        default_factory=list,
        description="Data sources used"
    )
    market_regime: MarketRegime = Field(
        default=MarketRegime.UNKNOWN,
        description="Detected market regime"
    )
    research_timestamp: Optional[str] = Field(
        default=None,
        description="When research was conducted"
    )
    data_freshness_hours: float = Field(
        default=0.0,
        description="Age of data in hours for recency scoring"
    )

    # Legacy compatibility - string format
    raw_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw data from research tool"
    )

    def get_news_summary(self) -> str:
        """Generate formatted news summary."""
        if not self.recent_news:
            return "No recent news available"

        summaries = []
        for item in self.recent_news[:3]:  # Top 3 news items
            sentiment_label = (
                "positive" if item.sentiment > 0.6
                else "negative" if item.sentiment < 0.4
                else "neutral"
            )
            summaries.append(f"- {item.title} ({sentiment_label})")

        return "\n".join(summaries)

    def get_data_completeness(self) -> float:
        """Calculate overall data completeness (0-1)."""
        total = 7
        present = sum([
            bool(self.recent_news),
            self.stock_info is not None,
            self.financials is not None,
            bool(self.key_developments),
            bool(self.sector),
            bool(self.ticker),
            bool(self.factor_data)
        ])
        return present / total


# ============================================================================
# MAIN STATE SCHEMA
# ============================================================================

class ResearchAssistantState(BaseModel):
    """
    Main state schema for the LangGraph Research Assistant workflow.

    This comprehensive state object flows through all 4 agents:
        1. Clarity Agent - Analyzes query and extracts company
        2. Research Agent - Gathers data and computes confidence
        3. Validator Agent - Assesses research quality
        4. Synthesis Agent - Generates final response

    The state supports:
        - Multi-turn conversation with message history
        - Human-in-the-loop interrupts for clarification
        - RAGHEAT-inspired confidence scoring
        - Research retry loop (max 3 attempts)
        - Comprehensive error handling
        - Audit trail for compliance

    Routing Logic:
        - After Clarity: clear -> Research, needs_clarification -> Interrupt
        - After Research: confidence >= 6 -> Synthesis, confidence < 6 -> Validator
        - After Validator: sufficient -> Synthesis, insufficient + attempts < 3 -> Research
    """

    # ========== USER QUERY ==========
    user_query: str = Field(
        default="",
        description="Current user query to process"
    )
    original_query: Optional[str] = Field(
        default=None,
        description="Original query preserved for context in follow-ups"
    )

    # ========== CONVERSATION HISTORY ==========
    messages: Annotated[List[Message], add_messages] = Field(
        default_factory=list,
        description="Full conversation history with reducer"
    )

    # ========== CLARITY AGENT STATE ==========
    clarity_status: Literal["clear", "needs_clarification", "pending"] = Field(
        default="pending",
        description="Query clarity status for routing"
    )
    clarification_request: Optional[str] = Field(
        default=None,
        description="Question to ask user if clarification needed"
    )
    detected_company: Optional[str] = Field(
        default=None,
        description="Extracted company name"
    )
    detected_ticker: Optional[str] = Field(
        default=None,
        description="Extracted stock ticker"
    )
    query_intent: Optional[str] = Field(
        default=None,
        description="Classified query intent"
    )

    # ========== RESEARCH AGENT STATE ==========
    research_findings: Optional[ResearchFindings] = Field(
        default=None,
        description="Structured research findings"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Research confidence score (0-10)"
    )
    confidence_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed confidence score breakdown"
    )
    factor_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual RAGHEAT factor scores"
    )

    # ========== VALIDATOR AGENT STATE ==========
    validation_result: Literal["sufficient", "insufficient", "pending"] = Field(
        default="pending",
        description="Validation result for routing"
    )
    validation_feedback: Optional[str] = Field(
        default=None,
        description="Specific feedback for research improvement"
    )
    data_completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Data completeness assessment"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Query relevance assessment"
    )

    # ========== RETRY TRACKING ==========
    research_attempts: int = Field(
        default=0,
        ge=0,
        description="Number of research attempts (max 3)"
    )
    retry_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of retry attempts for analysis"
    )

    # ========== SYNTHESIS AGENT STATE ==========
    final_response: Optional[str] = Field(
        default=None,
        description="Final synthesized response for user"
    )
    executive_summary: Optional[str] = Field(
        default=None,
        description="Brief executive summary"
    )
    detailed_analysis: Optional[str] = Field(
        default=None,
        description="Detailed analysis section"
    )

    # ========== WORKFLOW METADATA ==========
    current_agent: Optional[str] = Field(
        default=None,
        description="Currently executing agent"
    )
    workflow_status: Literal["in_progress", "completed", "interrupted", "error"] = Field(
        default="in_progress",
        description="Overall workflow status"
    )

    # ========== ERROR HANDLING ==========
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )
    error_traceback: Optional[str] = Field(
        default=None,
        description="Full error traceback for debugging"
    )
    has_error: bool = Field(
        default=False,
        description="Flag indicating error occurred"
    )
    error_node: Optional[str] = Field(
        default=None,
        description="Node where error occurred"
    )
    error_recoverable: bool = Field(
        default=False,
        description="Whether error is recoverable"
    )

    # ========== HUMAN-IN-THE-LOOP ==========
    awaiting_human_input: bool = Field(
        default=False,
        description="Whether waiting for human clarification"
    )
    human_response: Optional[str] = Field(
        default=None,
        description="Human's clarification response"
    )

    # ========== SESSION TRACKING ==========
    session_id: Optional[str] = Field(
        default=None,
        description="Unique session identifier"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for personalization"
    )
    request_timestamp: Optional[str] = Field(
        default=None,
        description="When request was received"
    )

    # ========== PERFORMANCE METRICS ==========
    agent_timestamps: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution timestamps by agent"
    )
    total_processing_time_ms: float = Field(
        default=0.0,
        description="Total processing time in milliseconds"
    )

    # ========== AUDIT TRAIL ==========
    audit_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compliance audit trail"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "allow"  # LangGraph may add additional fields
        use_enum_values = True


# ============================================================================
# RAGHEAT CONFIDENCE CALCULATION
# ============================================================================

# Default factor weights based on RAGHEAT Table II (adapted for research)
DEFAULT_FACTOR_WEIGHTS = {
    "data_completeness": 0.30,      # Company data quality (30%)
    "source_diversity": 0.20,       # Multiple sources (20%)
    "news_coverage": 0.15,          # News quality (15%)
    "financial_data": 0.15,         # Financial metrics (15%)
    "recency": 0.10,                # Data freshness (10%)
    "sentiment_consistency": 0.10,  # Sentiment alignment (10%)
}


def calculate_ragheat_confidence(
    findings: ResearchFindings,
    query: str = "",
    weights: Optional[Dict[str, float]] = None
) -> tuple[float, ConfidenceBreakdown]:
    """
    Calculate confidence score using RAGHEAT-inspired methodology.

    Implements the RAGHEAT formula: confidence = Σ(wi × fi) where Σwi = 1.0

    This approach provides:
        - Explainable confidence scores
        - Consistent, reproducible scoring
        - Identification of data gaps
        - Factor-level granularity

    Args:
        findings: Research findings to evaluate
        query: Original user query for relevance scoring
        weights: Optional custom weights (must sum to 1.0)

    Returns:
        Tuple of (confidence_score, confidence_breakdown)
        - confidence_score: Float 0-10
        - confidence_breakdown: Detailed breakdown with factor scores
    """
    weights = weights or DEFAULT_FACTOR_WEIGHTS

    # Validate weights sum to 1.0 (RAGHEAT constraint)
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        # Normalize weights
        weights = {k: v / weight_sum for k, v in weights.items()}

    factors = {}
    gaps = []
    strengths = []

    # 1. Data Completeness (30%)
    completeness = findings.get_data_completeness()
    factors["data_completeness"] = FactorScore(
        name="Data Completeness",
        weight=weights["data_completeness"],
        score=completeness,
        description="Presence of key data fields (news, financials, stock info)"
    )
    if completeness < 0.5:
        gaps.append("Missing key data fields")
    elif completeness > 0.8:
        strengths.append("Comprehensive data coverage")

    # 2. Source Diversity (20%)
    source_count = len(findings.sources) if findings.sources else 1
    source_diversity = min(1.0, source_count / 5)  # Max at 5 sources
    factors["source_diversity"] = FactorScore(
        name="Source Diversity",
        weight=weights["source_diversity"],
        score=source_diversity,
        description="Number of independent data sources"
    )
    if source_count < 2:
        gaps.append("Limited data sources")

    # 3. News Coverage (15%)
    news_score = 0.0
    if findings.recent_news:
        news_count = len(findings.recent_news)
        news_score = min(1.0, news_count / 5)

        # Boost for sentiment quality
        avg_sentiment = sum(n.sentiment for n in findings.recent_news) / news_count
        sentiment_quality = abs(avg_sentiment - 0.5)  # Deviation from neutral
        news_score = (news_score + sentiment_quality) / 2

        strengths.append(f"{news_count} news items with sentiment analysis")
    else:
        gaps.append("No recent news data")

    factors["news_coverage"] = FactorScore(
        name="News Coverage",
        weight=weights["news_coverage"],
        score=news_score,
        description="News quantity and sentiment quality"
    )

    # 4. Financial Data (15%)
    financial_score = 0.0
    if findings.financials:
        financial_score = findings.financials.get_completeness_score()
        if financial_score > 0.7:
            strengths.append("Strong financial data coverage")
    else:
        gaps.append("Missing financial metrics")

    factors["financial_data"] = FactorScore(
        name="Financial Data",
        weight=weights["financial_data"],
        score=financial_score,
        description="Financial metrics completeness"
    )

    # 5. Recency (10%)
    # Time decay following RAGHEAT exponential decay: γ ≈ 0.1 per day
    hours_old = findings.data_freshness_hours
    decay_rate = 0.1 / 24  # Per hour
    recency_score = max(0.0, 1.0 - (decay_rate * hours_old))

    factors["recency"] = FactorScore(
        name="Data Recency",
        weight=weights["recency"],
        score=recency_score,
        description="How fresh the data is"
    )
    if hours_old > 72:
        gaps.append("Data may be stale (>72 hours old)")

    # 6. Sentiment Consistency (10%)
    sentiment_consistency = 1.0
    if findings.factor_data and "sentiment" in findings.factor_data:
        sent_data = findings.factor_data["sentiment"]
        if isinstance(sent_data, dict):
            news_sent = sent_data.get("news_sentiment", 0.5)
            social_sent = sent_data.get("social_sentiment", 0.5)
            divergence = abs(news_sent - social_sent)
            sentiment_consistency = 1.0 - divergence

            if divergence > 0.3:
                gaps.append("Sentiment signals are inconsistent")

    factors["sentiment_consistency"] = FactorScore(
        name="Sentiment Consistency",
        weight=weights["sentiment_consistency"],
        score=sentiment_consistency,
        description="Alignment of sentiment signals"
    )

    # Calculate weighted sum (RAGHEAT equation)
    total = sum(f.weighted_score for f in factors.values())
    confidence_score = round(total * 10, 2)  # Scale to 0-10

    breakdown = ConfidenceBreakdown(
        factors=factors,
        total_score=confidence_score,
        gaps=gaps,
        strengths=strengths
    )

    return confidence_score, breakdown


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_initial_state(
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ResearchAssistantState:
    """
    Factory function to create properly initialized state.

    Use this to ensure all required fields have sensible defaults.

    Args:
        query: User's initial query
        session_id: Optional session identifier
        user_id: Optional user identifier

    Returns:
        Initialized ResearchAssistantState
    """
    import uuid

    now = datetime.now()

    return ResearchAssistantState(
        user_query=query,
        original_query=query,
        session_id=session_id or str(uuid.uuid4()),
        user_id=user_id,
        request_timestamp=now.isoformat(),
        messages=[
            Message(
                role="user",
                content=query,
                timestamp=now.isoformat(),
                metadata={"is_initial_query": True}
            )
        ]
    )


# Type alias for external use
StateDict = Dict[str, Any]
