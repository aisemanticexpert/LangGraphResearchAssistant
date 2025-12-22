"""
schemas.py - Pydantic schemas for structured LLM outputs

These schemas define the expected output format from each agent,
enabling type-safe, validated responses from the LLM.

Author: Rajesh Gupta
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClarityAnalysisResult(BaseModel):
    """Structured output from the Clarity Agent."""

    clarity_status: Literal["clear", "needs_clarification"] = Field(
        description="Whether the query is clear enough to proceed with research"
    )
    detected_company: Optional[str] = Field(
        default=None,
        description="The company name extracted from the query, if any"
    )
    clarification_request: Optional[str] = Field(
        default=None,
        description="Question to ask the user if clarification is needed"
    )
    reasoning: str = Field(
        description="Brief explanation of the clarity analysis decision"
    )


class ResearchAnalysisResult(BaseModel):
    """Structured output from the Research Agent's analysis."""

    analysis: Dict[str, str] = Field(
        default_factory=dict,
        description="Analyzed summaries of the research data"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Confidence score from 0-10 based on data quality"
    )
    confidence_reasoning: str = Field(
        description="Explanation of why this confidence score was assigned"
    )
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="List of missing information or gaps in the research"
    )


class ValidationResult(BaseModel):
    """Structured output from the Validator Agent."""

    validation_result: Literal["sufficient", "insufficient"] = Field(
        description="Whether the research findings are sufficient to answer the query"
    )
    validation_feedback: Optional[str] = Field(
        default=None,
        description="Specific feedback for improving research if insufficient"
    )
    quality_assessment: Dict[str, int] = Field(
        default_factory=dict,
        description="Quality scores for relevance, completeness, and quality (1-10 each)"
    )
    reasoning: str = Field(
        description="Detailed explanation of the validation decision"
    )


class QueryIntentAnalysis(BaseModel):
    """
    Structured analysis of query intent for enhanced understanding.

    Used by the Query Intent Classification system.
    """

    primary_intent: Literal[
        "news",
        "financial",
        "development",
        "competitor",
        "leadership",
        "general",
        "comparison"
    ] = Field(
        description="The primary intent category of the query"
    )

    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Additional intent categories that apply"
    )

    time_scope: Literal["current", "historical", "future", "unspecified"] = Field(
        default="unspecified",
        description="The temporal scope of the query"
    )

    depth_required: Literal["overview", "detailed", "comprehensive"] = Field(
        default="detailed",
        description="The level of detail expected in the response"
    )

    specific_aspects: List[str] = Field(
        default_factory=list,
        description="Specific aspects mentioned (e.g., 'stock price', 'CEO', 'products')"
    )

    company_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Confidence level (0-1) that the company was correctly identified"
    )


class GroundingValidation(BaseModel):
    """
    Validation result for response grounding against source data.

    Ensures responses only contain information present in the research.
    """

    is_grounded: bool = Field(
        description="Whether all claims in the response are grounded in source data"
    )

    grounded_claims: List[str] = Field(
        default_factory=list,
        description="Claims that are properly grounded in source data"
    )

    ungrounded_claims: List[str] = Field(
        default_factory=list,
        description="Claims that are NOT grounded in source data (potential hallucinations)"
    )

    grounding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall grounding score (0-1)"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving grounding"
    )


class RetryEffectivenessMetrics(BaseModel):
    """
    Metrics for tracking the effectiveness of research retries.

    Used to measure whether retries actually improve results.
    """

    attempt_number: int = Field(
        ge=1,
        description="Which attempt this is (1, 2, or 3)"
    )

    previous_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score from previous attempt"
    )

    current_confidence: float = Field(
        description="Confidence score from current attempt"
    )

    confidence_delta: float = Field(
        default=0.0,
        description="Change in confidence from previous attempt"
    )

    gaps_addressed: List[str] = Field(
        default_factory=list,
        description="Which gaps from previous attempt were addressed"
    )

    gaps_remaining: List[str] = Field(
        default_factory=list,
        description="Which gaps still remain"
    )

    improvement_rate: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Rate of gap resolution (0-1)"
    )

    retry_worthwhile: bool = Field(
        default=True,
        description="Whether this retry was worth the effort"
    )
