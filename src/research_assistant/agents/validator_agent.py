"""
Validator Agent for the Research Assistant
===========================================

The Validator Agent acts as a quality gate, responsible for:
    1. Reviewing research findings quality
    2. Assessing data completeness
    3. Evaluating query relevance
    4. Providing feedback for improvement
    5. Controlling the retry loop (max 3 attempts)

This agent implements a multi-criteria weighted assessment to determine
if research is sufficient for synthesis.

Routing Logic:
    - validation_result == "sufficient" -> Synthesis Agent
    - validation_result == "insufficient" AND attempts < 3 -> Research Agent
    - validation_result == "insufficient" AND attempts >= 3 -> Synthesis Agent

Assessment Criteria (with weights):
    - Confidence Score: 30%
    - Data Completeness: 25%
    - Query Relevance: 20%
    - News Coverage: 15%
    - Financial Data: 10%

Author: Rajesh Gupta
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..state import Message, ResearchFindings


@dataclass
class ValidationCriteria:
    """
    Configurable validation criteria with weights.

    All weights should sum to 1.0 for proper scoring.

    Attributes:
        min_confidence_threshold: Minimum acceptable confidence (0-10)
        min_completeness_threshold: Minimum data completeness (0-1)
        min_relevance_threshold: Minimum query relevance (0-1)
        max_attempts: Maximum research attempts allowed
        weights: Weight for each criterion in final assessment
    """
    min_confidence_threshold: float = 5.0
    min_completeness_threshold: float = 0.4
    min_relevance_threshold: float = 0.5
    max_attempts: int = 3
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "confidence_score": 0.30,
                "data_completeness": 0.25,
                "query_relevance": 0.20,
                "news_coverage": 0.15,
                "financial_data": 0.10
            }


class ValidatorAgent(BaseAgent):
    """
    Validates research quality and determines if findings are sufficient.

    This agent serves as a quality gate between Research and Synthesis,
    ensuring users receive high-quality, relevant information.

    Features:
        - Multi-criteria weighted validation
        - Specific, actionable feedback for improvement
        - Retry attempt tracking and limiting
        - Quality metrics calculation
        - LLM-assisted semantic relevance check

    Outputs:
        - validation_result: "sufficient" or "insufficient"
        - validation_feedback: Specific guidance for improvement
        - data_completeness_score: How complete the data is
        - relevance_score: How relevant to user's query
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        criteria: Optional[ValidationCriteria] = None
    ):
        """
        Initialize the Validator Agent.

        Args:
            model_name: LLM model to use
            temperature: LLM temperature
            criteria: Validation criteria configuration
        """
        super().__init__(model_name=model_name, temperature=temperature)
        self.criteria = criteria or ValidationCriteria()

    @property
    def name(self) -> str:
        return "ValidatorAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Validator Agent responsible for quality assurance.

Your task is to evaluate research findings and determine if they adequately
answer the user's original question.

Evaluate based on these criteria:
1. RELEVANCE: Does the research directly address the user's question?
   - High: Research covers exactly what was asked
   - Medium: Research covers related topics but not the specific question
   - Low: Research is tangential or off-topic

2. COMPLETENESS: Are the key aspects covered?
   - Check for: news, financials, developments, stock info
   - Missing critical categories lower the score

3. QUALITY: Is the information specific and actionable?
   - Specific data (numbers, dates, facts) = higher quality
   - Generic statements = lower quality

4. COHERENCE: Does the information make sense together?
   - Consistent data across sources
   - No contradictory information

Respond ONLY with valid JSON in this exact format:
{{
    "validation_result": "sufficient" or "insufficient",
    "relevance_score": 0.0 to 1.0,
    "completeness_score": 0.0 to 1.0,
    "quality_score": 0.0 to 1.0,
    "missing_elements": ["list of missing data types if any"],
    "validation_feedback": "Specific, actionable feedback for improvement",
    "reasoning": "Detailed explanation of your validation decision"
}}

Be pragmatic: After 3 attempts, even partial information should proceed to synthesis.
Focus on whether the research provides VALUE to the user, not perfection."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate research findings.

        This is the main entry point called by LangGraph.

        Steps:
            1. Calculate completeness and quality metrics
            2. Use LLM for semantic relevance assessment
            3. Compute weighted validation score
            4. Generate feedback for improvement
            5. Determine if sufficient or needs retry

        Args:
            state: Current workflow state

        Returns:
            State updates with validation results
        """
        start_time = datetime.now()

        user_query = state.get("user_query", "")
        company = state.get("detected_company", "Unknown")
        research_findings = state.get("research_findings")
        confidence_score = state.get("confidence_score", 0)
        attempts = state.get("research_attempts", 0)

        self._log_execution("Validating research", {
            "company": company,
            "confidence": confidence_score,
            "attempt": attempts
        })

        # Calculate rule-based scores
        completeness_score = self._calculate_completeness(research_findings)
        news_score = self._calculate_news_coverage(research_findings)
        financial_score = self._calculate_financial_coverage(research_findings)

        # Use LLM for semantic relevance check
        llm_assessment = self._get_llm_assessment(
            user_query, company, research_findings, confidence_score, attempts
        )

        # Get relevance from LLM or estimate
        relevance_score = llm_assessment.get("relevance_score", 0.5)
        quality_score = llm_assessment.get("quality_score", 0.5)
        missing_elements = llm_assessment.get("missing_elements", [])

        # Calculate weighted validation score
        validation_score = self._calculate_weighted_score(
            confidence_score=confidence_score,
            completeness_score=completeness_score,
            relevance_score=relevance_score,
            news_score=news_score,
            financial_score=financial_score
        )

        # Determine validation result
        validation_result, validation_feedback = self._determine_result(
            validation_score=validation_score,
            confidence_score=confidence_score,
            completeness_score=completeness_score,
            relevance_score=relevance_score,
            attempts=attempts,
            missing_elements=missing_elements,
            llm_assessment=llm_assessment
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        self._log_execution("Validation completed", {
            "result": validation_result,
            "validation_score": round(validation_score, 2),
            "completeness": round(completeness_score, 2),
            "relevance": round(relevance_score, 2),
            "processing_time_ms": round(processing_time, 2)
        })

        # Build summary message
        summary = self._build_validation_summary(
            validation_result, validation_score, completeness_score,
            relevance_score, attempts
        )

        return {
            "validation_result": validation_result,
            "validation_feedback": validation_feedback,
            "data_completeness_score": completeness_score,
            "relevance_score": relevance_score,
            "current_agent": self.name,
            "messages": [Message(
                role="assistant",
                content=summary,
                agent=self.name,
                metadata={
                    "validation_result": validation_result,
                    "validation_score": validation_score,
                    "completeness_score": completeness_score,
                    "relevance_score": relevance_score,
                    "processing_time_ms": processing_time
                }
            )]
        }

    def _calculate_completeness(self, findings: Optional[ResearchFindings]) -> float:
        """
        Calculate data completeness score.

        Args:
            findings: Research findings to evaluate

        Returns:
            Completeness score (0-1)
        """
        if not findings:
            return 0.0

        # Use the built-in method if available
        if hasattr(findings, 'get_data_completeness'):
            return findings.get_data_completeness()

        # Manual calculation for dict-like findings
        total_fields = 6
        present = 0

        if hasattr(findings, 'recent_news') and findings.recent_news:
            present += 1
        elif isinstance(findings, dict) and findings.get('recent_news'):
            present += 1

        if hasattr(findings, 'stock_info') and findings.stock_info:
            present += 1
        elif isinstance(findings, dict) and findings.get('stock_info'):
            present += 1

        if hasattr(findings, 'financials') and findings.financials:
            present += 1
        elif isinstance(findings, dict) and findings.get('financials'):
            present += 1

        if hasattr(findings, 'key_developments') and findings.key_developments:
            present += 1
        elif isinstance(findings, dict) and findings.get('key_developments'):
            present += 1

        if hasattr(findings, 'sector') and findings.sector:
            present += 1
        elif isinstance(findings, dict) and findings.get('sector'):
            present += 1

        if hasattr(findings, 'ticker') and findings.ticker:
            present += 1
        elif isinstance(findings, dict) and findings.get('ticker'):
            present += 1

        return present / total_fields

    def _calculate_news_coverage(self, findings: Optional[ResearchFindings]) -> float:
        """
        Calculate news coverage score.

        Args:
            findings: Research findings

        Returns:
            News coverage score (0-1)
        """
        if not findings:
            return 0.0

        news = None
        if hasattr(findings, 'recent_news'):
            news = findings.recent_news
        elif isinstance(findings, dict):
            news = findings.get('recent_news')

        if not news:
            return 0.0

        if isinstance(news, list):
            return min(1.0, len(news) / 5)  # Max score at 5 items
        elif isinstance(news, str) and news.strip():
            return 0.5  # Partial credit for string format

        return 0.0

    def _calculate_financial_coverage(self, findings: Optional[ResearchFindings]) -> float:
        """
        Calculate financial data coverage score.

        Args:
            findings: Research findings

        Returns:
            Financial coverage score (0-1)
        """
        if not findings:
            return 0.0

        financials = None
        if hasattr(findings, 'financials'):
            financials = findings.financials
        elif isinstance(findings, dict):
            financials = findings.get('financials')

        if not financials:
            return 0.0

        if hasattr(financials, 'get_completeness_score'):
            return financials.get_completeness_score()

        # Manual check for key fields
        key_fields = ['revenue', 'eps', 'pe_ratio', 'profit_margin', 'net_income']
        if isinstance(financials, dict):
            present = sum(1 for f in key_fields if financials.get(f) is not None)
            return present / len(key_fields)

        return 0.5  # Partial credit if financials exist

    def _calculate_weighted_score(
        self,
        confidence_score: float,
        completeness_score: float,
        relevance_score: float,
        news_score: float,
        financial_score: float
    ) -> float:
        """
        Calculate weighted validation score.

        Uses configured weights to combine individual scores.

        Args:
            confidence_score: Research confidence (0-10, will be normalized)
            completeness_score: Data completeness (0-1)
            relevance_score: Query relevance (0-1)
            news_score: News coverage (0-1)
            financial_score: Financial data coverage (0-1)

        Returns:
            Weighted validation score (0-1)
        """
        weights = self.criteria.weights

        # Normalize confidence to 0-1
        normalized_confidence = confidence_score / 10.0

        weighted_score = (
            weights["confidence_score"] * normalized_confidence +
            weights["data_completeness"] * completeness_score +
            weights["query_relevance"] * relevance_score +
            weights["news_coverage"] * news_score +
            weights["financial_data"] * financial_score
        )

        return min(1.0, max(0.0, weighted_score))

    def _determine_result(
        self,
        validation_score: float,
        confidence_score: float,
        completeness_score: float,
        relevance_score: float,
        attempts: int,
        missing_elements: List[str],
        llm_assessment: Dict[str, Any]
    ) -> tuple:
        """
        Determine validation result and generate feedback.

        Args:
            validation_score: Weighted validation score
            confidence_score: Research confidence
            completeness_score: Data completeness
            relevance_score: Query relevance
            attempts: Current attempt number
            missing_elements: List of missing data elements
            llm_assessment: LLM assessment results

        Returns:
            Tuple of (validation_result, validation_feedback)
        """
        # Check if max attempts reached - proceed regardless
        if attempts >= self.criteria.max_attempts:
            return (
                "sufficient",
                "Maximum attempts reached. Proceeding with best available data."
            )

        # Use LLM result if available
        llm_result = llm_assessment.get("validation_result")
        if llm_result in ["sufficient", "insufficient"]:
            # Trust LLM assessment but add our metrics
            if llm_result == "sufficient":
                return "sufficient", None

            # Generate feedback for insufficient
            feedback = llm_assessment.get("validation_feedback") or self._generate_feedback(
                completeness_score, relevance_score, confidence_score, missing_elements
            )
            return "insufficient", feedback

        # Rule-based fallback
        # Check minimum thresholds
        if confidence_score < self.criteria.min_confidence_threshold:
            feedback = f"Confidence score ({confidence_score:.1f}/10) is below threshold. "
            if missing_elements:
                feedback += f"Missing: {', '.join(missing_elements[:3])}. "
            feedback += "Try to gather more comprehensive data."
            return "insufficient", feedback

        if completeness_score < self.criteria.min_completeness_threshold:
            feedback = self._generate_feedback(
                completeness_score, relevance_score, confidence_score, missing_elements
            )
            return "insufficient", feedback

        if relevance_score < self.criteria.min_relevance_threshold:
            feedback = (
                "Research doesn't fully address the user's question. "
                "Focus on gathering information more directly related to their query."
            )
            return "insufficient", feedback

        # Validation score threshold
        if validation_score >= 0.6:
            return "sufficient", None
        else:
            feedback = self._generate_feedback(
                completeness_score, relevance_score, confidence_score, missing_elements
            )
            return "insufficient", feedback

    def _generate_feedback(
        self,
        completeness_score: float,
        relevance_score: float,
        confidence_score: float,
        missing_elements: List[str]
    ) -> str:
        """
        Generate specific, actionable feedback for improvement.

        Args:
            completeness_score: Data completeness
            relevance_score: Query relevance
            confidence_score: Research confidence
            missing_elements: Missing data elements

        Returns:
            Feedback string
        """
        feedback_parts = []

        # Identify main issues
        if completeness_score < 0.5:
            feedback_parts.append("Data coverage is limited")
            if missing_elements:
                feedback_parts.append(f"Missing: {', '.join(missing_elements[:3])}")

        if relevance_score < 0.5:
            feedback_parts.append("Research may not fully address the user's question")

        if confidence_score < 5.0:
            feedback_parts.append(f"Low confidence ({confidence_score:.1f}/10)")

        # Add actionable suggestion
        if not feedback_parts:
            feedback_parts.append("Consider gathering more specific information")

        feedback_parts.append("Try to find more detailed, recent data")

        return ". ".join(feedback_parts) + "."

    def _get_llm_assessment(
        self,
        query: str,
        company: str,
        findings: Optional[ResearchFindings],
        confidence: float,
        attempts: int
    ) -> Dict[str, Any]:
        """
        Get LLM assessment of research quality.

        Args:
            query: User's original query
            company: Company being researched
            findings: Research findings
            confidence: Confidence score
            attempts: Current attempt

        Returns:
            LLM assessment dictionary
        """
        try:
            findings_str = self._format_findings_for_llm(findings)

            prompt = self._create_prompt(
                "Validate these research findings:\n\n"
                "User's Original Question: {query}\n\n"
                "Company: {company}\n\n"
                "Research Findings:\n{findings}\n\n"
                "Confidence Score: {confidence}/10\n\n"
                "Current Attempt: {attempt}/3\n\n"
                "Determine if these findings adequately answer the user's question."
            )

            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "company": company,
                "findings": findings_str,
                "confidence": confidence,
                "attempt": attempts
            })

            result = self._parse_json_response(response.content)
            return result if result else {}

        except Exception as e:
            self.logger.warning(f"LLM assessment failed: {e}")
            return {}

    def _format_findings_for_llm(self, findings: Optional[ResearchFindings]) -> str:
        """
        Format findings for LLM prompt.

        Args:
            findings: Research findings

        Returns:
            Formatted string
        """
        if not findings:
            return "No findings available"

        parts = []

        # Handle Pydantic model
        if hasattr(findings, 'company_name'):
            if findings.company_name:
                parts.append(f"Company: {findings.company_name}")
            if hasattr(findings, 'recent_news') and findings.recent_news:
                if isinstance(findings.recent_news, list):
                    news_titles = [n.title if hasattr(n, 'title') else str(n) for n in findings.recent_news[:3]]
                    parts.append(f"Recent News: {'; '.join(news_titles)}")
                else:
                    parts.append(f"Recent News: {findings.recent_news}")
            if hasattr(findings, 'stock_info') and findings.stock_info:
                if hasattr(findings.stock_info, 'to_display_string'):
                    parts.append(f"Stock Info: {findings.stock_info.to_display_string()}")
                else:
                    parts.append(f"Stock Info: {findings.stock_info}")
            if hasattr(findings, 'key_developments') and findings.key_developments:
                parts.append(f"Key Developments: {'; '.join(findings.key_developments[:3])}")

        # Handle dict
        elif isinstance(findings, dict):
            for key, value in findings.items():
                if value and key not in ["raw_data", "factor_data"]:
                    parts.append(f"- {key}: {str(value)[:200]}")

        return "\n".join(parts) if parts else "No structured findings"

    def _build_validation_summary(
        self,
        result: str,
        score: float,
        completeness: float,
        relevance: float,
        attempts: int
    ) -> str:
        """
        Build validation summary message.

        Args:
            result: Validation result
            score: Validation score
            completeness: Completeness score
            relevance: Relevance score
            attempts: Attempt number

        Returns:
            Summary message
        """
        parts = [
            f"[Validator] Result: {result.upper()}",
            f"Score: {score:.2f}",
            f"Completeness: {completeness:.0%}",
            f"Relevance: {relevance:.0%}",
            f"Attempt: {attempts}/3"
        ]

        return " | ".join(parts)
