"""
Research Agent for the Research Assistant.

Conducts company research using available tools and assigns
confidence scores to findings using a hybrid scoring system.

Author: Rajesh Gupta
"""

from typing import Any, Dict

from .base import BaseAgent
from ..state import Message, ResearchFindings
from ..tools.research_tool import ResearchTool
from ..utils.confidence import calculate_hybrid_confidence, ConfidenceBreakdown


class ResearchAgent(BaseAgent):
    """
    Conducts company research using available tools.

    Responsibilities:
    - Search for company information (news, financials, developments)
    - Use mock data or search tools based on configuration
    - Assign confidence score (0-10) using hybrid scoring system
    - Structure findings for downstream processing

    Outputs:
    - research_findings: Structured ResearchFindings object
    - confidence_score: 0-10 score based on hybrid metrics
    - confidence_breakdown: Detailed scoring breakdown
    """

    def __init__(self):
        """Initialize the Research Agent with tools."""
        super().__init__()
        self.research_tool = ResearchTool()

    @property
    def name(self) -> str:
        return "ResearchAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Agent specializing in gathering company information.

Your task is to:
1. Analyze the provided research data about a company
2. Focus on: recent news, financial data, key developments
3. Assess the quality and completeness of the findings
4. Assign a confidence score (0-10) where:
   - 0-3: Very limited or unreliable information
   - 4-5: Partial information, significant gaps
   - 6-7: Good coverage, minor gaps
   - 8-10: Comprehensive and reliable information

Consider the user's specific question when assessing if the data answers their needs.

Respond ONLY with valid JSON in this exact format:
{{
    "analysis": {{
        "recent_news_summary": "Summary of news relevant to the query",
        "financial_summary": "Summary of financial/stock information",
        "developments_summary": "Summary of key developments"
    }},
    "confidence_score": 7,
    "confidence_reasoning": "Explanation of why this score was assigned",
    "gaps_identified": ["List of missing information if any"]
}}

If this is a retry after validation feedback, pay special attention to addressing the gaps mentioned."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research for the specified company.

        Args:
            state: Current workflow state

        Returns:
            State updates with research findings
        """
        company = state.get("detected_company", "Unknown Company")
        query = state.get("user_query", "")
        current_attempts = state.get("research_attempts", 0)
        validation_feedback = state.get("validation_feedback")

        attempt = current_attempts + 1

        self._log_execution("Starting research", {
            "company": company,
            "attempt": f"{attempt}/3"
        })

        # Get research data from tool
        raw_data = self.research_tool.search(
            company_name=company,
            query=query,
            validation_feedback=validation_feedback
        )

        # Use LLM to analyze and score the research
        prompt = self._create_prompt(
            "Analyze this research data and assess its quality:\n\n"
            "Company: {company}\n"
            "User Query: {query}\n\n"
            "Research Data:\n"
            "- Recent News: {recent_news}\n"
            "- Stock Info: {stock_info}\n"
            "- Key Developments: {key_developments}\n"
            "- Additional Info: {additional_info}\n\n"
            "Validation Feedback (if retry): {feedback}\n\n"
            "This is attempt {attempt} of 3. Analyze the data and assign a confidence score."
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "company": company,
            "query": query,
            "recent_news": raw_data.get("recent_news", "N/A"),
            "stock_info": raw_data.get("stock_info", "N/A"),
            "key_developments": raw_data.get("key_developments", "N/A"),
            "additional_info": str(raw_data.get("additional_info", {})),
            "feedback": validation_feedback or "N/A (first attempt)",
            "attempt": attempt
        })

        result = self._parse_json_response(response.content)

        # Extract LLM's confidence score for hybrid calculation
        llm_confidence = result.get("confidence_score", 5.0)
        if isinstance(llm_confidence, str):
            try:
                llm_confidence = float(llm_confidence)
            except ValueError:
                llm_confidence = 5.0

        llm_reasoning = result.get("confidence_reasoning", "")

        # Build structured findings
        findings = ResearchFindings(
            company_name=company,
            recent_news=raw_data.get("recent_news"),
            stock_info=raw_data.get("stock_info"),
            key_developments=raw_data.get("key_developments"),
            raw_data=raw_data,
            sources=["mock_data"] if "mock" in str(raw_data.get("source", "")).lower() else ["tavily_api"]
        )

        # Calculate hybrid confidence score (rule-based + LLM)
        # Convert findings to dict for scoring
        findings_dict = {
            "company_name": company,
            "recent_news": raw_data.get("recent_news"),
            "stock_info": raw_data.get("stock_info"),
            "key_developments": raw_data.get("key_developments"),
            "additional_info": raw_data.get("additional_info", {}),
            "source": raw_data.get("source", "mock_data"),
            "sources": findings.sources
        }

        confidence_score, confidence_breakdown = calculate_hybrid_confidence(
            findings=findings_dict,
            query=query,
            llm_score=llm_confidence,
            llm_reasoning=llm_reasoning
        )

        self._log_execution("Research completed", {
            "company": company,
            "confidence": round(confidence_score, 2),
            "llm_score": llm_confidence,
            "attempt": attempt,
            "breakdown": confidence_breakdown.to_dict()
        })

        # Build detailed reasoning from breakdown
        breakdown_summary = self._format_confidence_breakdown(confidence_breakdown)

        return {
            "research_findings": findings,
            "confidence_score": confidence_score,
            "confidence_breakdown": confidence_breakdown.to_dict(),
            "research_attempts": attempt,
            "current_agent": self.name,
            "validation_result": "pending",  # Reset for validator
            "messages": [Message(
                role="assistant",
                content=f"[Research Agent] Completed research for {company} "
                        f"(attempt {attempt}/3, confidence: {confidence_score:.1f}/10). "
                        f"{breakdown_summary}",
                agent=self.name
            )]
        }

    def _format_confidence_breakdown(self, breakdown: ConfidenceBreakdown) -> str:
        """Format confidence breakdown for message."""
        parts = []

        # Add key scores
        components = breakdown.to_dict()["components"]
        high_scores = [k for k, v in components.items() if v >= 7]
        low_scores = [k for k, v in components.items() if v < 5]

        if high_scores:
            parts.append(f"Strong: {', '.join(s.replace('_', ' ') for s in high_scores)}")

        if low_scores:
            parts.append(f"Weak: {', '.join(s.replace('_', ' ') for s in low_scores)}")

        if breakdown.gaps:
            parts.append(f"Gaps: {', '.join(breakdown.gaps[:2])}")

        return " | ".join(parts) if parts else "Scoring complete"
