"""
Validator Agent for the Research Assistant.

Reviews research quality and completeness, determining if findings
are sufficient to answer the user's question.
"""

from typing import Any, Dict

from .base import BaseAgent
from ..state import Message


class ValidatorAgent(BaseAgent):
    """
    Validates research quality and completeness.

    Responsibilities:
    - Review research findings for quality and relevance
    - Check if findings adequately answer the user's question
    - Determine if more research is needed
    - Provide specific feedback for retry attempts

    Outputs:
    - validation_result: "sufficient" or "insufficient"
    - validation_feedback: Specific guidance for improvement
    """

    @property
    def name(self) -> str:
        return "ValidatorAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Validator Agent responsible for quality assurance.

Your task is to evaluate research findings and determine if they adequately
answer the user's original question.

Evaluate based on:
1. RELEVANCE: Does the research address what the user asked?
2. COMPLETENESS: Are the key aspects of the question covered?
3. QUALITY: Is the information specific and useful (not generic)?
4. COHERENCE: Does the information make sense together?

Respond ONLY with valid JSON in this exact format:
{{
    "validation_result": "sufficient" or "insufficient",
    "validation_feedback": "Specific feedback for improvement (if insufficient)",
    "quality_assessment": {{
        "relevance": 1-10,
        "completeness": 1-10,
        "quality": 1-10
    }},
    "reasoning": "Detailed explanation of your validation decision"
}}

Mark as INSUFFICIENT if:
- Key aspects of the user's question are not addressed
- Information seems too generic or not specific to the company
- Critical data categories are missing (e.g., user asked about stock but no stock info)
- The confidence score is low and data quality can be improved

Mark as SUFFICIENT if:
- The research reasonably addresses the user's query
- Information is specific to the company
- Even if not perfect, the response would be helpful

Be constructive: If insufficient, provide SPECIFIC suggestions for what to look for on retry.
Remember: Maximum 3 attempts, so be pragmatic about expectations."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate research findings.

        Args:
            state: Current workflow state

        Returns:
            State updates with validation results
        """
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

        # Build findings string for LLM
        findings_str = self._format_findings(research_findings)

        prompt = self._create_prompt(
            "Validate these research findings:\n\n"
            "User's Original Question: {query}\n\n"
            "Company: {company}\n\n"
            "Research Findings:\n{findings}\n\n"
            "Confidence Score from Research Agent: {confidence}/10\n\n"
            "Current Attempt: {attempt}/3\n\n"
            "Determine if these findings are sufficient to answer the user's question."
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "company": company,
            "findings": findings_str,
            "confidence": confidence_score,
            "attempt": attempts
        })

        result = self._parse_json_response(response.content)

        # Handle parsing failure
        if not result:
            self.logger.warning("Failed to parse validation response, defaulting to sufficient")
            result = {
                "validation_result": "sufficient",
                "reasoning": "Parse error - proceeding with available data"
            }

        validation_result = result.get("validation_result", "sufficient")
        validation_feedback = result.get("validation_feedback")

        # Log quality assessment if available
        quality = result.get("quality_assessment", {})
        self._log_execution("Validation completed", {
            "result": validation_result,
            "relevance": quality.get("relevance"),
            "completeness": quality.get("completeness"),
            "quality": quality.get("quality")
        })

        return {
            "validation_result": validation_result,
            "validation_feedback": validation_feedback,
            "current_agent": self.name,
            "messages": [Message(
                role="assistant",
                content=f"[Validator] Result: {validation_result}. {result.get('reasoning', '')[:200]}",
                agent=self.name
            )]
        }

    def _format_findings(self, findings) -> str:
        """Format research findings for the LLM prompt."""
        if not findings:
            return "No findings available"

        if isinstance(findings, dict):
            parts = []
            for key, value in findings.items():
                if value and key != "raw_data":
                    parts.append(f"- {key}: {value}")
            return "\n".join(parts) if parts else "No structured findings"

        # Handle ResearchFindings object
        parts = []
        if hasattr(findings, "recent_news") and findings.recent_news:
            parts.append(f"- Recent News: {findings.recent_news}")
        if hasattr(findings, "stock_info") and findings.stock_info:
            parts.append(f"- Stock Info: {findings.stock_info}")
        if hasattr(findings, "key_developments") and findings.key_developments:
            parts.append(f"- Key Developments: {findings.key_developments}")

        return "\n".join(parts) if parts else "No structured findings"
