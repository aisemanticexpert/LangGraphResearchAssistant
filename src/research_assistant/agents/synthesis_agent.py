"""
Synthesis Agent for the Research Assistant.

Creates user-friendly summaries from research findings,
maintaining conversation context.
"""

from typing import Any, Dict

from .base import BaseAgent
from ..state import Message


class SynthesisAgent(BaseAgent):
    """
    Synthesizes research findings into user-friendly responses.

    Responsibilities:
    - Create coherent summary from research findings
    - Format response appropriately for users
    - Maintain context from conversation history
    - Acknowledge any limitations in the research

    Outputs:
    - final_response: User-friendly response text
    """

    @property
    def name(self) -> str:
        return "SynthesisAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Synthesis Agent specializing in creating clear, informative summaries.

Your task is to take research findings and create a helpful response for the user.

Guidelines:
1. START with a direct answer to their question
2. ORGANIZE information clearly with sections or bullet points
3. BE SPECIFIC - use actual data from the findings, not generic statements
4. ACKNOWLEDGE limitations if the research has gaps
5. OFFER to provide more details if relevant
6. Keep the tone professional but conversational
7. Don't repeat the question back - get straight to the answer

Format suggestions:
- Use bullet points for multiple items
- Use headers for different categories (News, Stock Info, Developments)
- Include relevant numbers and specifics
- Keep it concise but comprehensive

If the research was limited (low confidence or max attempts reached), acknowledge this
and still provide the best answer possible with available information.

Do NOT output JSON - write a natural language response for the user."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize findings into final response.

        Args:
            state: Current workflow state

        Returns:
            State updates with final response
        """
        user_query = state.get("user_query", "")
        company = state.get("detected_company", "the company")
        research_findings = state.get("research_findings")
        confidence_score = state.get("confidence_score", 0)
        attempts = state.get("research_attempts", 0)
        messages = state.get("messages", [])

        self._log_execution("Synthesizing response", {
            "company": company,
            "confidence": confidence_score
        })

        # Format findings for the prompt
        findings_str = self._format_findings(research_findings)

        # Build conversation context
        context = self._build_context_from_messages(messages, max_messages=5)

        # Determine if we should note limitations
        limitation_note = ""
        if confidence_score < 6:
            limitation_note = "Note: Research confidence is moderate - some information may be limited."
        if attempts >= 3:
            limitation_note = "Note: Maximum research attempts reached - providing best available information."

        prompt = self._create_prompt(
            "Create a comprehensive response for this research query:\n\n"
            "User's Question: {query}\n\n"
            "Company: {company}\n\n"
            "Research Findings:\n{findings}\n\n"
            "Confidence Score: {confidence}/10\n"
            "Research Attempts: {attempts}\n"
            "{limitation}\n\n"
            "Conversation Context:\n{context}\n\n"
            "Write a clear, helpful response that directly answers the user's question."
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "company": company,
            "findings": findings_str,
            "confidence": confidence_score,
            "attempts": attempts,
            "limitation": limitation_note,
            "context": context
        })

        final_response = response.content

        self._log_execution("Response synthesized", {
            "response_length": len(final_response)
        })

        return {
            "final_response": final_response,
            "current_agent": self.name,
            "messages": [Message(
                role="assistant",
                content=final_response,
                agent=self.name
            )]
        }

    def _format_findings(self, findings) -> str:
        """Format research findings for the synthesis prompt."""
        if not findings:
            return "No research findings available"

        if isinstance(findings, dict):
            parts = []
            for key, value in findings.items():
                if value and key not in ["raw_data", "sources"]:
                    formatted_key = key.replace("_", " ").title()
                    parts.append(f"{formatted_key}: {value}")
            return "\n".join(parts) if parts else "No structured findings"

        # Handle ResearchFindings object
        parts = []
        if hasattr(findings, "company_name") and findings.company_name:
            parts.append(f"Company: {findings.company_name}")
        if hasattr(findings, "recent_news") and findings.recent_news:
            parts.append(f"Recent News: {findings.recent_news}")
        if hasattr(findings, "stock_info") and findings.stock_info:
            parts.append(f"Stock Information: {findings.stock_info}")
        if hasattr(findings, "key_developments") and findings.key_developments:
            parts.append(f"Key Developments: {findings.key_developments}")

        # Include additional info from raw_data if available
        if hasattr(findings, "raw_data") and findings.raw_data:
            additional = findings.raw_data.get("additional_info", {})
            if additional:
                if additional.get("competitors"):
                    parts.append(f"Competitors: {', '.join(additional['competitors'])}")
                if additional.get("industry"):
                    parts.append(f"Industry: {additional['industry']}")
                if additional.get("ceo"):
                    parts.append(f"CEO: {additional['ceo']}")

        return "\n".join(parts) if parts else "No structured findings"
