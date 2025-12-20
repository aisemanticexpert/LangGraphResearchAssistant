"""
Clarity Agent for the Research Assistant.

Analyzes user queries to determine if they are clear and specific
enough for research, and extracts relevant company information.
"""

from typing import Any, Dict

from .base import BaseAgent
from ..state import Message


class ClarityAgent(BaseAgent):
    """
    Analyzes user queries for clarity and specificity.

    Responsibilities:
    - Check if company name is mentioned or identifiable
    - Detect if query is too vague
    - Extract company name if present
    - Formulate clarification question if needed

    Outputs:
    - clarity_status: "clear" or "needs_clarification"
    - detected_company: Company name if found
    - clarification_request: Question for user if unclear
    """

    @property
    def name(self) -> str:
        return "ClarityAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Clarity Analysis Agent specializing in evaluating user queries about companies.

Your task is to analyze the user's query and determine:
1. Is the query clear and specific enough to research?
2. Is a company name mentioned or identifiable?
3. What specific information is the user seeking?

IMPORTANT: Consider the conversation context. If a company was mentioned in previous messages,
the user may be asking follow-up questions about that same company.

Respond ONLY with valid JSON in this exact format:
{{
    "clarity_status": "clear" or "needs_clarification",
    "detected_company": "Company Name" or null,
    "clarification_request": "Question to ask user" or null,
    "reasoning": "Brief explanation of your analysis"
}}

A query NEEDS CLARIFICATION if:
- No company is mentioned or identifiable (and none in recent context)
- The query is too vague (e.g., "tell me about stocks")
- Multiple companies mentioned without clear focus
- The request type is ambiguous

A query is CLEAR if:
- A specific company is mentioned or clearly implied
- A company was mentioned in recent conversation context
- The user's intent is understandable (news, stock info, developments, etc.)

Examples:
- "Tell me about Apple's AI strategy" -> clear, company="Apple Inc."
- "What about their competitors?" (after discussing Tesla) -> clear, company="Tesla" (from context)
- "How are stocks doing?" -> needs_clarification, ask which company
- "Latest news" -> needs_clarification, ask which company"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query clarity and extract company information.

        Args:
            state: Current workflow state

        Returns:
            State updates with clarity analysis results
        """
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])

        self._log_execution("Analyzing query clarity", {"query": user_query[:100]})

        # Build context from conversation history
        context = self._build_context_from_messages(messages, max_messages=5)

        # Check for previously detected company in context
        previous_company = state.get("detected_company")
        if previous_company:
            context += f"\n\nPreviously discussed company: {previous_company}"

        prompt = self._create_prompt(
            "Analyze this user query about company research:\n\n"
            "Query: {query}\n\n"
            "Conversation context:\n{context}\n\n"
            "Determine if the query is clear and identify the company."
        )

        # Invoke LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "context": context
        })

        # Parse JSON response
        result = self._parse_json_response(response.content)

        # Handle parsing failure
        if not result:
            self.logger.warning("Failed to parse clarity response, defaulting to needs_clarification")
            result = {
                "clarity_status": "needs_clarification",
                "clarification_request": "I couldn't understand your query. Could you please specify which company you'd like to know about?",
                "reasoning": "Parse error"
            }

        # Build state updates
        clarity_status = result.get("clarity_status", "needs_clarification")
        detected_company = result.get("detected_company")
        clarification_request = result.get("clarification_request")

        # If no company detected but one was in context, use the previous one
        if not detected_company and previous_company and clarity_status == "clear":
            detected_company = previous_company
            self._log_execution("Using company from context", {"company": previous_company})

        updates = {
            "clarity_status": clarity_status,
            "detected_company": detected_company,
            "clarification_request": clarification_request,
            "current_agent": self.name,
            "messages": [Message(
                role="assistant",
                content=f"[Clarity Analysis] Status: {clarity_status}. {result.get('reasoning', '')}",
                agent=self.name
            )]
        }

        if clarity_status == "needs_clarification":
            updates["awaiting_human_input"] = True
            self._log_execution("Query needs clarification", {
                "request": clarification_request
            })
        else:
            self._log_execution("Query is clear", {
                "company": detected_company
            })

        return updates
