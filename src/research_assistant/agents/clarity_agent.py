"""
Clarity Agent for the Research Assistant
=========================================

The Clarity Agent is the first node in the LangGraph workflow, responsible for:
    1. Validating query safety via guardrails
    2. Analyzing query clarity and specificity
    3. Extracting company names and ticker symbols
    4. Classifying query intent accurately
    5. Triggering human-in-the-loop for unclear queries

This agent uses LLM for intelligent intent classification to handle
nuanced queries like "Tesla owner" vs "Tesla stock price".

Routing Logic:
    - clarity_status == "clear" -> Research Agent
    - clarity_status == "needs_clarification" -> Human Interrupt

Author: Rajesh Gupta
"""

import re
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from .base import BaseAgent
from ..state import Message, QueryIntent
from ..guardrails import (
    InputGuardrails,
    CompanyNameValidator,
    GuardrailConfig,
    AuditLogger,
    GuardrailResult
)


class ClarityAgent(BaseAgent):
    """
    Analyzes user queries for clarity, safety, and specificity.

    This is the gateway agent that ensures:
        - Queries are safe (no prompt injection, illegal requests)
        - Company name is identifiable
        - Query intent is correctly understood
        - User gets helpful clarification when needed

    The agent uses LLM for intelligent intent classification to handle
    nuanced queries like:
        - "Tesla owner" -> leadership intent (CEO info)
        - "Tesla stock" -> stock_price intent
        - "Tell me about Apple" -> company_overview intent

    Features:
        - LLM-powered intent classification for accuracy
        - Pattern-based fast path for common queries
        - Multi-strategy company extraction
        - Follow-up question detection
        - Guardrail integration
        - Audit logging

    Outputs:
        - clarity_status: "clear" or "needs_clarification"
        - detected_company: Company name if found
        - detected_ticker: Stock ticker if found
        - query_intent: Accurately classified intent
        - clarification_request: Question for user if unclear
    """

    # Query intent patterns for fast-path classification
    # These are checked first before falling back to LLM
    INTENT_PATTERNS = {
        "leadership": [
            r"\b(owner|ceo|chief|founder|chairman|president|executive|management|leadership|who\s+(runs?|leads?|owns?))\b",
            r"\b(who\s+is\s+the\s+)?(ceo|owner|founder|head)\b",
            r"\bwho\s+(founded|started|created|owns?)\b",
        ],
        "stock_price": [
            r"\b(stock|share)\s*(price|value|worth)\b",
            r"\btrading\s+at\b",
            r"\bcurrent\s+price\b",
            r"\bhow\s+much\s+(is|does|are)\b.*\b(stock|share|trading)\b",
            r"\bstock\s+performance\b",
            r"\bprice\s+target\b",
            r"\b52.?week\b",
            r"\bmarket\s+cap\b",
        ],
        "financial_analysis": [
            r"\bfinancials?\b",
            r"\brevenue\b",
            r"\bearnings\b",
            r"\bprofit\b",
            r"\bincome\s+statement\b",
            r"\bbalance\s+sheet\b",
            r"\bcash\s+flow\b",
            r"\bpe\s+ratio\b",
            r"\bp/e\b",
            r"\beps\b",
            r"\bmargin\b",
            r"\bdebt\b",
            r"\broe\b",
            r"\bfundamental\b",
        ],
        "news_developments": [
            r"\bnews\b",
            r"\brecent\s+(news|developments?|updates?)\b",
            r"\blatest\b",
            r"\bwhat'?s\s+(happening|new|going\s+on)\b",
            r"\bannouncements?\b",
            r"\bbreaking\b",
            r"\bheadlines?\b",
        ],
        "competitor_analysis": [
            r"\bcompetitors?\b",
            r"\bcompetition\b",
            r"\bcompare\s+(to|with)\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\brivals?\b",
            r"\bmarket\s+share\b",
            r"\balternatives?\b",
        ],
        "investment_research": [
            r"\b(should\s+i\s+)?(buy|sell|invest)\b",
            r"\binvestment\b",
            r"\banalyst\s+ratings?\b",
            r"\brecommendations?\b",
            r"\boutlook\b",
            r"\bforecast\b",
            r"\bgrowth\s+potential\b",
        ],
        "products_services": [
            r"\bproducts?\b",
            r"\bservices?\b",
            r"\bofferings?\b",
            r"\bwhat\s+(do|does)\s+.+\s+(make|sell|offer|provide)\b",
        ],
        "company_overview": [
            r"\b(tell|what).*(about)\b",
            r"\boverview\s+of\b",
            r"\bsummary\s+of\b",
            r"\bwhat\s+(is|does)\b",
            r"\bintroduce\b",
            r"\bdescribe\b",
        ],
        "follow_up": [
            r"\b(tell\s+me\s+)?more\s+about\b",
            r"\bwhat\s+about\s+their\b",
            r"\bhow\s+about\b",
            r"\band\s+(what|how|why)\b",
            r"\bcan\s+you\s+(also|elaborate)\b",
            r"\badditionally\b",
            r"\bfurthermore\b",
        ],
    }

    # Patterns that indicate vague queries
    VAGUE_PATTERNS = [
        r"^(what|how|tell|give)\s+(me\s+)?$",
        r"^help\s*$",
        r"^research\s*$",
        r"^analyze\s*$",
        r"^(the\s+)?(company|stock|market)\s*$",
        r"^stocks?\s*$",
        r"^info\s*$",
    ]

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        guardrail_config: Optional[GuardrailConfig] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        """
        Initialize the Clarity Agent.

        Args:
            model_name: LLM model to use
            temperature: LLM temperature setting
            guardrail_config: Configuration for input validation
            audit_logger: Optional audit logger for compliance
        """
        super().__init__(model_name=model_name, temperature=temperature)

        # Initialize guardrails
        self.input_guardrails = InputGuardrails(guardrail_config)
        self.company_validator = CompanyNameValidator
        self.audit_logger = audit_logger

        # Pre-compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._intent_regex = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        self._vague_regex = [
            re.compile(p, re.IGNORECASE) for p in self.VAGUE_PATTERNS
        ]

    @property
    def name(self) -> str:
        return "ClarityAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Clarity Analysis Agent specializing in evaluating user queries about companies and stocks.

Your task is to analyze the user's query and determine:
1. Is the query clear and specific enough to research?
2. What company is mentioned or implied?
3. What SPECIFIC information is the user seeking?

CRITICAL - Intent Classification:
You MUST accurately classify the user's intent. Examples:
- "Tesla owner" or "Who owns Tesla" -> intent: "leadership" (asking about CEO/owner)
- "Tesla stock price" -> intent: "stock_price" (asking about stock)
- "Tell me about Apple" -> intent: "company_overview" (general info)
- "Apple news" -> intent: "news_developments" (recent news)
- "Microsoft financials" -> intent: "financial_analysis" (financial data)
- "Who is the CEO of Amazon" -> intent: "leadership" (asking about leadership)
- "Google competitors" -> intent: "competitor_analysis"
- "Should I buy NVIDIA" -> intent: "investment_research"

Respond ONLY with valid JSON in this exact format:
{{
    "clarity_status": "clear" or "needs_clarification",
    "detected_company": "Company Name" or null,
    "detected_ticker": "TICKER" or null,
    "query_intent": "leadership|stock_price|financial_analysis|news_developments|competitor_analysis|investment_research|products_services|company_overview|follow_up|general",
    "specific_question": "What exactly the user wants to know",
    "clarification_request": "Question to ask user" or null,
    "reasoning": "Brief explanation of your analysis"
}}

IMPORTANT:
- "leadership" intent is for questions about CEO, owner, founder, executives, management
- "stock_price" is for stock/share price, market cap, trading info
- "company_overview" is for general "tell me about" questions
- Be precise with intent - it affects what information we gather"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query clarity and extract company information.

        This is the main entry point called by LangGraph.

        Steps:
            1. Validate input via guardrails
            2. Use LLM for intelligent analysis (company + intent)
            3. Determine clarity status
            4. Generate clarification if needed

        Args:
            state: Current workflow state

        Returns:
            State updates with clarity analysis results
        """
        start_time = datetime.now()
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])

        self._log_execution("Analyzing query clarity", {"query": user_query[:100]})

        # Step 1: Validate input via guardrails
        validation_result = self.input_guardrails.validate_query(user_query)

        if not validation_result.passed:
            return self._handle_validation_failure(state, validation_result)

        # Use sanitized query
        sanitized_query = validation_result.sanitized_content or user_query

        # Step 2: Use LLM for intelligent analysis (more accurate than patterns)
        llm_result = self._use_llm_for_analysis(sanitized_query, messages, state)

        if llm_result:
            company_name = llm_result.get("detected_company")
            ticker = llm_result.get("detected_ticker")
            intent = llm_result.get("query_intent", "general")
            specific_question = llm_result.get("specific_question", "")
            clarity_status = llm_result.get("clarity_status", "clear")
            clarification_request = llm_result.get("clarification_request")
        else:
            # Fallback to pattern-based extraction
            company_name, ticker = self._extract_company(sanitized_query, state)
            intent = self._classify_intent_patterns(sanitized_query)
            specific_question = sanitized_query
            clarity_status, clarification_request = self._determine_clarity(
                sanitized_query, company_name, ticker, intent, state
            )

        # Normalize company name if found
        if company_name:
            normalized_company, normalized_ticker = self.company_validator.normalize_company_name(company_name)
            if normalized_company:
                company_name = normalized_company
                ticker = normalized_ticker or ticker

        # Check for vagueness
        is_vague = self._is_vague_query(sanitized_query, company_name)
        if is_vague and not clarification_request:
            clarity_status = "needs_clarification"
            clarification_request = self._generate_clarification_request("too_vague", sanitized_query, intent)

        # If no company and not a follow-up, ask for clarification
        if not company_name and not self._is_follow_up(state):
            clarity_status = "needs_clarification"
            clarification_request = self._generate_clarification_request("company_missing", sanitized_query, intent)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        self._log_execution("Query analyzed", {
            "company": company_name,
            "intent": intent,
            "clarity": clarity_status
        })

        # Build state updates
        updates = {
            "clarity_status": clarity_status,
            "clarification_request": clarification_request,
            "detected_company": company_name,
            "detected_ticker": ticker,
            "query_intent": intent,
            "current_agent": self.name,
            "messages": [Message(
                role="assistant",
                content=f"[Clarity Analysis] Status: {clarity_status}, Company: {company_name or 'N/A'}, Intent: {intent}",
                agent=self.name,
                metadata={
                    "clarity_status": clarity_status,
                    "detected_company": company_name,
                    "detected_ticker": ticker,
                    "query_intent": intent,
                    "specific_question": specific_question,
                    "processing_time_ms": processing_time
                }
            )]
        }

        # Handle clarification needed
        if clarity_status == "needs_clarification":
            updates["awaiting_human_input"] = True

        # Audit logging
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="clarity_analysis",
                session_id=state.get("session_id", "unknown"),
                user_id=state.get("user_id"),
                details={
                    "query": sanitized_query[:100],
                    "clarity_status": clarity_status,
                    "detected_company": company_name,
                    "detected_ticker": ticker,
                    "intent": intent,
                    "processing_time_ms": processing_time
                }
            )

        return updates

    def _extract_company(
        self,
        query: str,
        state: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract company name and ticker from query using pattern matching.

        Args:
            query: Sanitized user query
            state: Current workflow state

        Returns:
            Tuple of (company_name, ticker) or (None, None)
        """
        # Strategy 1: Direct extraction via CompanyNameValidator
        company, ticker = self.company_validator.normalize_company_name(query)

        if company:
            return company, ticker

        # Strategy 2: Check conversation context
        messages = state.get("messages", [])
        if messages and len(messages) > 1:
            for msg in reversed(messages[:-1]):
                if hasattr(msg, 'metadata') and msg.metadata:
                    if msg.metadata.get("detected_company"):
                        return msg.metadata["detected_company"], msg.metadata.get("detected_ticker")
                elif isinstance(msg, dict) and msg.get("metadata"):
                    if msg["metadata"].get("detected_company"):
                        return msg["metadata"]["detected_company"], msg["metadata"].get("detected_ticker")

        # Strategy 3: Check state for previous company
        if state.get("detected_company"):
            return state["detected_company"], state.get("detected_ticker")

        return None, None

    def _classify_intent_patterns(self, query: str) -> str:
        """
        Classify intent using pattern matching (fallback method).

        The order of checking matters - more specific intents first.

        Args:
            query: User query

        Returns:
            Intent string
        """
        query_lower = query.lower()

        # Check in order of specificity (most specific first)
        intent_order = [
            "leadership",
            "stock_price",
            "financial_analysis",
            "news_developments",
            "competitor_analysis",
            "investment_research",
            "products_services",
            "follow_up",
            "company_overview",
        ]

        for intent in intent_order:
            if intent in self._intent_regex:
                for pattern in self._intent_regex[intent]:
                    if pattern.search(query_lower):
                        return intent

        return "general"

    def _is_vague_query(
        self,
        query: str,
        company_name: Optional[str]
    ) -> bool:
        """
        Check if query is too vague to process.

        Args:
            query: User query
            company_name: Detected company name

        Returns:
            True if query is too vague
        """
        # Very short queries without company are vague
        if len(query.split()) < 2 and not company_name:
            return True

        # Match known vague patterns
        for pattern in self._vague_regex:
            if pattern.match(query):
                return True

        return False

    def _is_follow_up(self, state: Dict[str, Any]) -> bool:
        """
        Determine if current query is a follow-up.

        Args:
            state: Current workflow state

        Returns:
            True if this is a follow-up question
        """
        messages = state.get("messages", [])

        if len(messages) < 2:
            return False

        # Check if there's recent research context
        if state.get("research_findings"):
            return True

        # Check for contextual references
        query = state.get("user_query", "").lower()
        follow_up_indicators = [
            "their", "they", "it", "its", "the company",
            "that", "those", "these", "this", "them"
        ]

        return any(indicator in query for indicator in follow_up_indicators)

    def _determine_clarity(
        self,
        query: str,
        company_name: Optional[str],
        ticker: Optional[str],
        intent: str,
        state: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """
        Determine if query is clear enough to proceed.

        Args:
            query: User query
            company_name: Detected company
            ticker: Detected ticker
            intent: Classified intent
            state: Current state

        Returns:
            Tuple of (clarity_status, clarification_request)
        """
        is_follow_up = self._is_follow_up(state)

        # Case 1: Clear query with company identified
        if company_name:
            return "clear", None

        # Case 2: Follow-up with previous context
        if is_follow_up and state.get("detected_company"):
            return "clear", None

        # Case 3: No company identified
        clarification = self._generate_clarification_request("company_missing", query, intent)
        return "needs_clarification", clarification

    def _generate_clarification_request(
        self,
        reason: str,
        query: str,
        intent: str
    ) -> str:
        """
        Generate a helpful clarification request.

        Args:
            reason: Why clarification is needed
            query: Original query
            intent: Detected intent

        Returns:
            Clarification question for user
        """
        if reason == "company_missing":
            examples = "For example: 'Tell me about Apple' or 'Who is the CEO of Tesla?'"
            return (
                f"I'd be happy to help with your research! Could you please specify "
                f"which company you're interested in? {examples}"
            )

        if reason == "too_vague":
            intent_suggestions = {
                "stock_price": "current stock price and performance",
                "financial_analysis": "financial metrics and earnings",
                "news_developments": "recent news and developments",
                "company_overview": "company overview and business model",
                "competitor_analysis": "competitive landscape",
                "investment_research": "investment analysis",
                "leadership": "CEO and leadership information",
            }

            suggestion = intent_suggestions.get(
                intent, "specific information about the company"
            )

            return (
                f"Your query seems a bit general. Could you be more specific? "
                f"For example, I can help you with {suggestion}. "
                f"Please provide the company name and what you'd like to know."
            )

        return (
            "Could you please provide more details about what you'd like to research? "
            "Include the company name or ticker symbol and what information you need."
        )

    def _handle_validation_failure(
        self,
        state: Dict[str, Any],
        validation_result: GuardrailResult
    ) -> Dict[str, Any]:
        """
        Handle query that failed guardrail validation.

        Args:
            state: Current state
            validation_result: Guardrail result

        Returns:
            State updates for validation failure
        """
        self._log_execution("Query validation failed", {
            "violation": validation_result.violation_type.value if validation_result.violation_type else "unknown"
        })

        # Add audit entry
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "event": "validation_failure",
            "violation_type": validation_result.violation_type.value if validation_result.violation_type else None,
            "message": validation_result.violation_message
        }

        return {
            "clarity_status": "needs_clarification",
            "clarification_request": validation_result.violation_message,
            "current_agent": self.name,
            "error_message": validation_result.violation_message,
            "awaiting_human_input": True,
            "audit_log": state.get("audit_log", []) + [audit_entry],
            "messages": [Message(
                role="assistant",
                content=f"[Clarity Agent] Validation issue: {validation_result.violation_message}",
                agent=self.name
            )]
        }

    def _use_llm_for_analysis(
        self,
        query: str,
        messages: List,
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM for intelligent query analysis.

        This is the primary analysis method for accurate intent classification.

        Args:
            query: User query
            messages: Conversation history
            state: Current state

        Returns:
            LLM analysis result or None
        """
        try:
            # Build context from messages
            context = self._build_context_from_messages(messages, max_messages=5)

            # Add previous company context
            previous_company = state.get("detected_company")
            if previous_company:
                context += f"\n\nPreviously discussed company: {previous_company}"

            prompt = self._create_prompt(
                "Analyze this user query about company research:\n\n"
                "Query: \"{query}\"\n\n"
                "Conversation context:\n{context}\n\n"
                "Classify the intent accurately. Pay special attention to:\n"
                "- Questions about owner/CEO/founder/executives -> intent: leadership\n"
                "- Questions about stock price/trading -> intent: stock_price\n"
                "- General 'tell me about' questions -> intent: company_overview\n\n"
                "Respond with JSON only."
            )

            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "context": context
            })

            result = self._parse_json_response(response.content)

            if result:
                self._log_execution("LLM analysis result", {
                    "intent": result.get("query_intent"),
                    "company": result.get("detected_company")
                })

            return result

        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}")
            return None
