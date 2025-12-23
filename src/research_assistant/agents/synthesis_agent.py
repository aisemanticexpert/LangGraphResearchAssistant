"""
Synthesis agent for generating final user responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..state import Message, ResearchFindings, MarketRegime
from ..guardrails import OutputGuardrails, GuardrailConfig


class SynthesisAgent(BaseAgent):
    """
    Synthesizes research findings into professional responses.

    This is the final agent that creates the user-facing output,
    ensuring responses are clear, accurate, and properly disclaimed.

    Features:
        - Template-based response generation
        - Executive summary + detailed analysis
        - Market regime awareness
        - Confidence-based warnings
        - Financial disclaimer injection
        - Output guardrail integration

    Outputs:
        - final_response: Complete synthesized response
        - executive_summary: Brief overview
        - detailed_analysis: In-depth breakdown
    """

    # Financial disclaimer (required for compliance)
    FINANCIAL_DISCLAIMER = (
        "\n\n---\n\n"
        "**DISCLAIMER:** This information is provided for educational and research "
        "purposes only. It does not constitute financial, investment, or trading advice. "
        "Always consult with a qualified financial advisor before making investment decisions. "
        "Past performance does not guarantee future results."
    )

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        guardrail_config: Optional[GuardrailConfig] = None
    ):
        """
        Initialize the Synthesis Agent.

        Args:
            model_name: LLM model to use
            temperature: LLM temperature
            guardrail_config: Configuration for output validation
        """
        super().__init__(model_name=model_name, temperature=temperature)
        self.output_guardrails = OutputGuardrails(guardrail_config)

    @property
    def name(self) -> str:
        return "SynthesisAgent"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Assistant that ANSWERS USER QUESTIONS directly.

CRITICAL: Read the user's ACTUAL question and answer THAT question, not a generic summary.

EXAMPLES:
- "best price to buy apple" → Discuss valuation, P/E ratios, analyst targets, NOT just current price
- "should I invest in Tesla" → Discuss pros/cons, risks, market outlook, NOT just company overview
- "is Microsoft overvalued" → Analyze valuation metrics, compare to peers
- "what is Apple's stock price" → Give current price and recent changes

RESPONSE BY QUESTION TYPE:

For INVESTMENT/BUY questions ("best price to buy", "should I buy", "good time to invest"):
- Discuss current valuation (P/E, P/S ratios)
- Mention analyst price targets if available
- Discuss recent price trends
- List key risks and opportunities
- NEVER give specific buy/sell advice - explain factors to consider
- Always add strong disclaimer

For VALUATION questions ("overvalued", "undervalued", "fair value"):
- Compare current price to historical averages
- Discuss P/E vs industry peers
- Mention analyst consensus
- Be objective about both bull and bear cases

For STOCK PRICE questions:
- Lead with current price
- Include recent changes
- 52-week range

For COMPANY questions:
- Brief overview
- Recent news
- Key metrics

IMPORTANT - ALWAYS END WITH OUTLOOK SUMMARY:
At the end of EVERY response (except greetings), include a brief outlook section:

**Outlook:** [Bullish/Bearish/Neutral] - [1-2 sentence summary of why]

Use the market regime and sentiment data provided to determine the outlook:
- BULL regime + positive sentiment = Bullish
- BEAR regime + negative sentiment = Bearish
- SIDEWAYS or mixed signals = Neutral

CRITICAL RULES:
1. ANSWER THE ACTUAL QUESTION - don't give a generic company report
2. Be conversational - this is a chat, not a formal report
3. Acknowledge when you don't have specific data
4. For investment questions, discuss factors but NEVER recommend buy/sell
5. Keep responses focused and relevant
6. ALWAYS include the Outlook summary at the end

Write naturally - do NOT output JSON."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize findings into final response.

        This is the main entry point called by LangGraph.

        Steps:
            1. Format research findings
            2. Generate response via LLM
            3. Apply output guardrails
            4. Add disclaimers and warnings

        Args:
            state: Current workflow state

        Returns:
            State updates with final response
        """
        start_time = datetime.now()

        user_query = state.get("user_query", "")
        company = state.get("detected_company", "the company")
        research_findings = state.get("research_findings")
        confidence_score = state.get("confidence_score", 0)
        confidence_breakdown = state.get("confidence_breakdown", {})
        attempts = state.get("research_attempts", 0)
        messages = state.get("messages", [])
        query_intent = state.get("query_intent", "general")

        self._log_execution("Synthesizing response", {
            "company": company,
            "confidence": confidence_score,
            "intent": query_intent
        })

        # Format findings for the prompt
        findings_str = self._format_findings(research_findings, query_intent)

        # Build conversation context
        context = self._build_context_from_messages(messages, max_messages=5)

        # Build limitation/confidence notes
        notes = self._build_confidence_notes(
            confidence_score, confidence_breakdown, attempts
        )

        # Detect market regime for context
        market_regime = self._get_market_regime(research_findings)

        # Generate outlook summary from data
        outlook_summary = self._generate_outlook_summary(research_findings, confidence_score)

        # Generate response via LLM
        prompt = self._create_prompt(
            "Create a professional research response:\n\n"
            "User's Question: {query}\n\n"
            "Company: {company}\n\n"
            "Query Intent: {intent}\n\n"
            "Market Regime: {regime}\n\n"
            "Calculated Outlook: {outlook}\n\n"
            "Research Findings:\n{findings}\n\n"
            "Confidence Score: {confidence}/10\n"
            "Research Attempts: {attempts}/3\n"
            "{notes}\n\n"
            "Conversation Context:\n{context}\n\n"
            "Generate a clear, well-organized response that addresses the user's specific question.\n"
            "IMPORTANT: End your response with:\n**Outlook:** {outlook}"
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "company": company,
            "intent": query_intent,
            "regime": market_regime,
            "outlook": outlook_summary,
            "findings": findings_str,
            "confidence": confidence_score,
            "attempts": attempts,
            "notes": notes,
            "context": context
        })

        raw_response = response.content

        # Apply output guardrails
        data_age = 0.0
        if research_findings and hasattr(research_findings, 'data_freshness_hours'):
            data_age = research_findings.data_freshness_hours

        guardrail_result = self.output_guardrails.validate_response(
            response=raw_response,
            confidence_score=confidence_score,
            data_age_hours=data_age
        )

        # Use enhanced response from guardrails
        final_response = guardrail_result.sanitized_content or raw_response

        # Extract executive summary (first paragraph)
        executive_summary = self._extract_executive_summary(final_response)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Determine data source
        data_source = "mock_data"
        if research_findings:
            # Check sources list (ResearchFindings model)
            if hasattr(research_findings, 'sources') and research_findings.sources:
                sources = research_findings.sources
                if "tavily_api" in sources or "tavily_search" in sources:
                    data_source = "tavily_search"
                elif "mock_data" in sources:
                    data_source = "mock_data"
                else:
                    data_source = sources[0] if sources else "mock_data"
            # Check raw_data for source field
            elif hasattr(research_findings, 'raw_data') and research_findings.raw_data:
                raw_source = research_findings.raw_data.get("source", "")
                if "tavily" in str(raw_source).lower():
                    data_source = "tavily_search"
                else:
                    data_source = raw_source or "mock_data"
            # Dict format
            elif isinstance(research_findings, dict):
                if "sources" in research_findings:
                    sources = research_findings["sources"]
                    if isinstance(sources, list) and sources:
                        if "tavily_api" in sources or "tavily_search" in sources:
                            data_source = "tavily_search"
                        else:
                            data_source = sources[0]
                elif "source" in research_findings:
                    data_source = research_findings.get("source", "mock_data")

        self._log_execution("Response synthesized", {
            "response_length": len(final_response),
            "has_warnings": guardrail_result.metadata.get("enhanced", False),
            "processing_time_ms": round(processing_time, 2),
            "data_source": data_source
        })

        return {
            "final_response": final_response,
            "executive_summary": executive_summary,
            "workflow_status": "completed",
            "current_agent": self.name,
            "total_processing_time_ms": processing_time,
            "data_source": data_source,
            "messages": [Message(
                role="assistant",
                content=final_response,
                agent=self.name,
                metadata={
                    "confidence_score": confidence_score,
                    "company": company,
                    "processing_time_ms": processing_time,
                    "data_source": data_source
                }
            )]
        }

    def _format_findings(
        self,
        findings: Optional[ResearchFindings],
        intent: str
    ) -> str:
        """
        Format research findings for the synthesis prompt.

        Prioritizes data relevant to the user's intent.

        Args:
            findings: Research findings
            intent: Query intent

        Returns:
            Formatted findings string
        """
        if not findings:
            return "No research findings available"

        parts = []

        # Handle Pydantic model
        if hasattr(findings, 'company_name'):
            # Company info
            if findings.company_name:
                parts.append(f"Company: {findings.company_name}")
            if hasattr(findings, 'ticker') and findings.ticker:
                parts.append(f"Ticker: {findings.ticker}")
            if hasattr(findings, 'sector') and findings.sector:
                parts.append(f"Sector: {findings.sector}")

            # Stock info (prioritize if intent is stock_price)
            if hasattr(findings, 'stock_info') and findings.stock_info:
                stock = findings.stock_info
                if hasattr(stock, 'to_display_string'):
                    parts.append(f"\nStock Information:\n{stock.to_display_string()}")
                else:
                    parts.append(f"\nStock Information: {stock}")

            # Financial data (prioritize if intent is financial_analysis)
            if hasattr(findings, 'financials') and findings.financials:
                fin = findings.financials
                fin_parts = []
                if hasattr(fin, 'revenue') and fin.revenue:
                    fin_parts.append(f"Revenue: {fin.revenue}")
                if hasattr(fin, 'net_income') and fin.net_income:
                    fin_parts.append(f"Net Income: {fin.net_income}")
                if hasattr(fin, 'eps') and fin.eps:
                    fin_parts.append(f"EPS: ${fin.eps}")
                if hasattr(fin, 'pe_ratio') and fin.pe_ratio:
                    fin_parts.append(f"P/E Ratio: {fin.pe_ratio}")
                if hasattr(fin, 'profit_margin') and fin.profit_margin:
                    fin_parts.append(f"Profit Margin: {fin.profit_margin}%")

                if fin_parts:
                    parts.append(f"\nFinancials:\n- " + "\n- ".join(fin_parts))

            # News (prioritize if intent is news_developments)
            if hasattr(findings, 'recent_news') and findings.recent_news:
                news_items = findings.recent_news
                if isinstance(news_items, list) and news_items:
                    news_strs = []
                    for item in news_items[:5]:
                        if hasattr(item, 'title'):
                            sentiment = ""
                            if hasattr(item, 'sentiment'):
                                if item.sentiment > 0.6:
                                    sentiment = " (positive)"
                                elif item.sentiment < 0.4:
                                    sentiment = " (negative)"
                            news_strs.append(f"- {item.title}{sentiment}")
                        else:
                            news_strs.append(f"- {str(item)[:100]}")
                    parts.append(f"\nRecent News:\n" + "\n".join(news_strs))

            # Key developments
            if hasattr(findings, 'key_developments') and findings.key_developments:
                devs = findings.key_developments[:5]
                parts.append(f"\nKey Developments:\n- " + "\n- ".join(devs))

            # Leadership info (prioritize if intent is leadership)
            if hasattr(findings, 'factor_data') and findings.factor_data:
                factor = findings.factor_data

                # Add leadership info prominently for leadership intent
                if 'leadership' in factor:
                    leadership = factor['leadership']
                    leader_parts = []
                    if leadership.get('ceo'):
                        leader_parts.append(f"CEO: {leadership['ceo']}")
                    if leadership.get('founder'):
                        leader_parts.append(f"Founder: {leadership['founder']}")
                    if leadership.get('founded'):
                        leader_parts.append(f"Founded: {leadership['founded']}")
                    if leadership.get('headquarters'):
                        leader_parts.append(f"Headquarters: {leadership['headquarters']}")
                    if leadership.get('employees'):
                        leader_parts.append(f"Employees: {leadership['employees']}")

                    if leader_parts:
                        # Put leadership info at the beginning for leadership intent
                        if intent == "leadership":
                            parts.insert(0, "\nLEADERSHIP INFORMATION:\n- " + "\n- ".join(leader_parts))
                        else:
                            parts.append(f"\nLeadership:\n- " + "\n- ".join(leader_parts))

                # Add sentiment summary
                if 'sentiment' in factor:
                    sent = factor['sentiment']
                    news_sent = sent.get('news_sentiment', 0.5)
                    sentiment_label = "positive" if news_sent > 0.6 else "negative" if news_sent < 0.4 else "neutral"
                    parts.append(f"\nOverall Sentiment: {sentiment_label} ({news_sent:.2f})")

        # Handle dict
        elif isinstance(findings, dict):
            for key, value in findings.items():
                if value and key not in ["raw_data", "factor_data", "sources"]:
                    formatted_key = key.replace("_", " ").title()
                    parts.append(f"{formatted_key}: {str(value)[:300]}")

        return "\n".join(parts) if parts else "No structured findings"

    def _build_confidence_notes(
        self,
        confidence_score: float,
        breakdown: Dict[str, Any],
        attempts: int
    ) -> str:
        """
        Build notes about confidence and limitations.

        Args:
            confidence_score: Research confidence
            breakdown: Confidence breakdown
            attempts: Research attempts

        Returns:
            Notes string for prompt
        """
        notes = []

        # Confidence level
        if confidence_score >= 8.0:
            notes.append("Research confidence is HIGH - comprehensive data available.")
        elif confidence_score >= 6.0:
            notes.append("Research confidence is GOOD - most key data available.")
        elif confidence_score >= 4.0:
            notes.append(
                "Research confidence is MODERATE - some data may be limited. "
                "Acknowledge any gaps in your response."
            )
        else:
            notes.append(
                "Research confidence is LOW - significant data gaps. "
                "Be transparent about limitations and focus on available information."
            )

        # Gaps from breakdown
        gaps = breakdown.get("gaps", [])
        if gaps:
            notes.append(f"Known gaps: {', '.join(gaps[:3])}")

        # Max attempts
        if attempts >= 3:
            notes.append(
                "Maximum research attempts reached. "
                "Provide the best response with available data."
            )

        return "\n".join(notes) if notes else ""

    def _get_market_regime(self, findings: Optional[ResearchFindings]) -> str:
        """
        Get market regime from findings.

        Args:
            findings: Research findings

        Returns:
            Market regime string
        """
        if findings and hasattr(findings, 'market_regime'):
            regime = findings.market_regime
            if hasattr(regime, 'value'):
                return regime.value
            return str(regime)

        return "unknown"

    def _generate_outlook_summary(
        self,
        findings: Optional[ResearchFindings],
        confidence_score: float
    ) -> str:
        """
        Generate outlook summary based on sentiment and market regime.

        Args:
            findings: Research findings with sentiment data
            confidence_score: RAGHEAT confidence score

        Returns:
            Outlook summary string (Bullish/Bearish/Neutral with reasoning)
        """
        if not findings:
            return "Neutral - Insufficient data for outlook assessment"

        # Get market regime
        regime = "unknown"
        if hasattr(findings, 'market_regime'):
            regime_val = findings.market_regime
            if hasattr(regime_val, 'value'):
                regime = regime_val.value
            else:
                regime = str(regime_val)

        # Get sentiment from factor data
        sentiment_score = 0.5  # Default neutral
        if hasattr(findings, 'factor_data') and findings.factor_data:
            if 'sentiment' in findings.factor_data:
                sent_data = findings.factor_data['sentiment']
                sentiment_score = sent_data.get('news_sentiment', 0.5)

        # Get news sentiment average if available
        if hasattr(findings, 'recent_news') and findings.recent_news:
            news_sentiments = [
                n.sentiment for n in findings.recent_news
                if hasattr(n, 'sentiment')
            ]
            if news_sentiments:
                sentiment_score = sum(news_sentiments) / len(news_sentiments)

        # Determine outlook
        bullish_signals = 0
        bearish_signals = 0
        reasons = []

        # Check regime
        if regime == "bull":
            bullish_signals += 2
            reasons.append("positive market momentum")
        elif regime == "bear":
            bearish_signals += 2
            reasons.append("negative market momentum")

        # Check sentiment
        if sentiment_score > 0.6:
            bullish_signals += 1
            reasons.append("positive news sentiment")
        elif sentiment_score < 0.4:
            bearish_signals += 1
            reasons.append("negative news sentiment")

        # Check stock performance if available
        if hasattr(findings, 'stock_info') and findings.stock_info:
            stock = findings.stock_info
            if hasattr(stock, 'change_percent') and stock.change_percent is not None:
                if stock.change_percent > 2:
                    bullish_signals += 1
                    reasons.append("strong recent price action")
                elif stock.change_percent < -2:
                    bearish_signals += 1
                    reasons.append("weak recent price action")

        # Generate final outlook
        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            outlook = "Bullish"
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            outlook = "Bearish"
        else:
            outlook = "Neutral"

        # Determine confidence label
        if confidence_score >= 8:
            confidence_label = "High"
        elif confidence_score >= 6:
            confidence_label = "Good"
        elif confidence_score >= 4:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        # Build reason string
        if reasons:
            reason_str = ", ".join(reasons[:3])
        else:
            reason_str = "mixed signals in available data"

        # Return outlook with confidence score
        return (
            f"{outlook} - {reason_str}\n"
            f"**Data Confidence:** {confidence_score:.1f}/10 ({confidence_label})"
        )

    def _extract_executive_summary(self, response: str) -> str:
        """
        Extract executive summary from response.

        Takes the first paragraph or first 2-3 sentences.

        Args:
            response: Full response text

        Returns:
            Executive summary string
        """
        # Split by double newlines to find paragraphs
        paragraphs = response.split("\n\n")

        if paragraphs:
            first_para = paragraphs[0]

            # If it's a header, skip to content
            if first_para.startswith("#") or first_para.startswith("**"):
                if len(paragraphs) > 1:
                    first_para = paragraphs[1]

            # Limit to ~200 characters
            if len(first_para) > 250:
                # Find sentence break
                sentences = first_para.split(". ")
                summary = sentences[0]
                if len(sentences) > 1 and len(summary) < 100:
                    summary += ". " + sentences[1]
                return summary + ("." if not summary.endswith(".") else "")

            return first_para

        return response[:200] + "..." if len(response) > 200 else response
