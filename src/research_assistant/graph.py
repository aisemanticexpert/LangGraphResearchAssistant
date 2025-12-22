"""
LangGraph Workflow with UltraThink Strategy
=============================================

This module defines the main workflow graph with an UltraThink-first approach.
The UltraThink agent DEEPLY ANALYZES intent BEFORE any action is taken.

UltraThink Workflow Architecture:
    START -> UltraThink (Deep Intent Analysis)
                |
                ├── BLOCKED (manipulation/illegal) -> Human Clarification -> END
                ├── GREETING -> Direct Response -> END
                ├── UNCLEAR -> Human Clarification -> UltraThink
                └── LEGITIMATE -> Research -> Validator -> Synthesis -> END

Key Principles:
    1. THINK FIRST, ACT LATER - Never take action without understanding intent
    2. DEEP REASONING - Chain-of-thought analysis for accurate classification
    3. SAFETY FIRST - Block manipulation/illegal queries before processing
    4. INTENT-AWARE - All downstream agents receive intent context

Features:
    - Human-in-the-loop interrupt for clarification
    - Error handling with graceful degradation
    - State persistence via checkpointer
    - UltraThink safety patterns (48+ patterns)
    - RAGHEAT confidence scoring integration

Developer: Rajesh Gupta
Copyright (c) 2024 Rajesh Gupta. All rights reserved.
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Literal, Optional
from functools import wraps

from typing_extensions import TypedDict
from typing import Annotated, List
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from .agents.ultrathink_intent_agent import UltraThinkIntentAgent
from .agents.research_agent import ResearchAgent
from .agents.validator_agent import ValidatorAgent
from .agents.synthesis_agent import SynthesisAgent
from .routing.conditions import (
    route_after_research,
    route_after_validation,
)
from .state import (
    ResearchAssistantState,
    Message,
    create_initial_state,
    add_messages
)
from .guardrails import AuditLogger, GuardrailConfig


# Define a TypedDict state for LangGraph proper state management
class GraphState(TypedDict, total=False):
    """
    State schema for LangGraph workflow with UltraThink.

    Using TypedDict ensures proper state merging between nodes.
    """
    # Query
    user_query: str
    original_query: str

    # Messages with reducer for accumulation
    messages: Annotated[List[Message], add_messages]

    # UltraThink Intent Analysis outputs
    ultrathink_complete: bool
    intent_category: Optional[str]  # legitimate_research, manipulation, etc.
    ultrathink_reasoning: List[str]
    ultrathink_confidence: float
    should_proceed: bool

    # Clarity/Intent outputs
    clarity_status: str
    clarification_request: Optional[str]
    detected_company: Optional[str]
    detected_ticker: Optional[str]
    query_intent: Optional[str]  # leadership, stock_price, etc.

    # Research Agent outputs
    research_findings: Optional[Any]
    confidence_score: float
    confidence_breakdown: Optional[Dict]
    factor_scores: Dict[str, float]

    # Validator Agent outputs
    validation_result: str
    validation_feedback: Optional[str]
    data_completeness_score: float
    relevance_score: float

    # Retry tracking
    research_attempts: int
    retry_history: List[Dict]

    # Synthesis Agent outputs
    final_response: Optional[str]
    executive_summary: Optional[str]
    detailed_analysis: Optional[str]

    # Workflow metadata
    current_agent: Optional[str]
    workflow_status: str

    # Error handling
    error_message: Optional[str]
    error_traceback: Optional[str]
    has_error: bool
    error_node: Optional[str]
    error_recoverable: bool

    # Human-in-the-loop
    awaiting_human_input: bool
    human_response: Optional[str]

    # Session tracking
    session_id: Optional[str]
    user_id: Optional[str]
    request_timestamp: Optional[str]
    agent_timestamps: Dict[str, str]
    total_processing_time_ms: float

    # Audit
    audit_log: List[Dict]


# Configure logging
logger = logging.getLogger(__name__)


def human_clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Human-in-the-Loop node for query clarification.

    This node pauses workflow execution and waits for human input using
    LangGraph's interrupt() mechanism.

    Args:
        state: Current workflow state

    Returns:
        State updates with clarified query
    """
    logger.info("Entering human clarification node")

    clarification_question = state.get(
        "clarification_request",
        "Could you please clarify your question? Which company are you asking about?"
    )

    # Check if this is a blocked query (manipulation, etc.)
    intent_category = state.get("intent_category", "")
    if intent_category in ["manipulation", "insider_trading", "harmful"]:
        # For blocked queries, show the block message and wait for new input
        human_response = interrupt({
            "type": "query_blocked",
            "question": clarification_question,  # Use "question" for consistency
            "original_query": state.get("user_query", ""),
            "instruction": "Your query was blocked. Please ask a legitimate research question."
        })
    else:
        # Normal clarification request
        human_response = interrupt({
            "type": "clarification_needed",
            "question": clarification_question,
            "original_query": state.get("user_query", ""),
            "instruction": "Please provide clarification to continue the research."
        })

    logger.info(f"Received human clarification: {human_response}")

    # Reset for re-evaluation
    return {
        "user_query": human_response,
        "human_response": human_response,
        "awaiting_human_input": False,
        "ultrathink_complete": False,  # Re-run UltraThink
        "clarity_status": "pending",
        "clarification_request": None,
        "intent_category": None,  # Reset intent
        "should_proceed": None,
        "current_agent": "HumanClarification",
        "messages": [Message(
            role="user",
            content=human_response,
            metadata={"is_clarification": True}
        )]
    }


def greeting_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle greeting/social interactions.

    Args:
        state: Current workflow state

    Returns:
        State updates with greeting response
    """
    logger.info("Handling greeting")

    response = state.get(
        "final_response",
        "Hello! I'm the Research Assistant. I can help you with:\n\n"
        "- Company research and overviews\n"
        "- Stock prices and market data\n"
        "- Financial analysis and earnings\n"
        "- Recent news and developments\n"
        "- Competitor analysis\n"
        "- Leadership and executive information\n\n"
        "What company would you like to research today?"
    )

    return {
        "final_response": response,
        "workflow_status": "completed",
        "current_agent": "GreetingHandler",
        "messages": [Message(
            role="assistant",
            content=response,
            agent="GreetingHandler"
        )]
    }


def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Error handling node for graceful degradation.

    Args:
        state: Current workflow state

    Returns:
        State updates with error handling result
    """
    error_message = state.get("error_message", "Unknown error occurred")
    error_node = state.get("error_node", "unknown")
    current_attempts = state.get("research_attempts", 0)

    logger.error(f"Error in node '{error_node}': {error_message}")

    # Determine recovery strategy
    recoverable = False
    recovery_action = None

    if error_node == "research" and current_attempts < 3:
        recoverable = True
        recovery_action = "Retrying research"
    elif error_node == "ultrathink":
        recoverable = True
        recovery_action = "Asking for clarification"
    elif error_node == "validator":
        recoverable = True
        recovery_action = "Proceeding with available data"

    user_message = (
        f"I encountered an issue. {recovery_action or 'Proceeding with best effort.'}"
    )

    return {
        "has_error": True,
        "error_recoverable": recoverable,
        "current_agent": "ErrorHandler",
        "validation_result": "sufficient" if not recoverable else state.get("validation_result", "pending"),
        "messages": [Message(
            role="assistant",
            content=f"[System] {user_message}",
            agent="ErrorHandler"
        )]
    }


def create_safe_node(node_name: str, node_func):
    """Wrap node functions with error handling."""
    @wraps(node_func)
    def safe_node(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = datetime.now()
            result = node_func(state)

            if "agent_timestamps" not in result:
                result["agent_timestamps"] = state.get("agent_timestamps", {})
            result["agent_timestamps"][node_name] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Exception in node '{node_name}': {str(e)}")
            return {
                "has_error": True,
                "error_message": str(e),
                "error_traceback": traceback.format_exc(),
                "error_node": node_name,
                "current_agent": node_name,
                "messages": [Message(
                    role="assistant",
                    content=f"[{node_name}] Error: {str(e)[:100]}",
                    agent=node_name,
                    metadata={"error": True}
                )]
            }

    return safe_node


def route_after_ultrathink(state: Dict[str, Any]) -> Literal[
    "error_handler", "human_clarification", "greeting", "research"
]:
    """
    Route after UltraThink intent analysis.

    This is the critical routing decision based on deep intent analysis.

    Routing Logic:
        - error -> error_handler
        - greeting -> greeting_response
        - blocked/unclear -> human_clarification
        - legitimate research -> research
    """
    # Check for errors
    if state.get("has_error"):
        return "error_handler"

    # Check intent category
    intent_category = state.get("intent_category", "")

    # Blocked queries (manipulation, insider trading, harmful)
    if intent_category in ["manipulation", "insider_trading", "harmful"]:
        logger.info(f"Query BLOCKED: {intent_category}")
        return "human_clarification"

    # Greeting
    if intent_category == "greeting" or state.get("workflow_status") == "greeting":
        return "greeting"

    # Needs clarification
    if state.get("clarity_status") == "needs_clarification":
        return "human_clarification"

    if state.get("clarity_status") == "blocked":
        return "human_clarification"

    # Check if we should proceed
    if not state.get("should_proceed", True):
        return "human_clarification"

    # Legitimate research - proceed
    return "research"


def route_after_research_with_error(
    state: Dict[str, Any]
) -> Literal["error_handler", "validator", "synthesis"]:
    """Route after research with error checking."""
    if state.get("has_error"):
        return "error_handler"
    return route_after_research(state)


def route_after_validation_with_error(
    state: Dict[str, Any]
) -> Literal["error_handler", "research", "synthesis"]:
    """Route after validation with error checking."""
    if state.get("has_error"):
        return "error_handler"
    return route_after_validation(state)


def route_after_error(
    state: Dict[str, Any]
) -> Literal["ultrathink", "research", "synthesis"]:
    """Route after error based on recovery strategy."""
    error_node = state.get("error_node", "")
    recoverable = state.get("error_recoverable", False)

    if not recoverable:
        return "synthesis"

    if error_node == "ultrathink":
        return "ultrathink"
    elif error_node == "research":
        return "research"

    return "synthesis"


def build_research_graph(
    checkpointer: Optional[Any] = None,
    safe_mode: bool = True,
    guardrail_config: Optional[GuardrailConfig] = None,
    audit_logger: Optional[AuditLogger] = None
) -> StateGraph:
    """
    Build and compile the UltraThink research assistant workflow graph.

    The UltraThink-first architecture ensures:
        1. Deep intent analysis BEFORE any action
        2. Blocked queries never reach research agents
        3. Intent context flows to all downstream agents
        4. Accurate routing based on true user intent

    Args:
        checkpointer: Optional checkpointer for state persistence
        safe_mode: If True, wrap nodes with error handling
        guardrail_config: Configuration for guardrails
        audit_logger: Optional audit logger for compliance

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Building research assistant graph with UltraThink strategy")

    # Create agents
    ultrathink_agent = UltraThinkIntentAgent(
        guardrail_config=guardrail_config,
        audit_logger=audit_logger
    )
    research_agent = ResearchAgent()
    validator_agent = ValidatorAgent()
    synthesis_agent = SynthesisAgent(guardrail_config=guardrail_config)

    # Create the workflow
    workflow = StateGraph(GraphState)

    # Wrap nodes with error handling
    if safe_mode:
        ultrathink_node = create_safe_node("ultrathink", ultrathink_agent.run)
        research_node = create_safe_node("research", research_agent.run)
        validator_node = create_safe_node("validator", validator_agent.run)
        synthesis_node = create_safe_node("synthesis", synthesis_agent.run)
    else:
        ultrathink_node = ultrathink_agent.run
        research_node = research_agent.run
        validator_node = validator_agent.run
        synthesis_node = synthesis_agent.run

    # Add all nodes
    workflow.add_node("ultrathink", ultrathink_node)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("greeting", greeting_response_node)
    workflow.add_node("research", research_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("error_handler", error_handler_node)

    # START -> UltraThink (ALWAYS analyze intent first)
    workflow.add_edge(START, "ultrathink")

    # UltraThink routing (critical decision point)
    workflow.add_conditional_edges(
        "ultrathink",
        route_after_ultrathink,
        {
            "error_handler": "error_handler",
            "human_clarification": "human_clarification",
            "greeting": "greeting",
            "research": "research"
        }
    )

    # Human Clarification -> Back to UltraThink for re-analysis
    workflow.add_edge("human_clarification", "ultrathink")

    # Greeting -> END
    workflow.add_edge("greeting", END)

    # Research routing
    workflow.add_conditional_edges(
        "research",
        route_after_research_with_error,
        {
            "error_handler": "error_handler",
            "validator": "validator",
            "synthesis": "synthesis"
        }
    )

    # Validator routing
    workflow.add_conditional_edges(
        "validator",
        route_after_validation_with_error,
        {
            "error_handler": "error_handler",
            "research": "research",
            "synthesis": "synthesis"
        }
    )

    # Error handler routing
    workflow.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "ultrathink": "ultrathink",
            "research": "research",
            "synthesis": "synthesis"
        }
    )

    # Synthesis -> END
    workflow.add_edge("synthesis", END)

    # Use in-memory checkpointer by default
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile the graph
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Research assistant graph built successfully with UltraThink")

    return graph


def get_graph_visualization() -> str:
    """Generate Mermaid diagram of the UltraThink workflow."""
    return """
```mermaid
graph TD
    START((Start)) --> ultrathink[UltraThink Intent Agent]

    ultrathink -->|BLOCKED: manipulation/illegal| human[Human Clarification]
    ultrathink -->|UNCLEAR: needs clarification| human
    ultrathink -->|GREETING| greeting[Greeting Response]
    ultrathink -->|LEGITIMATE RESEARCH| research[Research Agent]
    ultrathink -->|error| error[Error Handler]

    human --> ultrathink

    greeting --> END((End))

    research -->|confidence < 6| validator[Validator Agent]
    research -->|confidence >= 6| synthesis[Synthesis Agent]
    research -->|error| error

    validator -->|insufficient & attempts < 3| research
    validator -->|sufficient OR max attempts| synthesis
    validator -->|error| error

    error -->|recoverable| ultrathink
    error -->|unrecoverable| synthesis

    synthesis --> END

    style ultrathink fill:#ff9800,stroke:#e65100,color:#fff
    style research fill:#2196f3
    style validator fill:#9c27b0
    style synthesis fill:#4caf50
    style human fill:#ffc107
    style greeting fill:#00bcd4
    style error fill:#f44336
```
"""


def print_workflow_description():
    """Print UltraThink workflow description."""
    description = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║              ULTRATHINK RESEARCH ASSISTANT WORKFLOW                   ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  0. ULTRATHINK INTENT AGENT (First, Always)                          ║
    ║     - Deep chain-of-thought reasoning                                ║
    ║     - Safety analysis (manipulation, insider trading, injection)    ║
    ║     - Intent classification (leadership, stock, news, etc.)         ║
    ║     - Entity extraction (company, ticker)                            ║
    ║     Routing:                                                          ║
    ║       - BLOCKED → Human Clarification (asks for legitimate query)   ║
    ║       - GREETING → Greeting Response → END                           ║
    ║       - UNCLEAR → Human Clarification → Re-analyze                   ║
    ║       - LEGITIMATE → Research Agent                                  ║
    ║                                                                       ║
    ║  1. RESEARCH AGENT                                                    ║
    ║     - Intent-aware research (uses classified intent)                 ║
    ║     - RAGHEAT confidence scoring                                      ║
    ║     Routing:                                                          ║
    ║       - confidence >= 6.0 → Synthesis                                 ║
    ║       - confidence < 6.0 → Validator                                  ║
    ║                                                                       ║
    ║  2. VALIDATOR AGENT                                                   ║
    ║     - Quality assessment                                              ║
    ║     Routing:                                                          ║
    ║       - "sufficient" → Synthesis                                      ║
    ║       - "insufficient" + attempts < 3 → Research (retry)             ║
    ║       - "insufficient" + attempts >= 3 → Synthesis                   ║
    ║                                                                       ║
    ║  3. SYNTHESIS AGENT                                                   ║
    ║     - Intent-aware response generation                               ║
    ║     - Financial disclaimers                                           ║
    ║     → END                                                             ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(description)
