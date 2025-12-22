"""
The main workflow - where all the agents get wired together.

Uses LangGraph's StateGraph to connect the 4 agents. Each agent
is a node, routing functions decide where to go next based on
what each agent returns.

Also has error handling so we don't just crash if something breaks.
"""

import logging
import traceback
from typing import Any, Dict, TypedDict, Annotated, List, Literal
from functools import wraps

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from .agents.clarity_agent import ClarityAgent
from .agents.research_agent import ResearchAgent
from .agents.validator_agent import ValidatorAgent
from .agents.synthesis_agent import SynthesisAgent
from .routing.conditions import (
    route_after_clarity,
    route_after_research,
    route_after_validation,
)
from .state import Message

logger = logging.getLogger(__name__)


def add_messages(left: List, right: List) -> List:
    """Just appends new messages to the list"""
    return left + right


class GraphState(TypedDict, total=False):
    """
    Shared state that gets passed between agents.
    Each agent reads what it needs and adds its outputs.
    """
    # What the user asked
    user_query: str
    messages: Annotated[List, add_messages]

    # From Clarity Agent
    clarity_status: str
    clarification_request: str
    detected_company: str

    # From Research Agent
    research_findings: Any
    confidence_score: float
    confidence_breakdown: Dict[str, Any]  # Detailed confidence metrics

    # From Validator
    validation_result: str
    validation_feedback: str

    # Tracking retries
    research_attempts: int

    # Final output
    final_response: str

    # Misc
    current_agent: str
    error_message: str
    awaiting_human_input: bool
    human_response: str

    # Error handling
    has_error: bool
    error_node: str
    error_recoverable: bool

    # Retry effectiveness tracking
    retry_history: List[Dict[str, Any]]


def human_clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pauses and asks the user for clarification.
    Uses LangGraph's interrupt() to halt execution until we get input.
    """
    logger.info("Entering human clarification node")

    clarification_question = state.get(
        "clarification_request",
        "Could you please clarify your question? Which company are you asking about?"
    )

    # Create the interrupt - this pauses execution and waits for input
    human_response = interrupt({
        "type": "clarification_needed",
        "question": clarification_question,
        "original_query": state.get("user_query", ""),
        "instruction": "Please provide clarification to continue."
    })

    logger.info(f"Received human clarification: {human_response}")

    # When resumed, update state with the clarified query
    return {
        "user_query": human_response,
        "human_response": human_response,
        "awaiting_human_input": False,
        "clarity_status": "pending",  # Reset for re-evaluation
        "clarification_request": None,
    }


def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles errors that occur during agent execution.

    Provides graceful degradation by:
    1. Logging the error with full context
    2. Generating a user-friendly error message
    3. Attempting recovery if possible
    4. Routing to synthesis with available data if unrecoverable
    """
    error_message = state.get("error_message", "Unknown error occurred")
    error_node = state.get("error_node", "unknown")
    current_attempts = state.get("research_attempts", 0)

    logger.error(f"Error in node '{error_node}': {error_message}")

    # Determine if we can recover
    recoverable = False
    recovery_action = None

    if error_node == "research" and current_attempts < 3:
        # Can retry research
        recoverable = True
        recovery_action = "Retrying research with adjusted parameters"
        logger.info(f"Attempting recovery: {recovery_action}")

    elif error_node == "clarity":
        # Ask user for help
        recoverable = True
        recovery_action = "Asking user for clarification due to processing error"

    elif error_node == "validator":
        # Skip validation and proceed to synthesis
        recoverable = True
        recovery_action = "Skipping validation and proceeding with available data"

    # Build error response
    if recoverable:
        user_message = f"I encountered an issue while processing your request, but I'm attempting to recover. {recovery_action}"
    else:
        user_message = (
            f"I encountered an error while researching your query. "
            f"I'll provide the best response possible with the information gathered so far."
        )

    return {
        "has_error": True,
        "error_recoverable": recoverable,
        "current_agent": "ErrorHandler",
        "messages": [Message(
            role="assistant",
            content=f"[System] {user_message}",
            agent="ErrorHandler"
        )],
        # If not recoverable, set validation as sufficient to proceed to synthesis
        "validation_result": "sufficient" if not recoverable else state.get("validation_result", "pending"),
    }


def create_safe_node(node_name: str, node_func):
    """
    Wraps a node function with error handling.

    Catches exceptions and routes to error handler instead of crashing.
    """
    @wraps(node_func)
    def safe_node(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return node_func(state)
        except Exception as e:
            logger.error(f"Exception in node '{node_name}': {str(e)}")
            logger.debug(traceback.format_exc())

            return {
                "has_error": True,
                "error_message": str(e),
                "error_node": node_name,
                "current_agent": node_name,
                "messages": [Message(
                    role="assistant",
                    content=f"[{node_name}] Error occurred: {str(e)[:100]}",
                    agent=node_name
                )]
            }

    return safe_node


def route_with_error_check(route_func):
    """
    Wraps a routing function to check for errors first.
    """
    @wraps(route_func)
    def safe_route(state: Dict[str, Any]) -> str:
        if state.get("has_error"):
            return "error_handler"
        return route_func(state)

    return safe_route


def build_research_graph(checkpointer=None, safe_mode: bool = True) -> StateGraph:
    """
    Builds and returns the compiled graph.

    This is where we connect all the pieces:
    - Add each agent as a node
    - Wire up the edges between them
    - Set up the conditional routing
    - Add error handling for graceful degradation

    Args:
        checkpointer: Optional checkpointer for state persistence
        safe_mode: If True, wrap nodes with error handling (default: True)
    """
    logger.info("Building research assistant graph")

    # Create our agents
    clarity_agent = ClarityAgent()
    research_agent = ResearchAgent()
    validator_agent = ValidatorAgent()
    synthesis_agent = SynthesisAgent()

    workflow = StateGraph(GraphState)

    # Wrap nodes with error handling if safe_mode is enabled
    if safe_mode:
        clarity_node = create_safe_node("clarity", clarity_agent.run)
        research_node = create_safe_node("research", research_agent.run)
        validator_node = create_safe_node("validator", validator_agent.run)
        synthesis_node = create_safe_node("synthesis", synthesis_agent.run)
    else:
        clarity_node = clarity_agent.run
        research_node = research_agent.run
        validator_node = validator_agent.run
        synthesis_node = synthesis_agent.run

    # Add all nodes
    workflow.add_node("clarity", clarity_node)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("research", research_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("error_handler", error_handler_node)

    # Start with clarity
    workflow.add_edge(START, "clarity")

    # After clarity: check for errors first, then route normally
    def route_after_clarity_with_error(state: Dict[str, Any]) -> Literal["error_handler", "human_clarification", "research"]:
        if state.get("has_error"):
            return "error_handler"
        return route_after_clarity(state)

    workflow.add_conditional_edges(
        "clarity",
        route_after_clarity_with_error,
        {
            "error_handler": "error_handler",
            "human_clarification": "human_clarification",
            "research": "research"
        }
    )

    # After getting clarification, go back to clarity to re-evaluate
    workflow.add_edge("human_clarification", "clarity")

    # After research: check for errors, then validate or synthesize
    def route_after_research_with_error(state: Dict[str, Any]) -> Literal["error_handler", "validator", "synthesis"]:
        if state.get("has_error"):
            return "error_handler"
        return route_after_research(state)

    workflow.add_conditional_edges(
        "research",
        route_after_research_with_error,
        {
            "error_handler": "error_handler",
            "validator": "validator",
            "synthesis": "synthesis"
        }
    )

    # After validation: check for errors, then retry or synthesize
    def route_after_validation_with_error(state: Dict[str, Any]) -> Literal["error_handler", "research", "synthesis"]:
        if state.get("has_error"):
            return "error_handler"
        return route_after_validation(state)

    workflow.add_conditional_edges(
        "validator",
        route_after_validation_with_error,
        {
            "error_handler": "error_handler",
            "research": "research",
            "synthesis": "synthesis"
        }
    )

    # After error handler: try to recover or go to synthesis
    def route_after_error(state: Dict[str, Any]) -> Literal["clarity", "research", "synthesis"]:
        error_node = state.get("error_node", "")
        recoverable = state.get("error_recoverable", False)

        if not recoverable:
            # Unrecoverable - go to synthesis with available data
            return "synthesis"

        # Attempt recovery based on where error occurred
        if error_node == "clarity":
            return "clarity"  # Will trigger human clarification
        elif error_node == "research":
            return "research"  # Retry research
        elif error_node == "validator":
            return "synthesis"  # Skip validation

        return "synthesis"

    workflow.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "clarity": "clarity",
            "research": "research",
            "synthesis": "synthesis"
        }
    )

    # Synthesis is the end
    workflow.add_edge("synthesis", END)

    # Use in-memory checkpointer by default
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Graph built successfully with error handling enabled")
    return graph


def get_graph_visualization() -> str:
    """Returns a mermaid diagram of the workflow for docs"""
    return """
```mermaid
graph TD
    START((Start)) --> clarity[Clarity Agent]

    clarity -->|needs_clarification| human[Human Clarification]
    clarity -->|clear| research[Research Agent]
    clarity -->|error| error[Error Handler]

    human --> clarity

    research -->|confidence < 6| validator[Validator Agent]
    research -->|confidence >= 6| synthesis[Synthesis Agent]
    research -->|error| error

    validator -->|insufficient & attempts < 3| research
    validator -->|sufficient OR max attempts| synthesis
    validator -->|error| error

    error -->|recoverable| research
    error -->|unrecoverable| synthesis

    synthesis --> END((End))

    style clarity fill:#e1f5fe
    style research fill:#fff3e0
    style validator fill:#fce4ec
    style synthesis fill:#e8f5e9
    style human fill:#fff9c4
    style error fill:#ffcdd2
```
"""
