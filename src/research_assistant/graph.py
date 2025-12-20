"""
graph.py - The main workflow

This wires up all 4 agents using LangGraph's StateGraph. Each agent becomes
a node, and we connect them with edges. The routing functions decide where
to go based on what each agent returns.

Author: Rajesh Gupta
"""

import logging
from typing import Any, Dict, TypedDict, Annotated, List

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


def build_research_graph(checkpointer=None) -> StateGraph:
    """
    Builds and returns the compiled graph.

    This is where we connect all the pieces:
    - Add each agent as a node
    - Wire up the edges between them
    - Set up the conditional routing
    """
    logger.info("Building research assistant graph")

    # Create our agents
    clarity_agent = ClarityAgent()
    research_agent = ResearchAgent()
    validator_agent = ValidatorAgent()
    synthesis_agent = SynthesisAgent()

    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("clarity", clarity_agent.run)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("research", research_agent.run)
    workflow.add_node("validator", validator_agent.run)
    workflow.add_node("synthesis", synthesis_agent.run)

    # Start with clarity
    workflow.add_edge(START, "clarity")

    # After clarity: either ask for clarification or proceed to research
    workflow.add_conditional_edges(
        "clarity",
        route_after_clarity,
        {
            "human_clarification": "human_clarification",
            "research": "research"
        }
    )

    # After getting clarification, go back to clarity to re-evaluate
    workflow.add_edge("human_clarification", "clarity")

    # After research: validate if low confidence, otherwise synthesize
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "validator": "validator",
            "synthesis": "synthesis"
        }
    )

    # After validation: retry research if needed, otherwise synthesize
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
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
    logger.info("Graph built successfully")
    return graph


def get_graph_visualization() -> str:
    """Returns a mermaid diagram of the workflow for docs"""
    return """
```mermaid
graph TD
    START((Start)) --> clarity[Clarity Agent]

    clarity -->|needs_clarification| human[Human Clarification]
    clarity -->|clear| research[Research Agent]

    human --> clarity

    research -->|confidence < 6| validator[Validator Agent]
    research -->|confidence >= 6| synthesis[Synthesis Agent]

    validator -->|insufficient & attempts < 3| research
    validator -->|sufficient OR max attempts| synthesis

    synthesis --> END((End))

    style clarity fill:#e1f5fe
    style research fill:#fff3e0
    style validator fill:#fce4ec
    style synthesis fill:#e8f5e9
    style human fill:#fff9c4
```
"""
