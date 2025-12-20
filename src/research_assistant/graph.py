"""
LangGraph workflow definition for the Research Assistant.

This module defines the complete multi-agent workflow using LangGraph,
including nodes, edges, conditional routing, and human-in-the-loop support.
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


# Define state schema as TypedDict for LangGraph compatibility
def add_messages(left: List, right: List) -> List:
    """Reducer to append messages."""
    return left + right


class GraphState(TypedDict, total=False):
    """State schema for the LangGraph workflow."""
    # User Input
    user_query: str
    messages: Annotated[List, add_messages]

    # Clarity Agent Outputs
    clarity_status: str
    clarification_request: str
    detected_company: str

    # Research Agent Outputs
    research_findings: Any
    confidence_score: float

    # Validator Agent Outputs
    validation_result: str
    validation_feedback: str

    # Loop Control
    research_attempts: int

    # Synthesis Output
    final_response: str

    # Workflow Control
    current_agent: str
    error_message: str

    # Human-in-the-Loop
    awaiting_human_input: bool
    human_response: str


def human_clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Human-in-the-loop interrupt node.

    This node pauses the workflow execution and waits for human input
    when the Clarity Agent determines the query needs clarification.

    Args:
        state: Current workflow state

    Returns:
        State updates with human clarification
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
    Construct the LangGraph workflow for the Research Assistant.

    The workflow implements:
    - 4 specialized agents (Clarity, Research, Validator, Synthesis)
    - Conditional routing based on agent outputs
    - Human-in-the-loop for unclear queries
    - Feedback loop for research retry (max 3 attempts)

    Graph Flow:
    ```
    START -> clarity
             |
             ├─[needs_clarification]─> human_clarification -> clarity
             └─[clear]─> research
                         |
                         ├─[confidence < 6]─> validator
                         │                     |
                         │   ├─[insufficient & attempts < 3]─> research
                         │   └─[sufficient OR max attempts]─> synthesis
                         └─[confidence >= 6]─> synthesis -> END
    ```

    Args:
        checkpointer: Optional checkpointer for state persistence.
                     Defaults to MemorySaver if not provided.

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Building research assistant graph")

    # Initialize agents
    clarity_agent = ClarityAgent()
    research_agent = ResearchAgent()
    validator_agent = ValidatorAgent()
    synthesis_agent = SynthesisAgent()

    # Create StateGraph with our state schema
    workflow = StateGraph(GraphState)

    # === Add Nodes ===
    workflow.add_node("clarity", clarity_agent.run)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("research", research_agent.run)
    workflow.add_node("validator", validator_agent.run)
    workflow.add_node("synthesis", synthesis_agent.run)

    # === Add Entry Edge ===
    workflow.add_edge(START, "clarity")

    # === Conditional Edge: After Clarity ===
    # Routes to human_clarification if query is unclear, else to research
    workflow.add_conditional_edges(
        "clarity",
        route_after_clarity,
        {
            "human_clarification": "human_clarification",
            "research": "research"
        }
    )

    # === Edge: After Human Clarification ===
    # Returns to clarity agent to re-evaluate the clarified query
    workflow.add_edge("human_clarification", "clarity")

    # === Conditional Edge: After Research ===
    # Routes to validator if low confidence, else directly to synthesis
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "validator": "validator",
            "synthesis": "synthesis"
        }
    )

    # === Conditional Edge: After Validation ===
    # Routes back to research if insufficient (retry loop), else to synthesis
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "research": "research",  # Retry loop
            "synthesis": "synthesis"
        }
    )

    # === Terminal Edge ===
    workflow.add_edge("synthesis", END)

    # Use provided checkpointer or create default
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile the graph with checkpointer for persistence
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Research assistant graph built successfully")
    return graph


def get_graph_visualization() -> str:
    """
    Generate a Mermaid diagram of the workflow.

    Returns:
        Mermaid diagram string for visualization
    """
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
