"""
State schema definitions for the Research Assistant LangGraph workflow.

This module defines the Pydantic models that represent the state passed
between agents in the multi-agent workflow.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def add_messages(left: List["Message"], right: List["Message"]) -> List["Message"]:
    """Reducer function to append messages to conversation history."""
    return left + right


class Message(BaseModel):
    """Individual message in conversation history."""

    role: Literal["user", "assistant", "system"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(
        description="The content of the message"
    )
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO format timestamp of the message"
    )
    agent: Optional[str] = Field(
        default=None,
        description="Name of the agent that generated this message"
    )


class ResearchFindings(BaseModel):
    """Structured research findings from the Research Agent."""

    company_name: Optional[str] = Field(
        default=None,
        description="Name of the company researched"
    )
    recent_news: Optional[str] = Field(
        default=None,
        description="Recent news about the company"
    )
    stock_info: Optional[str] = Field(
        default=None,
        description="Stock and financial information"
    )
    key_developments: Optional[str] = Field(
        default=None,
        description="Key business developments"
    )
    raw_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw research data from the tool"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of information sources"
    )


class ResearchAssistantState(BaseModel):
    """
    Main state schema for the Research Assistant LangGraph workflow.

    This state is passed between all agents and maintains the complete
    context of the conversation and research process. Uses TypedDict-like
    behavior for LangGraph compatibility while providing Pydantic validation.
    """

    # === User Input ===
    user_query: str = Field(
        default="",
        description="Current user query to process"
    )

    # === Conversation History ===
    # Using Annotated with reducer for append-only behavior
    messages: Annotated[List[Message], add_messages] = Field(
        default_factory=list,
        description="Full conversation history (append-only)"
    )

    # === Clarity Agent Outputs ===
    clarity_status: Literal["clear", "needs_clarification", "pending"] = Field(
        default="pending",
        description="Result from Clarity Agent analysis"
    )
    clarification_request: Optional[str] = Field(
        default=None,
        description="Question to ask user if query is unclear"
    )
    detected_company: Optional[str] = Field(
        default=None,
        description="Company name extracted from query"
    )

    # === Research Agent Outputs ===
    research_findings: Optional[ResearchFindings] = Field(
        default=None,
        description="Structured research results"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Research confidence score (0-10)"
    )

    # === Validator Agent Outputs ===
    validation_result: Literal["sufficient", "insufficient", "pending"] = Field(
        default="pending",
        description="Validation status of research findings"
    )
    validation_feedback: Optional[str] = Field(
        default=None,
        description="Feedback for improving research if insufficient"
    )

    # === Loop Control ===
    research_attempts: int = Field(
        default=0,
        ge=0,
        description="Number of research attempts (max 3)"
    )

    # === Synthesis Agent Outputs ===
    final_response: Optional[str] = Field(
        default=None,
        description="Final synthesized response to user"
    )

    # === Workflow Control ===
    current_agent: Optional[str] = Field(
        default=None,
        description="Currently executing agent name"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if any step fails"
    )

    # === Human-in-the-Loop ===
    awaiting_human_input: bool = Field(
        default=False,
        description="Flag indicating if waiting for human input"
    )
    human_response: Optional[str] = Field(
        default=None,
        description="Response received from human clarification"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "allow"  # Allow extra fields for LangGraph compatibility


# Type alias for state dictionary (used in LangGraph nodes)
StateDict = Dict[str, Any]
