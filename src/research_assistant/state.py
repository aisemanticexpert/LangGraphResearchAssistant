"""
State definitions - the data that flows between agents.

These Pydantic models define what gets passed around in the workflow.
Messages, research findings, status flags - it all lives here.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def add_messages(left: List["Message"], right: List["Message"]) -> List["Message"]:
    """LangGraph reducer - just appends new messages to the list."""
    return left + right


class Message(BaseModel):
    """One message in the conversation."""

    role: Literal["user", "assistant", "system"] = Field(description="who said it")
    content: str = Field(description="what they said")
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="when"
    )
    agent: Optional[str] = Field(default=None, description="which agent sent this")


class ResearchFindings(BaseModel):
    """What the research agent dug up about a company."""

    company_name: Optional[str] = None
    recent_news: Optional[str] = None
    stock_info: Optional[str] = None
    key_developments: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    sources: List[str] = Field(default_factory=list)


class ResearchAssistantState(BaseModel):
    """
    The full state that gets passed through the workflow.
    All the agents read/write to this.
    """

    # what the user asked
    user_query: str = ""

    # conversation history
    messages: Annotated[List[Message], add_messages] = Field(default_factory=list)

    # from clarity agent
    clarity_status: Literal["clear", "needs_clarification", "pending"] = "pending"
    clarification_request: Optional[str] = None
    detected_company: Optional[str] = None

    # from research agent
    research_findings: Optional[ResearchFindings] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=10.0)

    # from validator
    validation_result: Literal["sufficient", "insufficient", "pending"] = "pending"
    validation_feedback: Optional[str] = None

    # retry tracking
    research_attempts: int = Field(default=0, ge=0)

    # final output
    final_response: Optional[str] = None

    # bookkeeping
    current_agent: Optional[str] = None
    error_message: Optional[str] = None

    # human-in-the-loop stuff
    awaiting_human_input: bool = False
    human_response: Optional[str] = None

    class Config:
        validate_assignment = True
        extra = "allow"  # LangGraph might add extra fields


# for type hints
StateDict = Dict[str, Any]
