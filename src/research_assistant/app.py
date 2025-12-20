"""
Application class for the Research Assistant.

Provides a high-level interface for interacting with the multi-agent
workflow, handling conversations, interrupts, and state management.
"""

import logging
import uuid
from typing import Any, Dict, Optional

from langgraph.types import Command

from .graph import build_research_graph
from .state import Message
from .utils.persistence import get_checkpointer


logger = logging.getLogger(__name__)


class ResearchAssistantApp:
    """
    Main application class for the Research Assistant.

    Provides methods for:
    - Starting new conversations
    - Continuing existing conversations (follow-up questions)
    - Handling human-in-the-loop interrupts
    - Managing conversation threads with persistence
    """

    def __init__(self, checkpointer=None):
        """
        Initialize the Research Assistant application.

        Args:
            checkpointer: Optional checkpointer for state persistence.
                         Uses configured backend (memory/sqlite/postgres) by default.
        """
        self.checkpointer = checkpointer or get_checkpointer()
        self.graph = build_research_graph(checkpointer=self.checkpointer)
        self._thread_counter = 0
        logger.info("ResearchAssistantApp initialized")

    def _generate_thread_id(self) -> str:
        """Generate a unique thread ID for a new conversation."""
        self._thread_counter += 1
        return f"thread-{uuid.uuid4().hex[:8]}-{self._thread_counter}"

    def start_conversation(self, user_query: str) -> Dict[str, Any]:
        """
        Start a new conversation thread with an initial query.

        Args:
            user_query: The user's initial question

        Returns:
            Dictionary containing:
            - thread_id: Unique identifier for this conversation
            - result: The workflow execution result
            - final_response: The synthesized response (if completed)
            - interrupted: Whether the workflow was interrupted
            - interrupt_info: Information about the interrupt (if any)
        """
        thread_id = self._generate_thread_id()
        logger.info(f"Starting new conversation: {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "user_query": user_query,
            "messages": [Message(role="user", content=user_query)],
            "clarity_status": "pending",
            "validation_result": "pending",
            "research_attempts": 0,
            "confidence_score": 0.0,
            "awaiting_human_input": False,
        }

        try:
            result = self.graph.invoke(initial_state, config=config)
            return self._process_result(thread_id, result)
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            return {
                "thread_id": thread_id,
                "error": str(e),
                "interrupted": False,
            }

    def continue_conversation(
        self,
        thread_id: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Continue an existing conversation with a follow-up query.

        Args:
            thread_id: The conversation thread ID
            user_query: The follow-up question

        Returns:
            Same structure as start_conversation()
        """
        logger.info(f"Continuing conversation {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}

        # Get current state to check existing context
        try:
            current_state = self.graph.get_state(config)
            current_values = current_state.values if current_state else {}
        except Exception:
            current_values = {}

        # Prepare updates for the follow-up
        updates = {
            "user_query": user_query,
            "messages": [Message(role="user", content=user_query)],
            "clarity_status": "pending",  # Re-evaluate clarity
            "validation_result": "pending",
            "research_attempts": 0,  # Reset for new query
            "confidence_score": 0.0,
            "final_response": None,
            "awaiting_human_input": False,
        }

        # Preserve detected company from context for follow-up questions
        if current_values.get("detected_company"):
            updates["detected_company"] = current_values["detected_company"]

        try:
            result = self.graph.invoke(updates, config=config)
            return self._process_result(thread_id, result)
        except Exception as e:
            logger.error(f"Error continuing conversation: {e}")
            return {
                "thread_id": thread_id,
                "error": str(e),
                "interrupted": False,
            }

    def check_interrupt(self, thread_id: str) -> Dict[str, Any]:
        """
        Check if a conversation thread is in an interrupted state.

        Args:
            thread_id: The conversation thread ID

        Returns:
            Dictionary containing:
            - interrupted: Whether the thread is interrupted
            - question: The clarification question (if interrupted)
            - original_query: The original query that needs clarification
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self.graph.get_state(config)

            if state and state.tasks:
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_info = task.interrupts[0]
                        return {
                            "interrupted": True,
                            "question": interrupt_info.value.get("question"),
                            "original_query": interrupt_info.value.get("original_query"),
                            "type": interrupt_info.value.get("type"),
                        }

            return {"interrupted": False}

        except Exception as e:
            logger.error(f"Error checking interrupt: {e}")
            return {"interrupted": False, "error": str(e)}

    def resume_with_clarification(
        self,
        thread_id: str,
        clarification: str
    ) -> Dict[str, Any]:
        """
        Resume an interrupted conversation with human clarification.

        Args:
            thread_id: The conversation thread ID
            clarification: The user's clarification response

        Returns:
            Same structure as start_conversation()
        """
        logger.info(f"Resuming {thread_id} with clarification")

        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Resume the graph with the clarification as the response
            result = self.graph.invoke(
                Command(resume=clarification),
                config=config
            )
            return self._process_result(thread_id, result)
        except Exception as e:
            logger.error(f"Error resuming conversation: {e}")
            return {
                "thread_id": thread_id,
                "error": str(e),
                "interrupted": False,
            }

    def get_conversation_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a conversation.

        Args:
            thread_id: The conversation thread ID

        Returns:
            Current state dictionary or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self.graph.get_state(config)
            if state:
                return dict(state.values)
            return None
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return None

    def _process_result(
        self,
        thread_id: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process workflow result into standardized response format."""
        # Check for interrupts
        interrupt_info = self.check_interrupt(thread_id)

        response = {
            "thread_id": thread_id,
            "result": result,
            "final_response": result.get("final_response"),
            "interrupted": interrupt_info.get("interrupted", False),
        }

        if interrupt_info.get("interrupted"):
            response["interrupt_info"] = interrupt_info

        return response
