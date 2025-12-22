"""
Application Class for the Research Assistant
==============================================

Provides a high-level interface for interacting with the multi-agent
research workflow, handling conversations, interrupts, and state management.

This is the main entry point for both CLI and API usage.

Features:
    - Session management with unique thread IDs
    - Multi-turn conversation support
    - Human-in-the-loop interrupt handling
    - State persistence via checkpointer
    - Guardrail configuration
    - Audit logging for compliance

Usage:
    from app import ResearchAssistantApp

    app = ResearchAssistantApp()
    result = app.start_conversation("Tell me about Apple")

    if result["interrupted"]:
        result = app.resume_with_clarification(
            result["thread_id"],
            "I'm interested in their recent products"
        )

    print(result["final_response"])

Author: Rajesh Gupta
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.types import Command

from .graph import build_research_graph
from .state import Message, create_initial_state
from .utils.persistence import get_checkpointer
from .guardrails import AuditLogger, GuardrailConfig


logger = logging.getLogger(__name__)


class ResearchAssistantApp:
    """
    Main application class for the Research Assistant.

    This class provides a clean interface for:
        - Starting new conversations
        - Continuing existing conversations (follow-up questions)
        - Handling human-in-the-loop interrupts
        - Managing conversation threads with persistence
        - Audit logging for compliance

    Attributes:
        graph: The compiled LangGraph workflow
        checkpointer: State persistence backend
        audit_logger: Compliance audit logger
        sessions: Active session tracking

    Example:
        >>> app = ResearchAssistantApp()
        >>> result = app.start_conversation("Tell me about Apple")
        >>> print(result["final_response"])
    """

    def __init__(
        self,
        checkpointer=None,
        guardrail_config: Optional[GuardrailConfig] = None,
        enable_audit_logging: bool = True
    ):
        """
        Initialize the Research Assistant application.

        Args:
            checkpointer: Optional checkpointer for state persistence.
                         Uses configured backend (memory/sqlite) by default.
            guardrail_config: Configuration for input/output guardrails
            enable_audit_logging: Whether to enable compliance audit logging
        """
        self.checkpointer = checkpointer or get_checkpointer()

        # Initialize audit logger
        self.audit_logger = AuditLogger() if enable_audit_logging else None

        # Build the workflow graph with configuration
        self.graph = build_research_graph(
            checkpointer=self.checkpointer,
            guardrail_config=guardrail_config,
            audit_logger=self.audit_logger
        )

        # Session tracking
        self._thread_counter = 0
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("ResearchAssistantApp initialized successfully")

    def _generate_thread_id(self) -> str:
        """
        Generate a unique thread ID for a new conversation.

        Format: thread-{8-char-uuid}-{counter}

        Returns:
            Unique thread identifier string
        """
        self._thread_counter += 1
        return f"thread-{uuid.uuid4().hex[:8]}-{self._thread_counter}"

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID for audit tracking.

        Returns:
            Unique session identifier string
        """
        return f"session-{uuid.uuid4().hex[:12]}"

    def start_conversation(
        self,
        user_query: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation thread with an initial query.

        Creates a new thread, initializes state, and executes the workflow.
        Handles any interrupts (clarification requests) gracefully.

        Args:
            user_query: The user's initial question
            user_id: Optional user identifier for personalization

        Returns:
            Dictionary containing:
                - thread_id: Unique identifier for this conversation
                - session_id: Session ID for audit tracking
                - result: The workflow execution result
                - final_response: The synthesized response (if completed)
                - executive_summary: Brief summary (if available)
                - confidence_score: Research confidence (0-10)
                - interrupted: Whether the workflow was interrupted
                - interrupt_info: Information about the interrupt (if any)
                - error: Error message (if any)

        Example:
            >>> result = app.start_conversation("Tell me about Apple")
            >>> print(result["thread_id"])
            'thread-abc12345-1'
        """
        thread_id = self._generate_thread_id()
        session_id = self._generate_session_id()

        logger.info(f"Starting new conversation: {thread_id}")

        # Track session
        self._active_sessions[thread_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "queries": [user_query]
        }

        config = {"configurable": {"thread_id": thread_id}}

        # Build initial state
        initial_state = {
            "user_query": user_query,
            "original_query": user_query,
            "messages": [Message(role="user", content=user_query)],
            "clarity_status": "pending",
            "validation_result": "pending",
            "research_attempts": 0,
            "confidence_score": 0.0,
            "awaiting_human_input": False,
            "session_id": session_id,
            "user_id": user_id,
            "request_timestamp": datetime.now().isoformat(),
            "workflow_status": "in_progress",
            "audit_log": []
        }

        # Log to audit
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="conversation_started",
                session_id=session_id,
                user_id=user_id,
                details={"query": user_query[:100], "thread_id": thread_id}
            )

        try:
            result = self.graph.invoke(initial_state, config=config)
            return self._process_result(thread_id, session_id, result)

        except Exception as e:
            logger.error(f"Error in conversation: {e}")

            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="conversation_error",
                    session_id=session_id,
                    user_id=user_id,
                    details={"error": str(e), "thread_id": thread_id}
                )

            return {
                "thread_id": thread_id,
                "session_id": session_id,
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

        Preserves context from the previous conversation (detected company,
        previous findings) for follow-up questions.

        Args:
            thread_id: The conversation thread ID
            user_query: The follow-up question

        Returns:
            Same structure as start_conversation()

        Example:
            >>> result = app.continue_conversation(
            ...     "thread-abc12345-1",
            ...     "What about their recent news?"
            ... )
        """
        logger.info(f"Continuing conversation {thread_id}")

        # Get session info
        session_info = self._active_sessions.get(thread_id, {})
        session_id = session_info.get("session_id", self._generate_session_id())
        user_id = session_info.get("user_id")

        # Track query
        if thread_id in self._active_sessions:
            self._active_sessions[thread_id]["queries"].append(user_query)

        config = {"configurable": {"thread_id": thread_id}}

        # Get current state to preserve context
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
            "executive_summary": None,
            "awaiting_human_input": False,
            "workflow_status": "in_progress",
        }

        # Preserve context from previous query
        if current_values.get("detected_company"):
            updates["detected_company"] = current_values["detected_company"]
        if current_values.get("detected_ticker"):
            updates["detected_ticker"] = current_values["detected_ticker"]
        if current_values.get("original_query"):
            updates["original_query"] = current_values["original_query"]

        # Log to audit
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="conversation_continued",
                session_id=session_id,
                user_id=user_id,
                details={
                    "query": user_query[:100],
                    "thread_id": thread_id,
                    "preserved_company": updates.get("detected_company")
                }
            )

        try:
            result = self.graph.invoke(updates, config=config)
            return self._process_result(thread_id, session_id, result)

        except Exception as e:
            logger.error(f"Error continuing conversation: {e}")
            return {
                "thread_id": thread_id,
                "session_id": session_id,
                "error": str(e),
                "interrupted": False,
            }

    def check_interrupt(self, thread_id: str) -> Dict[str, Any]:
        """
        Check if a conversation thread is in an interrupted state.

        Used to determine if human clarification is needed before
        the workflow can continue.

        Args:
            thread_id: The conversation thread ID

        Returns:
            Dictionary containing:
                - interrupted: Whether the thread is interrupted
                - question: The clarification question (if interrupted)
                - original_query: The original query that needs clarification
                - type: Type of interrupt (e.g., "clarification_needed")

        Example:
            >>> status = app.check_interrupt("thread-abc12345-1")
            >>> if status["interrupted"]:
            ...     print(status["question"])
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
                            "instruction": interrupt_info.value.get("instruction"),
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

        Used after check_interrupt() indicates the workflow is waiting
        for human input.

        Args:
            thread_id: The conversation thread ID
            clarification: The user's clarification response

        Returns:
            Same structure as start_conversation()

        Example:
            >>> result = app.resume_with_clarification(
            ...     "thread-abc12345-1",
            ...     "I meant Apple Inc., the technology company"
            ... )
        """
        logger.info(f"Resuming {thread_id} with clarification: {clarification[:50]}...")

        # Get session info
        session_info = self._active_sessions.get(thread_id, {})
        session_id = session_info.get("session_id", "unknown")
        user_id = session_info.get("user_id")

        config = {"configurable": {"thread_id": thread_id}}

        # Log to audit
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="clarification_provided",
                session_id=session_id,
                user_id=user_id,
                details={
                    "clarification": clarification[:100],
                    "thread_id": thread_id
                }
            )

        try:
            # Resume the graph with the clarification as the response
            result = self.graph.invoke(
                Command(resume=clarification),
                config=config
            )
            return self._process_result(thread_id, session_id, result)

        except Exception as e:
            logger.error(f"Error resuming conversation: {e}")
            return {
                "thread_id": thread_id,
                "session_id": session_id,
                "error": str(e),
                "interrupted": False,
            }

    def get_conversation_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a conversation.

        Useful for debugging, displaying status, or exporting conversation data.

        Args:
            thread_id: The conversation thread ID

        Returns:
            Current state dictionary or None if not found

        Example:
            >>> state = app.get_conversation_state("thread-abc12345-1")
            >>> print(f"Company: {state.get('detected_company')}")
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

    def get_session_info(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information for a thread.

        Args:
            thread_id: The conversation thread ID

        Returns:
            Session info dictionary or None
        """
        return self._active_sessions.get(thread_id)

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all active sessions.

        Returns:
            List of session information dictionaries
        """
        return [
            {"thread_id": tid, **info}
            for tid, info in self._active_sessions.items()
        ]

    def export_audit_logs(self, filepath: str) -> None:
        """
        Export audit logs to a file.

        Args:
            filepath: Path to export file
        """
        if self.audit_logger:
            self.audit_logger.export_logs(filepath)
            logger.info(f"Audit logs exported to {filepath}")

    def _process_result(
        self,
        thread_id: str,
        session_id: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process workflow result into standardized response format.

        Args:
            thread_id: Conversation thread ID
            session_id: Session ID for tracking
            result: Raw workflow result

        Returns:
            Standardized response dictionary
        """
        # Check for interrupts
        interrupt_info = self.check_interrupt(thread_id)

        response = {
            "thread_id": thread_id,
            "session_id": session_id,
            "result": result,
            "final_response": result.get("final_response"),
            "executive_summary": result.get("executive_summary"),
            "confidence_score": result.get("confidence_score", 0.0),
            "detected_company": result.get("detected_company"),
            "workflow_status": result.get("workflow_status", "completed"),
            "interrupted": interrupt_info.get("interrupted", False),
        }

        if interrupt_info.get("interrupted"):
            response["interrupt_info"] = interrupt_info
            response["workflow_status"] = "interrupted"

        # Log completion to audit
        if self.audit_logger and not response["interrupted"]:
            self.audit_logger.log_event(
                event_type="conversation_completed",
                session_id=session_id,
                details={
                    "thread_id": thread_id,
                    "confidence_score": response["confidence_score"],
                    "has_response": bool(response["final_response"])
                }
            )

        return response
