"""
Base agent class for the Research Assistant.

Provides common functionality for all specialized agents including
LLM interaction, logging, and state management.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from ..config import settings


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Research Assistant.

    Provides common functionality for LLM interaction, logging,
    and state management. All specialized agents should inherit
    from this class.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize the base agent.

        Args:
            model_name: LLM model to use (defaults to settings.default_model)
            temperature: Temperature for LLM responses (defaults to settings.temperature)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self._llm: Optional[ChatAnthropic] = None

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy-loaded LLM instance."""
        if self._llm is None:
            if not settings.validate_api_key():
                raise ValueError(
                    "ANTHROPIC_API_KEY not configured. "
                    "Please set it in .env file or environment variables."
                )
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.anthropic_api_key,
                max_tokens=4096,
            )
            self.logger.debug(f"Initialized LLM: {self.model_name}")
        return self._llm

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this agent."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's logic.

        Args:
            state: Current workflow state as dictionary

        Returns:
            Dictionary of state updates
        """
        pass

    def _create_prompt(self, human_template: str) -> ChatPromptTemplate:
        """
        Create a chat prompt template with system and human messages.

        Args:
            human_template: Template string for the human message

        Returns:
            ChatPromptTemplate instance
        """
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", human_template)
        ])

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM, handling markdown code blocks.

        Args:
            content: Raw response content from LLM

        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw content: {content}")
            return {}

    def _log_execution(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log agent execution details.

        Args:
            action: Description of the action being performed
            details: Additional details to log
        """
        msg = f"{self.name} - {action}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def _build_context_from_messages(
        self,
        messages: list,
        max_messages: int = 10
    ) -> str:
        """
        Build context string from message history.

        Args:
            messages: List of Message objects or dicts
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted context string
        """
        if not messages:
            return "No previous conversation."

        recent = messages[-max_messages:]
        context_parts = []

        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")

            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."

            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)
