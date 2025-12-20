"""
base.py - Parent class for all agents

Handles the boring stuff every agent needs: LLM setup, JSON parsing, etc.
Each specific agent just implements run() and its prompts.

Author: Rajesh Gupta
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
    Parent class for our agents. Handles LLM setup, logging, JSON parsing.
    Subclasses just need to define their name, system_prompt, and run().
    """

    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self._llm: Optional[ChatAnthropic] = None

    @property
    def llm(self) -> ChatAnthropic:
        """Creates the LLM client on first use"""
        if self._llm is None:
            if not settings.validate_api_key():
                raise ValueError("ANTHROPIC_API_KEY not set - check your .env file")
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.anthropic_api_key,
                max_tokens=4096,
            )
            self.logger.debug(f"Using model: {self.model_name}")
        return self._llm

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for logging"""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The system prompt that defines this agent's personality"""
        pass

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main logic - takes state, returns updates to state"""
        pass

    def _create_prompt(self, human_template: str) -> ChatPromptTemplate:
        """Builds a prompt from system + human message templates"""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", human_template)
        ])

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Extracts JSON from LLM response. Handles the annoying case where
        the LLM wraps it in markdown code blocks.
        """
        try:
            # Strip markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse failed: {e}")
            self.logger.debug(f"Raw: {content}")
            return {}

    def _log_execution(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Simple helper for consistent logging"""
        msg = f"{self.name} - {action}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def _build_context_from_messages(self, messages: list, max_messages: int = 10) -> str:
        """Formats recent messages into a string for context"""
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
