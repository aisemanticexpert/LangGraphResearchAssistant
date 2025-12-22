"""
Base class for agents. Saves me from repeating the same LLM setup
and JSON parsing code in every agent.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ..config import settings


LLM_TIMEOUT = 60  # seconds - bump this up if you're getting timeouts


class BaseAgent(ABC):
    """
    All agents inherit from this. Handles:
    - LLM client setup (lazy init)
    - JSON parsing (with the markdown wrapper nonsense)
    - Logging
    - Timeout handling
    """

    def __init__(self, model_name=None, temperature=None, timeout=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.timeout = timeout or LLM_TIMEOUT
        self._llm = None

    @property
    def llm(self):
        """Lazy init for the LLM client"""
        if self._llm is None:
            if not settings.validate_api_key():
                raise ValueError("ANTHROPIC_API_KEY not set - check your .env")

            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.anthropic_api_key,
                max_tokens=4096,
                timeout=self.timeout,
            )
            self.logger.debug(f"init'd LLM: {self.model_name}")
        return self._llm

    def get_llm_with_structured_output(self, schema: Type[BaseModel], method="json_schema"):
        """For when you want typed outputs instead of raw JSON"""
        return self.llm.with_structured_output(schema, method=method)

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier - used in logs and messages"""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The prompt that tells the LLM what this agent does"""
        pass

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point. Takes state, returns state updates."""
        pass

    def _create_prompt(self, human_template: str) -> ChatPromptTemplate:
        """Combines system prompt with human message template"""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", human_template)
        ])

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Claude likes to wrap JSON in markdown code blocks sometimes,
        so we have to strip those out first.
        """
        try:
            # handle ```json ... ``` wrapper
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            self.logger.warning(f"couldn't parse JSON: {e}")
            self.logger.debug(f"raw content: {content[:200]}")
            return {}

    def _log_execution(self, action: str, details: Optional[Dict] = None):
        """Consistent log format across agents"""
        msg = f"{self.name} - {action}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def _build_context_from_messages(self, messages: list, max_messages: int = 10) -> str:
        """
        Build a context string from recent messages.
        Truncates long messages to keep prompts manageable.
        """
        if not messages:
            return "No previous conversation."

        recent = messages[-max_messages:]
        parts = []

        for msg in recent:
            # handle both dict and object formats
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")

            # truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."

            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def invoke_with_timeout(self, chain, input_data: Dict, timeout: int = None):
        """
        Run a chain with a timeout.

        Uses ThreadPoolExecutor since asyncio doesn't play nice
        with langchain's sync invoke.
        """
        timeout = timeout or self.timeout

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(chain.invoke, input_data)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                self.logger.error(f"timed out after {timeout}s")
                raise TimeoutError(f"LLM call timed out after {timeout}s")

    def invoke_structured(
        self,
        schema: Type[BaseModel],
        prompt: ChatPromptTemplate,
        input_data: Dict,
        timeout: int = None
    ) -> BaseModel:
        """
        Invoke with structured output.

        More reliable than asking for JSON and parsing it.
        """
        structured_llm = self.get_llm_with_structured_output(schema)
        chain = prompt | structured_llm

        try:
            result = self.invoke_with_timeout(chain, input_data, timeout)
            self.logger.debug(f"got structured output: {type(result).__name__}")
            return result
        except Exception as e:
            self.logger.error(f"structured invoke failed: {e}")
            raise
