"""
Research Assistant - Multi-Agent System

Built this to explore LangGraph's capabilities for building agent workflows.
The idea is simple: instead of one big agent trying to do everything, we have
4 smaller agents that each do one thing well and pass work to each other.

Author: Rajesh Gupta
"""

__version__ = "1.0.0"
__author__ = "Rajesh Gupta"

from .app import ResearchAssistantApp
from .state import ResearchAssistantState

__all__ = ["ResearchAssistantApp", "ResearchAssistantState", "__version__"]
