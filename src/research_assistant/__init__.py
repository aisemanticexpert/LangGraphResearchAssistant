"""
LangGraph Multi-Agent Research Assistant

A production-grade multi-agent system for company research using LangGraph.
Features 4 specialized agents: Clarity, Research, Validator, and Synthesis.
"""

__version__ = "1.0.0"
__author__ = "Research Assistant Team"

from .app import ResearchAssistantApp
from .state import ResearchAssistantState

__all__ = ["ResearchAssistantApp", "ResearchAssistantState", "__version__"]
