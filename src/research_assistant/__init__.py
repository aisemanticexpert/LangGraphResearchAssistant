"""
LangGraph Research Assistant - Production-Ready Multi-Agent System
===================================================================

A sophisticated multi-agent system for company research featuring:
- UltraThink Intent Analysis (deep reasoning before action)
- RAGHEAT-inspired confidence scoring
- Comprehensive guardrails (48+ safety patterns)
- Human-in-the-loop capabilities
- Tavily Search API integration

The system uses 5 specialized AI agents working together:
1. UltraThink Intent Agent - Deep intent analysis (ALWAYS FIRST)
2. Research Agent - Data gathering with RAGHEAT scoring
3. Validator Agent - Quality assessment
4. Synthesis Agent - Response generation
5. (Legacy) Clarity Agent - Query understanding

Developer: Rajesh Gupta
Copyright (c) 2024 Rajesh Gupta. All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "Rajesh Gupta"
__email__ = "rajesh.gupta@example.com"
__license__ = "MIT"

from .app import ResearchAssistantApp
from .state import ResearchAssistantState

__all__ = ["ResearchAssistantApp", "ResearchAssistantState", "__version__"]
