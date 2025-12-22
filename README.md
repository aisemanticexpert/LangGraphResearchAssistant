# LangGraph Research Assistant

A production-ready multi-agent system for company research built with LangGraph.

**Author:** Rajesh Gupta

## Overview

This system uses 5 specialized AI agents working together to provide accurate company research:

1. **UltraThink Intent Agent** - Deep intent analysis (always runs first)
2. **Research Agent** - Data gathering with confidence scoring
3. **Validator Agent** - Quality assessment
4. **Synthesis Agent** - Response generation
5. **Clarity Agent** - Query understanding (legacy)

### Key Features

- **UltraThink Strategy**: Deep reasoning before action with 48+ safety patterns
- **RAGHEAT Confidence Scoring**: 6-factor weighted quality assessment
- **Comprehensive Guardrails**: Market manipulation, insider trading, prompt injection detection
- **Human-in-the-Loop**: Intelligent interrupts for query clarification
- **Tavily Search API**: Real-time data with mock data fallback

---

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd LangGraphResearchAssistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Run
python -m src.research_assistant.main
```

### Usage

**Interactive Mode:**
```bash
python -m src.research_assistant.main

> Tell me about Tesla
> What's their latest news?
> exit
```

**Single Query:**
```bash
python -m src.research_assistant.main -q "Tell me about Apple"
```

**API Server:**
```bash
python -m src.research_assistant.main --api
# Open http://localhost:8000/docs
```

---

## UltraThink Strategy

UltraThink is a custom intent analysis system that thinks before acting. It solves two critical problems:

### Problem 1: Intent Misclassification

Without UltraThink:
```
User: "Tesla owner"
System: "Tesla Inc. is an electric car company..."  (wrong - user wanted Elon Musk info)
```

With UltraThink:
```
User: "Tesla owner"
UltraThink: Detects "owner" keyword -> classifies as LEADERSHIP intent
System: "Elon Musk is the owner and CEO of Tesla Inc..."  (correct)
```

### Problem 2: Dangerous Query Detection

Without UltraThink:
```
User: "How can I dump moderna"
System: "Here's how to sell Moderna shares..."  (helped with illegal activity)
```

With UltraThink:
```
User: "How can I dump moderna"
UltraThink: Detects "dump" pattern -> BLOCKED as market manipulation
System: "I cannot assist with market manipulation activities."
```

### How It Works

UltraThink runs 4 steps on every query:

1. **Safety Check** - Scans 48+ dangerous patterns (manipulation, insider trading, injection)
2. **Intent Classification** - Determines query type (leadership, stock_price, news, etc.)
3. **Entity Extraction** - Identifies company name and ticker
4. **Routing Decision** - Proceed, block, or request clarification

### Safety Patterns

| Category | Patterns | Example Blocked Query |
|----------|----------|----------------------|
| Market Manipulation | 30+ | "pump and dump", "crash the stock" |
| Insider Trading | 8+ | "trade before announcement" |
| Prompt Injection | 10+ | "ignore previous instructions" |

### Intent Types

| Intent | Trigger Keywords | Response Focus |
|--------|-----------------|----------------|
| `leadership` | owner, CEO, founder | Executive information |
| `stock_price` | stock, price, trading | Current price data |
| `financials` | revenue, earnings, profit | Financial metrics |
| `news` | news, recent, latest | Recent developments |
| `competitors` | competitors, vs, compare | Competitive analysis |

---

## Architecture

```
User Query
    |
    v
+------------------------+
|   ULTRATHINK AGENT     |  <-- Always first
|   - Safety Check       |
|   - Intent Analysis    |
|   - Entity Extraction  |
+------------------------+
    |
    +---> [BLOCKED] ---------> Show Block Message
    |     (manipulation,
    |      insider trading)
    |
    +---> [GREETING] --------> Friendly Response -> END
    |
    +---> [UNCLEAR] ---------> Request Clarification -> Back to UltraThink
    |
    +---> [LEGITIMATE] ------> Research Agent
              |
              v
    +-------------------+
    |  RESEARCH AGENT   |
    |   - Data Gather   |
    |   - RAGHEAT Score |
    +-------------------+
        |
        v [confidence < 6.0]          [confidence >= 6.0]
    +-------------------+                    |
    | VALIDATOR AGENT   |                    |
    |   - Quality Gate  |                    |
    |   - Feedback Loop |----[retry]----+    |
    +-------------------+    (max 3x)   |    |
        | [sufficient]                  |    |
        v                               v    v
    +-------------------+
    | SYNTHESIS AGENT   |
    |   - Response Gen  |
    |   - Disclaimers   |
    +-------------------+
        |
        v
    Final Response
```

---

## RAGHEAT Confidence Scoring

Multi-factor confidence assessment based on weighted taxonomy:

```
confidence = sum(weight_i * factor_i) where sum(weights) = 1.0
```

| Factor | Weight | Description |
|--------|--------|-------------|
| `data_completeness` | 30% | Presence of key data fields |
| `source_diversity` | 20% | Number of independent sources |
| `news_coverage` | 15% | News quantity and sentiment |
| `financial_data` | 15% | Financial metrics completeness |
| `recency` | 10% | Time decay (exponential) |
| `sentiment_consistency` | 10% | Alignment of sentiment signals |

**Routing:**
- Score >= 6.0: Direct to synthesis
- Score < 6.0: Requires validation

---

## Guardrails

### Input Validation

1. Empty/null check
2. Length constraints (3-2000 chars)
3. Content sanitization
4. Prompt injection detection
5. Market manipulation blocking
6. Insider trading detection

### Output Enhancement

- Low confidence warnings (< 3.0)
- Stale data notifications (> 72 hours)
- Financial disclaimer injection

### Company Normalization

50+ company aliases supported:
- "Apple" -> "Apple Inc."
- "AAPL" -> "Apple Inc."
- "Google" -> "Alphabet Inc."

---

## Testing

### Quick Test

```bash
python -m src.research_assistant.main
```

Test these queries:

| Query | Expected Result |
|-------|-----------------|
| `Hello` | Greeting response |
| `Tell me about Apple` | Apple company info |
| `Tesla owner` | Elon Musk information |
| `How can I dump moderna` | BLOCKED - manipulation |
| `pump and dump` | BLOCKED - manipulation |
| `insider information` | BLOCKED - insider trading |

### Run Automated Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src/research_assistant --cov-report=html
```

### Test UltraThink Patterns

```bash
python3 -c "
from src.research_assistant.agents.ultrathink_intent_agent import UltraThinkIntentAgent

agent = UltraThinkIntentAgent()

tests = [
    ('How can I dump moderna', False),
    ('Tesla owner', True),
    ('pump and dump', False),
    ('Apple stock price', True),
]

for query, should_proceed in tests:
    result = agent._check_safety_patterns(query)
    status = 'PASS' if result.should_proceed == should_proceed else 'FAIL'
    print(f'{status}: {query}')
"
```

---

## Configuration

### Environment Variables

```env
# Required
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Optional - Research
TAVILY_API_KEY=tvly-xxxxx
USE_MOCK_DATA=true
MAX_RESEARCH_ATTEMPTS=3
CONFIDENCE_THRESHOLD=6.0

# Optional - Model
DEFAULT_MODEL=claude-sonnet-4-20250514
TEMPERATURE=0.0

# Optional - Persistence
CHECKPOINT_BACKEND=memory
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Start new conversation |
| POST | `/continue` | Continue existing conversation |
| POST | `/clarify` | Resume with clarification |
| GET | `/conversation/{thread_id}` | Get conversation state |
| GET | `/companies` | List available companies |
| GET | `/health` | Health check |

### Example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Apple"}'
```

---

## Project Structure

```
src/research_assistant/
├── agents/
│   ├── base.py                     # Base agent class
│   ├── ultrathink_intent_agent.py  # Intent analysis (800 lines)
│   ├── clarity_agent.py            # Query understanding
│   ├── research_agent.py           # Data gathering
│   ├── validator_agent.py          # Quality gate
│   └── synthesis_agent.py          # Response generation
├── routing/
│   └── conditions.py               # Routing logic
├── tools/
│   ├── research_tool.py            # Tavily + Mock search
│   └── mock_data.py                # 25+ companies
├── utils/
│   ├── confidence.py               # RAGHEAT scoring
│   ├── intent.py                   # Intent classification
│   └── grounding.py                # Hallucination detection
├── state.py                        # Pydantic state schemas
├── guardrails.py                   # Input/output validation
├── graph.py                        # LangGraph workflow
├── app.py                          # Application layer
├── api.py                          # FastAPI endpoints
├── config.py                       # Configuration
└── main.py                         # CLI entry point

tests/
├── test_enhanced_system.py         # Comprehensive tests
├── test_agents.py                  # Agent tests
└── test_routing.py                 # Routing tests
```

---

## Supported Companies

**Technology:** Apple, Microsoft, Google, Amazon, Meta, NVIDIA, AMD, Intel, Salesforce, Oracle, Adobe, Netflix

**Finance:** JPMorgan Chase, Visa, PayPal, Block/Square

**Healthcare:** Pfizer, Johnson & Johnson, UnitedHealth, Moderna

**Retail:** Walmart, Costco, Nike, Starbucks

**Automotive:** Tesla, Toyota, Ford

**Entertainment:** Disney, Spotify

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ANTHROPIC_API_KEY not set` | Add key to `.env` file |
| `No module named 'langgraph'` | Run `pip install -r requirements.txt` |
| Low confidence scores | Company may not be in mock data |
| Clarification loops | Be more specific in query |

---

## Author

**Rajesh Gupta**

This project demonstrates:
- LangGraph multi-agent orchestration
- UltraThink deep intent analysis with 48+ safety patterns
- RAGHEAT confidence scoring methodology
- Production-ready guardrails for SEC/FINRA compliance
- Human-in-the-loop workflow management
