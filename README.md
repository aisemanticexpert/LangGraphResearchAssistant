# LangGraph Multi-Agent Research Assistant

A production-ready multi-agent research system built with LangGraph and Claude. Features 4 specialized AI agents that collaborate to research companies, with intelligent routing, human-in-the-loop clarification, and automatic retry logic.

**Author:** Rajesh Gupta

---

## Table of Contents

1. [Requirements Checklist](#requirements-checklist)
2. [The Four Agents](#the-four-agents)
3. [State Schema](#state-schema)
4. [Conditional Routing](#conditional-routing)
5. [Feedback Loop & Retry Logic](#feedback-loop--retry-logic)
6. [Human-in-the-Loop Interrupt](#human-in-the-loop-interrupt)
7. [Multi-Turn Conversation with Memory](#multi-turn-conversation-with-memory)
8. [Example Conversations](#example-conversations)
9. [How to Run](#how-to-run)
10. [Project Structure](#project-structure)
11. [Assumptions](#assumptions)
12. [Beyond Expected Deliverable](#beyond-expected-deliverable)

---

## Requirements Checklist

| # | Requirement | Status | Location |
|---|-------------|--------|----------|
| 1 | Working LangGraph with 4 agents | ✅ | `src/research_assistant/agents/` |
| 2 | State schema with all required fields | ✅ | `src/research_assistant/state.py` |
| 3 | 3 conditional routing functions | ✅ | `src/research_assistant/routing/conditions.py` |
| 4 | Feedback loop (Validator → Research) with counter | ✅ | `graph.py` lines 155-163, `conditions.py` |
| 5 | Interrupt mechanism for unclear queries | ✅ | `graph.py` lines 71-100 (interrupt()) |
| 6 | Multi-turn conversation with memory | ✅ | `app.py` (checkpointer + thread_id) |
| 7 | 2+ example conversation turns | ✅ | See [Example Conversations](#example-conversations) |
| 8 | Software engineering best practices | ✅ | Classes, modular structure, tests |
| 9 | Instructions in README.md | ✅ | See [How to Run](#how-to-run) |
| 10 | Assumptions documented | ✅ | See [Assumptions](#assumptions) |
| 11 | Beyond Expected features | ✅ | See [Beyond Expected](#beyond-expected-deliverable) |

---

## The Four Agents

| Agent | File | Role | What It Does |
|-------|------|------|--------------|
| **Clarity** | `clarity_agent.py` | Query Analyzer | Analyzes user query, detects company name, determines if clarification needed |
| **Research** | `research_agent.py` | Data Gatherer | Searches for company info using Tavily API or mock data, assigns confidence score |
| **Validator** | `validator_agent.py` | Quality Checker | Validates research quality, provides feedback for improvement |
| **Synthesis** | `synthesis_agent.py` | Response Writer | Creates comprehensive, user-friendly response from research findings |

### Agent Workflow

```
START
  │
  ▼
┌─────────────┐
│  CLARITY    │──── needs_clarification ───▶ HUMAN INTERRUPT
│   AGENT     │                                    │
└─────┬───────┘                                    │
      │ clear                                      │
      ▼                                            │
┌─────────────┐                                    │
│  RESEARCH   │◀───────────────────────────────────┘
│   AGENT     │
└─────┬───────┘
      │
      ▼ confidence < 6.0
┌─────────────┐
│  VALIDATOR  │──── insufficient & attempts < 3 ───▶ RESEARCH (retry)
│   AGENT     │
└─────┬───────┘
      │ sufficient OR max attempts
      ▼
┌─────────────┐
│  SYNTHESIS  │
│   AGENT     │
└─────────────┘
      │
      ▼
    END
```

---

## State Schema

**File:** `src/research_assistant/state.py`

The state schema contains all required fields for the workflow:

```python
class GraphState(TypedDict):
    # User Input
    user_query: str                    # Current query
    messages: List[Message]            # Conversation history (with reducer)

    # Clarity Agent Outputs
    clarity_status: str                # "clear" | "needs_clarification" | "pending"
    clarification_request: str         # Question to ask user
    detected_company: str              # Extracted company name

    # Research Agent Outputs
    research_findings: ResearchFindings  # Structured research data
    confidence_score: float            # 0-10 score

    # Validator Agent Outputs
    validation_result: str             # "sufficient" | "insufficient" | "pending"
    validation_feedback: str           # Improvement suggestions

    # Loop Control
    research_attempts: int             # Counter for retries (max 3)

    # Synthesis Output
    final_response: str                # Final answer to user

    # Workflow Control
    current_agent: str                 # Currently executing agent
    error_message: str                 # Error details
    awaiting_human_input: bool         # Human-in-the-loop flag
    human_response: str                # Clarification from user
```

---

## Conditional Routing

**File:** `src/research_assistant/routing/conditions.py`

### 3 Routing Functions:

**1. `route_after_clarity()`** - Routes based on query clarity:
```python
def route_after_clarity(state) -> Literal["human_clarification", "research"]:
    if state.get("clarity_status") == "needs_clarification":
        return "human_clarification"
    return "research"
```

**2. `route_after_research()`** - Routes based on confidence score:
```python
def route_after_research(state) -> Literal["validator", "synthesis"]:
    confidence = state.get("confidence_score", 0.0)
    if confidence < 6.0:  # CONFIDENCE_THRESHOLD
        return "validator"
    return "synthesis"
```

**3. `route_after_validation()`** - Routes based on validation + attempt count:
```python
def route_after_validation(state) -> Literal["research", "synthesis"]:
    result = state.get("validation_result", "pending")
    attempts = state.get("research_attempts", 0)
    if result == "insufficient" and attempts < 3:
        return "research"  # Retry
    return "synthesis"
```

---

## Feedback Loop & Retry Logic

**Location:** `graph.py` lines 155-163

The feedback loop is implemented between Validator and Research agents:

```python
# After validation: retry research if needed, otherwise synthesize
workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "research": "research",    # Loop back for retry
        "synthesis": "synthesis"   # Continue to synthesis
    }
)
```

**Attempt Counter:** The `research_attempts` field in state tracks retries:
- Incremented each time Research agent runs
- Maximum 3 attempts before forcing synthesis
- Ensures workflow eventually completes

---

## Human-in-the-Loop Interrupt

**Location:** `graph.py` lines 71-100

Uses LangGraph's `interrupt()` function to pause execution:

```python
def human_clarification_node(state):
    # Create interrupt - pauses workflow
    human_response = interrupt({
        "type": "clarification_needed",
        "question": state.get("clarification_request"),
        "original_query": state.get("user_query"),
    })

    # When resumed, update state with clarified query
    return {
        "user_query": human_response,
        "clarity_status": "pending",  # Re-evaluate
    }
```

**Usage:** When Clarity Agent returns `needs_clarification`, the workflow pauses and waits for user input via `resume_with_clarification()`.

---

## Multi-Turn Conversation with Memory

**Location:** `src/research_assistant/app.py`

Memory is implemented via LangGraph's checkpointing system:

```python
class ResearchAssistantApp:
    def __init__(self):
        self.checkpointer = get_checkpointer()  # MemorySaver or SqliteSaver
        self.graph = build_research_graph(checkpointer=self.checkpointer)

    def start_conversation(self, query):
        thread_id = self._generate_thread_id()
        config = {"configurable": {"thread_id": thread_id}}
        # State persists across turns via thread_id

    def continue_conversation(self, thread_id, query):
        # Retrieves previous state, preserves context
        current_state = self.graph.get_state(config)
        # Detected company carried forward for follow-ups
```

**Features:**
- Each conversation has unique `thread_id`
- State persisted between turns
- Company context preserved for follow-up questions
- Messages accumulate via reducer function

---

## Example Conversations

### Example 1: Basic Query with Follow-up (Multi-Turn)

```
Turn 1:
User: "Tell me about Apple's latest products"

[Clarity Agent] → Query is clear, company: Apple Inc.
[Research Agent] → Confidence: 8.5/10, skips validator
[Synthesis Agent] → Generates response

Response: "Apple has launched Vision Pro, expanding services revenue.
Apple Intelligence features rolling out across iOS 18, macOS Sequoia..."

Turn 2 (Follow-up - same thread_id):
User: "What about their competitors?"

[Clarity Agent] → Clear, context: Apple Inc. (preserved from Turn 1)
[Research Agent] → Finds competitor info, Confidence: 7.0/10
[Synthesis Agent] → Generates competitor analysis

Response: "Apple's main competitors include Microsoft, Google, and Samsung..."
```

### Example 2: Low Confidence with Retry Loop

```
User: "Tell me about Acme Corp"

[Clarity Agent] → Query is clear, company: Acme Corp
[Research Agent] → Attempt 1, Confidence: 3.0/10 (not in mock data)
[Validator Agent] → "insufficient" - needs more data
[Research Agent] → Attempt 2, Confidence: 3.5/10
[Validator Agent] → "insufficient" - still not enough
[Research Agent] → Attempt 3 (max reached), Confidence: 4.0/10
[Synthesis Agent] → Generates response with available data

Response: "Limited information found for Acme Corp. Based on available data..."
```

### Example 3: Human-in-the-Loop Interrupt

```
User: "What's the stock price?"

[Clarity Agent] → "needs_clarification" - no company specified
[INTERRUPT] → Workflow pauses

System asks: "Which company's stock price are you asking about?"

User provides clarification: "Tesla"

[Clarity Agent] → Now clear, company: Tesla
[Research Agent] → Confidence: 9.0/10
[Synthesis Agent] → Generates response

Response: "Tesla is trading at $242, volatile quarter..."
```

---

## How to Run

### Prerequisites

- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))

### Installation

```bash
# Clone repository
git clone https://github.com/aisemanticexpert/LangGraphResearchAssistant.git
cd LangGraphResearchAssistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running the Assistant

```bash
# Interactive chat mode (recommended for testing)
python -m src.research_assistant.main

# Single query mode
python -m src.research_assistant.main -q "Tell me about NVIDIA's AI chips"

# Start REST API server
python -m src.research_assistant.main --api

# Debug mode with verbose logging
python -m src.research_assistant.main -v
```

### Docker

```bash
# Build and run with docker-compose
docker-compose up research-api

# Run single query in Docker
docker-compose run --rm research-cli python -m src.research_assistant.main -q "Tell me about Tesla"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/research_assistant --cov-report=html
```

---

## Project Structure

```
LangGraphResearchAssistant/
├── src/research_assistant/
│   ├── agents/                 # The 4 agent implementations
│   │   ├── base.py             # Abstract base class
│   │   ├── clarity_agent.py    # Query analysis
│   │   ├── research_agent.py   # Data gathering
│   │   ├── validator_agent.py  # Quality validation
│   │   └── synthesis_agent.py  # Response generation
│   ├── routing/
│   │   └── conditions.py       # 3 routing functions
│   ├── tools/
│   │   ├── research_tool.py    # Tavily/mock data tool
│   │   └── mock_data.py        # 28 companies database
│   ├── utils/
│   │   ├── logging.py          # Rotating file logs
│   │   ├── cache.py            # LRU query caching
│   │   ├── export.py           # JSON/Markdown export
│   │   └── persistence.py      # SQLite checkpointing
│   ├── api.py                  # FastAPI REST server
│   ├── app.py                  # Main application class
│   ├── config.py               # Pydantic settings
│   ├── graph.py                # LangGraph workflow definition
│   ├── main.py                 # CLI entry point
│   └── state.py                # State schema definitions
├── tests/                      # Comprehensive test suite
│   ├── test_api.py
│   ├── test_cache.py
│   ├── test_export.py
│   ├── test_mock_data.py
│   ├── test_routing.py
│   ├── test_state.py
│   └── test_tools.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Assumptions

The following assumptions were made during implementation:

1. **Mock Data for Testing:** By default, the system uses built-in mock data for 28 companies to allow testing without external API calls. Set `USE_MOCK_DATA=false` and provide `TAVILY_API_KEY` for real-time search.

2. **Confidence Threshold:** A confidence score of 6.0/10 is used as the threshold for skipping validation. Below this, research goes through the Validator.

3. **Maximum Retries:** The feedback loop allows maximum 3 research attempts to prevent infinite loops while giving multiple chances to improve results.

4. **Company-Focused Research:** The system is designed for company research queries. General questions may trigger clarification requests.

5. **Single Company Per Query:** Each query is expected to be about one company. Multi-company comparisons work but focus on the first detected company.

6. **Claude Model:** Uses Claude claude-sonnet-4-20250514 by default. Can be configured via `DEFAULT_MODEL` in `.env`.

7. **In-Memory Checkpointing:** Default uses `MemorySaver` for simplicity. For persistence across restarts, set `CHECKPOINT_BACKEND=sqlite`.

---

## Beyond Expected Deliverable

This implementation goes significantly beyond the basic requirements:

### 1. Production-Ready REST API
- **FastAPI server** with 11 endpoints
- Automatic **OpenAPI/Swagger documentation** at `/docs`
- **CORS support** for frontend integration
- Request/response **Pydantic models** for validation

### 2. Query Caching System
- **LRU cache** with configurable TTL (time-to-live)
- Reduces API costs by avoiding duplicate queries
- Cache statistics endpoint (`/cache/stats`)
- Configurable max size and expiration

### 3. Conversation Export
- Export conversations to **JSON** or **Markdown**
- Full conversation history with metadata
- Research findings included
- Useful for archival and sharing

### 4. SQLite Persistence
- Optional **SQLite checkpointing** for state recovery
- Conversations survive application restarts
- Configurable via `CHECKPOINT_BACKEND=sqlite`

### 5. Comprehensive Logging
- **Rotating file logs** with configurable size limits
- Backup count for log rotation
- Separate console and file handlers
- Debug mode for troubleshooting

### 6. Extended Mock Database
- **28 companies** across 6 industries:
  - Technology: Apple, Microsoft, Google, Amazon, Meta, NVIDIA, AMD, Intel, Salesforce, Oracle, Adobe
  - Finance: JPMorgan Chase, Visa, PayPal, Square
  - Healthcare: Pfizer, Johnson & Johnson, UnitedHealth
  - Retail: Walmart, Costco, Nike, Starbucks
  - Automotive: Tesla, Toyota, Ford
  - Entertainment: Netflix, Disney, Spotify
- Company aliases (ticker symbols, common names)
- Rich data: news, stock info, developments, competitors, CEO, headquarters

### 7. Docker Support
- Production-ready **Dockerfile** with multi-stage build
- **docker-compose.yml** with multiple services:
  - `research-api`: REST API server
  - `research-cli`: Interactive CLI
  - `research-dev`: Development with hot-reload
- Health checks and restart policies

### 8. Comprehensive Test Suite
- **11 test files** covering:
  - API endpoints
  - Caching functionality
  - Export features
  - Mock data completeness
  - Routing logic
  - State validation
  - Tool functionality
- Pytest fixtures for mocking

### 9. Software Engineering Best Practices
- **Modular architecture** with clear separation of concerns
- **Abstract base class** for agents
- **Pydantic models** for configuration and state
- **Type hints** throughout
- **Docstrings** for all public methods
- **Error handling** with logging
- **Environment-based configuration**

### 10. Interactive CLI Features
- Commands: `new`, `state`, `export`, `cache`, `companies`, `help`, `quit`
- Real-time conversation state display
- Cache statistics
- Company listing

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `TAVILY_API_KEY` | (optional) | For real-time search |
| `DEFAULT_MODEL` | claude-sonnet-4-20250514 | Claude model to use |
| `USE_MOCK_DATA` | true | Use mock data vs Tavily |
| `MAX_RESEARCH_ATTEMPTS` | 3 | Max retry attempts |
| `CONFIDENCE_THRESHOLD` | 6.0 | Min confidence to skip validator |
| `CHECKPOINT_BACKEND` | memory | memory, sqlite, or postgres |
| `ENABLE_CACHE` | true | Enable query caching |
| `CACHE_TTL_SECONDS` | 3600 | Cache expiration time |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `LOG_TO_FILE` | true | Enable file logging |

---

## License

MIT

---

Built with LangGraph, LangChain, and Claude by Rajesh Gupta