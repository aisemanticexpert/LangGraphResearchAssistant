# LangGraph Multi-Agent Research Assistant

A production-ready multi-agent research system built with LangGraph and Claude. Features 4 specialized AI agents that collaborate to research companies, with intelligent routing, human-in-the-loop clarification, and automatic retry logic.

**Author:** Rajesh Gupta

## Key Features

- **Multi-Agent Architecture** - 4 specialized agents working together
- **Human-in-the-Loop** - Asks for clarification when queries are unclear
- **Smart Retry Logic** - Automatically retries research up to 3 times if confidence is low
- **REST API** - FastAPI server for integration with other applications
- **Query Caching** - Avoid redundant API calls with intelligent caching
- **Conversation Export** - Export chats to JSON or Markdown
- **SQLite Persistence** - Save conversation state across sessions
- **25+ Companies** - Built-in mock data for major companies across industries
- **Comprehensive Logging** - File-based rotating logs for debugging

## The Four Agents

| Agent | Role | What It Does |
|-------|------|--------------|
| **Clarity** | Query Analyzer | Understands what you're asking, identifies the company |
| **Research** | Data Gatherer | Finds news, stock info, and key developments |
| **Validator** | Quality Checker | Ensures research quality, requests more data if needed |
| **Synthesis** | Response Writer | Creates a clear, comprehensive answer |

## How It Works

```
Your Question
      │
      ▼
┌─────────────┐
│  Clarity    │──── Unclear? ───▶ Ask for Clarification
│   Agent     │                          │
└─────┬───────┘                          │
      │ Clear                            │
      ▼                                  │
┌─────────────┐                          │
│  Research   │◀─────────────────────────┘
│   Agent     │
└─────┬───────┘
      │
      ▼ Low Confidence?
┌─────────────┐
│  Validator  │──── Not Enough? ───▶ Retry (up to 3x)
│   Agent     │
└─────┬───────┘
      │ Good Enough
      ▼
┌─────────────┐
│  Synthesis  │
│   Agent     │
└─────────────┘
      │
      ▼
   Your Answer
```

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))

### Installation

```bash
git clone https://github.com/aisemanticexpert/LangGraphResearchAssistant.git
cd LangGraphResearchAssistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running the Assistant

```bash
# Interactive chat mode
python -m src.research_assistant.main

# Single query
python -m src.research_assistant.main -q "Tell me about NVIDIA's AI chips"

# Start REST API server
python -m src.research_assistant.main --api

# Debug mode
python -m src.research_assistant.main -v
```

## REST API

Start the API server:

```bash
python -m src.research_assistant.main --api --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check with cache stats |
| `/companies` | GET | List available companies |
| `/query` | POST | Process a new research query |
| `/continue` | POST | Continue an existing conversation |
| `/clarify` | POST | Provide clarification for interrupted query |
| `/conversation/{thread_id}` | GET | Get conversation state |
| `/export` | POST | Export conversation to JSON/Markdown |
| `/exports` | GET | List all exports |
| `/cache/stats` | GET | Get cache statistics |
| `/cache/clear` | POST | Clear the cache |

### API Examples

```bash
# Process a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Apple working on?", "use_cache": true}'

# Continue conversation
curl -X POST http://localhost:8000/continue \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "your-thread-id", "query": "What about their competitors?"}'

# Export conversation
curl -X POST http://localhost:8000/export \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "your-thread-id", "format": "markdown"}'
```

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Interactive Commands

When running in interactive mode, these commands are available:

| Command | Description |
|---------|-------------|
| `new` | Start a fresh conversation |
| `state` | Show current conversation state |
| `export` | Export conversation to JSON and Markdown |
| `cache` | Show cache statistics |
| `companies` | List all 25+ available companies |
| `help` | Show help message |
| `quit` | Exit the application |

## Configuration

Edit `.env` to customize behavior:

```bash
# Required
ANTHROPIC_API_KEY=your_key_here

# Model Settings
DEFAULT_MODEL=claude-sonnet-4-20250514
TEMPERATURE=0.0

# Research Settings
USE_MOCK_DATA=true          # Set false for real Tavily search
MAX_RESEARCH_ATTEMPTS=3
CONFIDENCE_THRESHOLD=6.0

# Persistence (memory, sqlite, postgres)
CHECKPOINT_BACKEND=memory
SQLITE_PATH=data/checkpoints.db

# Caching
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=100

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_PATH=logs/research_assistant.log

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

## Supported Companies (Mock Data)

The built-in mock data includes 25+ companies across multiple industries:

**Technology:** Apple, Microsoft, Google, Amazon, Meta, NVIDIA, AMD, Intel, Salesforce, Oracle, Adobe

**Finance:** JPMorgan Chase, Visa, PayPal, Square (Block)

**Healthcare:** Pfizer, Johnson & Johnson, UnitedHealth

**Retail:** Walmart, Costco, Nike, Starbucks

**Automotive:** Tesla, Toyota, Ford

**Entertainment:** Netflix, Disney, Spotify

For real-time data, get a [Tavily API key](https://tavily.com) and set `USE_MOCK_DATA=false`.

## Project Structure

```
LangGraphResearchAssistant/
├── src/research_assistant/
│   ├── agents/              # The 4 agent implementations
│   │   ├── base.py          # Base agent class
│   │   ├── clarity_agent.py
│   │   ├── research_agent.py
│   │   ├── validator_agent.py
│   │   └── synthesis_agent.py
│   ├── routing/             # Workflow routing logic
│   │   └── conditions.py
│   ├── tools/               # Research tools and mock data
│   │   ├── research_tool.py
│   │   └── mock_data.py     # 25+ company database
│   ├── utils/               # Utility modules
│   │   ├── logging.py       # File logging with rotation
│   │   ├── cache.py         # Query caching (LRU + TTL)
│   │   ├── export.py        # JSON/Markdown export
│   │   └── persistence.py   # SQLite/Postgres persistence
│   ├── api.py               # FastAPI REST server
│   ├── app.py               # Main application class
│   ├── config.py            # Configuration management
│   ├── graph.py             # LangGraph workflow
│   ├── main.py              # CLI entry point
│   └── state.py             # State schemas
├── tests/                   # Test suite
│   ├── test_api.py
│   ├── test_cache.py
│   ├── test_export.py
│   ├── test_mock_data.py
│   └── ...
├── logs/                    # Log files (auto-created)
├── exports/                 # Exported conversations
├── data/                    # SQLite database
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Docker

```bash
# Build image
docker build -t research-assistant .

# Run with docker-compose
docker-compose up

# Run single query
docker-compose run --rm research-assistant \
  python -m src.research_assistant.main -q "Tell me about Tesla"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/research_assistant --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Architecture Highlights

- **LangGraph StateGraph** for workflow orchestration
- **Pydantic** models for type-safe state management
- **LangGraph interrupt()** for human-in-the-loop
- **MemorySaver/SqliteSaver** for state checkpointing
- **LRU cache** with TTL for query caching
- **FastAPI** for REST API with automatic OpenAPI docs
- **Rotating file logs** for production debugging

## What Makes This Special

1. **Production-Ready** - Not just a demo, but a full-featured system with caching, logging, persistence, and REST API

2. **Beyond Expectations** - Goes well beyond basic multi-agent chat with:
   - Query caching to reduce API costs
   - Conversation export for archival
   - SQLite persistence for state recovery
   - REST API for integration
   - 25+ companies in mock database
   - Comprehensive test suite

3. **Clean Architecture** - Modular design with clear separation of concerns

4. **Well Documented** - Comprehensive README, API docs, and code comments

## License

MIT

---

Built with LangGraph, LangChain, and Claude by Rajesh Gupta
