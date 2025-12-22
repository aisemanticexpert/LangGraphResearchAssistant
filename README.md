# LangGraph Research Assistant

Multi-agent system for company research. Built this as a learning project to understand how LangGraph handles agent orchestration, state management, and the whole "human-in-the-loop" thing.

## What's this?

4 agents that work together:
- **Clarity** - figures out what you're asking (and bugs you if it's unclear)
- **Research** - grabs company info from mock data or Tavily
- **Validator** - checks if research is good enough, loops back if not
- **Synthesis** - writes up the final answer

The interesting part is the routing - agents can loop back, interrupt for user input, etc.

## Quick Start

```bash
# setup
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env

# run it
python -m src.research_assistant.main
```

Or start the API:
```bash
python -m src.research_assistant.main --api
# then hit http://localhost:8000/docs
```

## How it works

```
User Query
    ↓
[Clarity] → unclear? → interrupt, ask user → loop back
    ↓ clear
[Research] → low confidence? → [Validator] → not good enough? → retry (max 3x)
    ↓                              ↓ good enough
[Synthesis] ←──────────────────────┘
    ↓
Response
```

The routing logic is in `routing/conditions.py`. Pretty straightforward - just checking state values.

## State

Everything flows through a typed dict (see `state.py`):

```python
user_query: str
messages: List[Message]  # conversation history
clarity_status: str  # "clear" | "needs_clarification"
detected_company: str
research_findings: ResearchFindings
confidence_score: float  # 0-10
validation_result: str  # "sufficient" | "insufficient"
research_attempts: int  # caps at 3
final_response: str
# ... plus some housekeeping fields
```

## The agents

Each one inherits from `BaseAgent` which handles the boring stuff (LLM setup, JSON parsing, logging).

**ClarityAgent** - Looks at the query, tries to extract a company name. If it can't figure out what you want, sets `clarity_status = "needs_clarification"` and the graph interrupts.

**ResearchAgent** - Calls the research tool (either Tavily or mock data depending on config). Scores its own confidence. If score < 6, validator gets involved.

**ValidatorAgent** - Reviews what research found. Can send it back for another try if it's not good enough.

**SynthesisAgent** - Takes everything and writes a human-readable response.

## Multi-turn conversations

Uses LangGraph's checkpointing. Each conversation gets a thread_id, state persists between turns. So you can ask "tell me about Apple" then follow up with "what about their competitors?" and it remembers the context.

## Configuration

Most stuff is in `.env`:

| Variable | Default | Notes |
|----------|---------|-------|
| ANTHROPIC_API_KEY | - | required |
| USE_MOCK_DATA | true | set false + add TAVILY_API_KEY for real search |
| MAX_RESEARCH_ATTEMPTS | 3 | how many times validator can send back to research |
| CONFIDENCE_THRESHOLD | 6.0 | below this triggers validation |
| CHECKPOINT_BACKEND | memory | or "sqlite" if you want persistence |

## Project layout

```
src/research_assistant/
├── agents/          # the 4 agents + base class
├── routing/         # routing functions
├── tools/           # research tool + mock data (28 companies)
├── utils/           # caching, logging, export, etc
├── graph.py         # the langgraph workflow
├── app.py           # main app class
├── api.py           # fastapi endpoints
└── main.py          # cli entrypoint
```

## Testing

```bash
pytest tests/ -v
```

252 tests, mostly passing. Some of the agent tests are more like integration tests since they need to mock the LLM.

## Extra stuff I added

Beyond the basic requirements:

1. **REST API** - FastAPI with swagger docs
2. **Caching** - LRU cache so repeated queries don't hit the API
3. **Confidence scoring** - hybrid approach: rule-based (60%) + LLM assessment (40%). More predictable than pure LLM scoring.
4. **Error handling** - wrapped agents in try/catch, routes to error handler node instead of crashing
5. **Intent classification** - figures out if you want news, financials, competitor info, etc.
6. **Grounding validation** - checks if the response actually uses facts from the research (catches hallucinations)
7. **Retry tracking** - measures if retries actually help or just waste time

The confidence scoring thing was interesting to build - see `utils/confidence.py`. Basically scores data completeness, freshness, relevance, etc. individually then combines them.

## Assumptions

- Mock data has 28 companies. Unknown companies get low confidence scores.
- One company per query (mostly). Follow-ups preserve context.
- Claude sonnet is the default model. Can change in config.
- In-memory state by default. SQLite available if you need persistence.

## Known issues / TODOs

- [ ] The intent classification could be smarter
- [ ] Would be nice to support multi-company comparisons better
- [ ] Grounding validation is pretty basic - lots of false positives
- [ ] Some tests are flaky when mocking LLM responses

---

Built with LangGraph + Claude. Learned a lot about agent orchestration patterns.
