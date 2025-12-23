# LangGraph Research Assistant

**Author:** Rajesh Gupta

A multi-agent company research assistant built using LangGraph. The system uses four specialized agents working together to handle user queries about companies, stocks, and financial information.

---

## What I Built

This project implements a research assistant that can:

- Answer questions about 50+ companies (Apple, Tesla, Microsoft, Google, Amazon, etc.)
- Handle follow-up questions while maintaining conversation context
- Ask for clarification when queries are unclear
- Block harmful queries (market manipulation, insider trading attempts)
- Retry research automatically when data quality is low

The core of the system is a LangGraph workflow with four agents:

1. **ThinkSemantic Agent** - Analyzes user intent, detects company names, and blocks unsafe queries
2. **Research Agent** - Gathers company data from Tavily Search API or mock data
3. **Validator Agent** - Checks if research quality is sufficient (retries up to 3 times if not)
4. **Synthesis Agent** - Generates the final user-friendly response

---

## How to Set Up and Run

### Prerequisites

- Python 3.10 or higher
- An Anthropic API key (get one from https://console.anthropic.com/)

### Step 1: Clone and Install

```bash
git clone <repo-url>
cd LangGraphResearchAssistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Open `.env` in any text editor and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

That's all you need. The system uses mock data by default, so you can test without a Tavily key.

### Step 3: Run the Application

You have three options:

**Option A: Interactive Chat (CLI)**
```bash
python -m src.research_assistant.main
```

**Option B: Web Interface**
```bash
python -m src.research_assistant.main --api
```
Then open http://localhost:8000/chat in your browser.

**Option C: Single Query**
```bash
python -m src.research_assistant.main -q "Tell me about Apple"
```

### Step 4: Run Tests (Optional)

```bash
pytest tests/ -v
```

There are 281 tests covering all the agents, routing logic, and API endpoints.

---

## Try These Queries

| Query | What Happens |
|-------|--------------|
| `Tesla` | Returns company overview with stock info, news, financials |
| `MSFT stock price` | Shows current price, changes, 52-week range |
| `Who is Apple's CEO?` | Returns leadership information |
| `Should I buy NVDA?` | Discusses valuation factors (with disclaimer) |
| `Hello` | Friendly greeting |
| `What about their competitors?` | Follow-up that uses context from previous query |
| `asdfghjkl` | Blocked as gibberish |
| `Pump AAPL stock` | Blocked as market manipulation |

---

## How the LangGraph Workflow Works

Here's the flow of a typical query:

```
User Query
     │
     ▼
┌─────────────────┐
│  ThinkSemantic  │ ─── Detects intent, company, safety issues
└────────┬────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
 greeting  research   blocked
    │         │          │
   END        ▼         ask for
          Validator   clarification
              │
         ┌────┴────┐
         ▼         ▼
    sufficient  insufficient
         │         │
         ▼         ▼
     Synthesis   retry (max 3x)
         │         │
        END    ───┘
```

### Key LangGraph Features I Used

**StateGraph** - The workflow is defined as a graph where each node is an agent:

```python
workflow = StateGraph(ResearchAssistantState)
workflow.add_node("thinksemantic", thinksemantic_agent.run)
workflow.add_node("research", research_agent.run)
workflow.add_node("validator", validator_agent.run)
workflow.add_node("synthesis", synthesis_agent.run)
```

**Conditional Routing** - Different paths based on state:

```python
workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "research": "research",   # retry if insufficient
        "synthesis": "synthesis"  # proceed if good enough
    }
)
```

**Checkpointer** - Saves conversation state for multi-turn:

```python
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

**Interrupt** - Pauses for human clarification:

```python
from langgraph.types import interrupt

response = interrupt({
    "question": "Which company are you asking about?"
})
```

---

## Project Structure

```
src/research_assistant/
├── agents/
│   ├── thinksemantic_intent_agent.py  # Intent + safety analysis
│   ├── research_agent.py              # Data gathering
│   ├── validator_agent.py             # Quality checking
│   └── synthesis_agent.py             # Response generation
├── routing/
│   └── conditions.py                  # Routing functions
├── tools/
│   ├── research_tool.py               # Tavily integration
│   └── mock_data.py                   # Sample company data
├── graph.py                           # LangGraph workflow
├── state.py                           # State schema (Pydantic)
├── guardrails.py                      # Safety patterns
├── api.py                             # FastAPI endpoints
├── app.py                             # Application logic
└── main.py                            # CLI entry point
```

---

## Requirements Checklist

| # | Requirement | Where to Find It |
|---|-------------|------------------|
| 1 | 4 agents working together | `src/research_assistant/agents/` |
| 2 | State schema with all fields | `state.py` - `ResearchAssistantState` class |
| 3 | 3 conditional routing functions | `routing/conditions.py` |
| 4 | Validator→Research feedback loop | `validator_agent.py` + `route_after_validation()` |
| 5 | Interrupt for unclear queries | `graph.py` - `human_clarification_node()` |
| 6 | Multi-turn conversation memory | `app.py` - uses `MemorySaver` checkpointer |
| 7 | Example conversations | See below |
| 8 | Software engineering practices | Classes, type hints, 281 tests |
| 9 | README with instructions | This file |
| 10 | Assumptions documented | See "Assumptions" section |
| 11 | Beyond requirements | See "Beyond Requirements" section |

---

## Example Conversations

### Example 1: Validator Retry Loop

Shows the feedback loop between Validator and Research:

```
You: Tell me about Apple

[Internal]
→ ThinkSemantic: company_overview intent, company=Apple
→ Research: gathering data... confidence=4.2
→ Validator: insufficient (missing financials), attempt 1/3
→ Research: retrying with feedback... confidence=7.1
→ Validator: sufficient, proceeding
→ Synthesis: generating response

Bot: Apple Inc. (AAPL) is a technology company based in Cupertino, California.

Stock: $195.00 (+45% YTD)
Revenue: $383B | Net Income: $97B | P/E: 29.5

Recent news:
- Vision Pro launch
- Services revenue growing
- AI integration across products

CEO: Tim Cook
```

### Example 2: Multi-Turn Follow-Up

Shows conversation memory working:

```
You: Tesla stock price

Bot: Tesla (TSLA) is currently trading at $242.00
- Today: -1.2%
- 52-week range: $152 - $299
- Market cap: $770B

You: What about their competitors?

[System knows "their" = Tesla from context]

Bot: Tesla's main EV competitors:
- Rivian (RIVN): $17, focus on trucks
- Lucid (LCID): $4.50, luxury segment
- BYD: Largest by volume globally
- Ford/GM: Traditional automakers entering EV

You: Who is the CEO?

[Still in Tesla context]

Bot: Elon Musk is CEO of Tesla since 2008...
```

### Example 3: Clarification Request

Shows human-in-the-loop interrupt:

```
You: What's the stock?

Bot: Could you specify which company? For example "Apple stock" or "MSFT"

You: Microsoft

Bot: Microsoft (MSFT): $378.00 (+52% YTD)
52-week range: $245 - $385
Market cap: $2.8T
```

### Example 4: Blocked Query

Shows safety guardrails:

```
You: Give me insider tips on NVDA

Bot: I can't help with that - it appears to reference insider trading.

I can provide public information about NVIDIA:
- News and announcements
- Stock price and metrics
- Company overview

Want me to research NVIDIA using public info?

You: Yes

Bot: NVIDIA Corporation (NVDA)...
[normal research response]
```

---

## Assumptions I Made

Since this was an open-ended problem, I made these decisions:

1. **ThinkSemantic instead of Clarity Agent** - The requirements mentioned a "Clarity Agent" for checking if queries are clear. I built ThinkSemantic which does that plus intent classification, company detection, and safety checks. It's a superset of what Clarity would do.

2. **Mock data works without API keys** - You can test the full system with just an Anthropic key. Tavily is optional for live web search.

3. **One company per query** - The system handles one company at a time. "Compare Apple and Microsoft" would need separate queries.

4. **Confidence threshold of 6.0** - After testing, I found scores below 6 usually mean missing important data. This is configurable in `.env`.

5. **Max 3 research attempts** - If data quality is still low after 3 tries, synthesize the best available answer rather than failing.

6. **Investment queries get disclaimers** - Any query that sounds like investment advice includes a financial disclaimer.

---

## Beyond Requirements

Things I added beyond the base requirements:

### 1. ThinkSemantic with Chain-of-Thought

Instead of a simple clarity check, I built an intent analysis system that:
- Classifies queries into types (greeting, stock_price, investment, leadership, etc.)
- Detects 50+ company names and tickers
- Blocks 30+ manipulation patterns
- Filters gibberish queries
- Logs reasoning for debugging

### 2. RAGHEAT-Inspired Confidence Scoring

Instead of simple heuristics, the Research Agent uses weighted factors:

| Factor | Weight |
|--------|--------|
| Data completeness | 30% |
| Source diversity | 20% |
| News coverage | 15% |
| Financial data | 15% |
| Data recency | 10% |
| Sentiment consistency | 10% |

### 3. Web UI

Browser-based chat at `/chat` with typing indicators and suggested queries.

### 4. REST API with Swagger

Full API with documentation at `/docs`.

### 5. 281 Tests

Comprehensive test coverage for all agents, routing, and endpoints.

### 6. Mock Data for 50+ Companies

Tech, finance, healthcare, retail, automotive sectors covered.

---

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Claude API key |
| `TAVILY_API_KEY` | optional | For live web search |
| `USE_MOCK_DATA` | true | Use mock data |
| `CONFIDENCE_THRESHOLD` | 6.0 | Min score to skip validator |
| `MAX_RESEARCH_ATTEMPTS` | 3 | Max retries |

---

## Troubleshooting

**"ANTHROPIC_API_KEY not set"**
- Check that `.env` exists and has your key

**"No data found for company X"**
- Company might not be in mock data
- Add `TAVILY_API_KEY` for live search

**Slow first response**
- Model loading takes a moment
- Subsequent queries are faster

---

## Key Files to Review

- `graph.py` - The LangGraph workflow definition
- `agents/thinksemantic_intent_agent.py` - Intent analysis logic
- `agents/validator_agent.py` - Quality checking and retry logic
- `state.py` - All the state fields and Pydantic models
- `routing/conditions.py` - The three routing functions

---

Rajesh Gupta
