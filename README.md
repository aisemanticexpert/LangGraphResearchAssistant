# Research Assistant

A multi-agent system I built to explore how LangGraph handles agent workflows. It helps you research companies using 4 specialized agents that work together.

**Author:** Rajesh Gupta

## What's This?

I wanted to understand LangGraph better, so I built this research assistant as a learning project. The idea is simple - instead of one monolithic agent doing everything, we have 4 smaller agents that each focus on one thing and hand off work to each other.

The cool part? If your question is unclear, it asks for clarification instead of guessing. And if the research isn't good enough, it'll try again (up to 3 times).

## The Four Agents

| Agent | Job |
|-------|-----|
| **Clarity** | Figures out what you're asking and which company |
| **Research** | Finds the actual data (news, stock info, etc) |
| **Validator** | Checks if we have enough info or need to dig deeper |
| **Synthesis** | Writes up a nice response |

## How It Works

```
Your question
      ↓
[Clarity] → Unclear? → Ask for clarification
      ↓
   Clear
      ↓
[Research] → Low confidence? → [Validator] → Not enough? → Retry
      ↓
   Good enough
      ↓
[Synthesis]
      ↓
Your answer
```

## Getting Started

### You'll Need

- Python 3.10+
- Anthropic API key ([grab one here](https://console.anthropic.com))

### Setup

```bash
git clone https://github.com/aisemanticexpert/LangGraphResearchAssistant.git
cd LangGraphResearchAssistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run It

```bash
# Chat mode
python -m src.research_assistant.main

# One-off question
python -m src.research_assistant.main -q "What's Tesla working on?"

# Debug mode
python -m src.research_assistant.main -v
```

## Configuration

Edit `.env`:

```bash
# Required
ANTHROPIC_API_KEY=your_key_here

# Optional - real web search
TAVILY_API_KEY=your_tavily_key
USE_MOCK_DATA=false

# Defaults
DEFAULT_MODEL=claude-sonnet-4-20250514
MAX_RESEARCH_ATTEMPTS=3
LOG_LEVEL=INFO
```

### Mock vs Real Data

By default uses built-in data for Apple, Tesla, Microsoft, Google, Amazon, Meta, NVIDIA, Netflix. Good for testing without API costs.

For real-time search: get a [Tavily key](https://tavily.com), add to `.env`, set `USE_MOCK_DATA=false`.

## Example Chat

```
You: Tell me about Apple

Assistant: Apple Inc. is one of the largest tech companies...
(gives you a summary based on the research)
```

## Project Layout

```
src/research_assistant/
├── agents/          # The 4 agent implementations
├── routing/         # Logic for deciding which agent goes next
├── tools/           # Research tools and mock data
├── app.py           # Main application class
├── graph.py         # LangGraph workflow definition
└── state.py         # Shared state between agents
```

## Docker

If you prefer containers:

```bash
docker build -t research-assistant .
docker-compose up
```

## Tests

```bash
pytest tests/ -v
```

## Notes

- Uses mock data by default so you can test without burning API credits
- Conversation context is maintained so you can ask follow-up questions
- Max 3 research attempts before it gives you the best answer it has
- The human-in-the-loop uses LangGraph's interrupt() function

## License

MIT
