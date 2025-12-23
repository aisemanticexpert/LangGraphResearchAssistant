"""
FastAPI REST API for the Research Assistant.

Provides HTTP endpoints for interacting with the multi-agent workflow,
including query processing, conversation management, and exports.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
import base64

from .app import ResearchAssistantApp
from .config import settings
from .utils.cache import query_cache
from .utils.export import exporter
from .tools.mock_data import list_available_companies

logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Research Assistant API",
    description="Multi-agent research assistant powered by LangGraph and Claude",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global app instance
research_app: Optional[ResearchAssistantApp] = None


def get_app() -> ResearchAssistantApp:
    """Get or create the research assistant app instance."""
    global research_app
    if research_app is None:
        research_app = ResearchAssistantApp()
    return research_app


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for new queries."""
    query: str = Field(..., description="The research query", min_length=1)
    use_cache: bool = Field(default=True, description="Whether to use cached results")


class ContinueRequest(BaseModel):
    """Request model for continuing a conversation."""
    thread_id: str = Field(..., description="The conversation thread ID")
    query: str = Field(..., description="The follow-up query", min_length=1)


class ClarificationRequest(BaseModel):
    """Request model for providing clarification."""
    thread_id: str = Field(..., description="The conversation thread ID")
    clarification: str = Field(..., description="The clarification response", min_length=1)


class ExportRequest(BaseModel):
    """Request model for exporting conversations."""
    thread_id: str = Field(..., description="The conversation thread ID")
    format: str = Field(default="json", description="Export format (json or markdown)")


class QueryResponse(BaseModel):
    """Response model for queries."""
    thread_id: str
    final_response: Optional[str] = None
    interrupted: bool = False
    interrupt_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False
    data_source: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    cache_stats: Dict[str, Any]


class CompanyListResponse(BaseModel):
    """Response model for company list."""
    companies: List[str]
    total: int


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cache_stats=query_cache.get_stats(),
    )


@app.get("/companies", response_model=CompanyListResponse)
async def list_companies():
    """List all available companies in mock data."""
    companies = list_available_companies()
    return CompanyListResponse(
        companies=companies,
        total=len(companies),
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a new research query.

    Starts a new conversation thread and processes the query through
    the multi-agent workflow.
    """
    try:
        app_instance = get_app()

        # Check cache first if enabled
        if request.use_cache:
            cached_result = query_cache.get(request.query)
            if cached_result:
                return QueryResponse(
                    thread_id=cached_result.get("thread_id", "cached"),
                    final_response=cached_result.get("final_response"),
                    interrupted=False,
                    cached=True,
                )

        # Process query
        result = app_instance.start_conversation(request.query)

        # Cache successful results
        if result.get("final_response") and not result.get("interrupted"):
            query_cache.set(request.query, result)

        return QueryResponse(
            thread_id=result["thread_id"],
            final_response=result.get("final_response"),
            interrupted=result.get("interrupted", False),
            interrupt_info=result.get("interrupt_info"),
            error=result.get("error"),
            cached=False,
            data_source=result.get("data_source"),
        )

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continue", response_model=QueryResponse)
async def continue_conversation(request: ContinueRequest):
    """
    Continue an existing conversation with a follow-up query.
    """
    try:
        app_instance = get_app()
        result = app_instance.continue_conversation(
            request.thread_id,
            request.query
        )

        return QueryResponse(
            thread_id=result["thread_id"],
            final_response=result.get("final_response"),
            interrupted=result.get("interrupted", False),
            interrupt_info=result.get("interrupt_info"),
            error=result.get("error"),
            cached=False,
            data_source=result.get("data_source"),
        )

    except Exception as e:
        logger.error(f"Continue conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clarify", response_model=QueryResponse)
async def provide_clarification(request: ClarificationRequest):
    """
    Provide clarification for an interrupted conversation.
    """
    try:
        app_instance = get_app()
        result = app_instance.resume_with_clarification(
            request.thread_id,
            request.clarification
        )

        return QueryResponse(
            thread_id=result["thread_id"],
            final_response=result.get("final_response"),
            interrupted=result.get("interrupted", False),
            interrupt_info=result.get("interrupt_info"),
            error=result.get("error"),
            cached=False,
            data_source=result.get("data_source"),
        )

    except Exception as e:
        logger.error(f"Clarification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{thread_id}")
async def get_conversation_state(thread_id: str):
    """
    Get the current state of a conversation.
    """
    try:
        app_instance = get_app()
        state = app_instance.get_conversation_state(thread_id)

        if state is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "thread_id": thread_id,
            "state": {
                "detected_company": state.get("detected_company"),
                "clarity_status": state.get("clarity_status"),
                "research_attempts": state.get("research_attempts", 0),
                "confidence_score": state.get("confidence_score", 0.0),
                "validation_result": state.get("validation_result"),
                "final_response": state.get("final_response"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_conversation(request: ExportRequest):
    """
    Export a conversation to JSON or Markdown format.
    """
    try:
        app_instance = get_app()
        state = app_instance.get_conversation_state(request.thread_id)

        if state is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = state.get("messages", [])

        if request.format.lower() == "markdown":
            filepath = exporter.export_to_markdown(request.thread_id, state, messages)
        else:
            filepath = exporter.export_to_json(request.thread_id, state, messages)

        return {
            "success": True,
            "filepath": filepath,
            "format": request.format,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exports")
async def list_exports():
    """List all exported conversations."""
    try:
        exports = exporter.list_exports()
        return {
            "exports": exports,
            "total": len(exports),
        }
    except Exception as e:
        logger.error(f"List exports error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache."""
    query_cache.clear()
    return {"success": True, "message": "Cache cleared"}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return query_cache.get_stats()


CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Assistant</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.ico">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: #1e293b;
            padding: 1rem 2rem;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-left h1 { font-size: 1.25rem; font-weight: 600; }
        .header-left p { font-size: 0.875rem; color: #94a3b8; margin-top: 0.25rem; }
        .new-chat-btn {
            padding: 0.5rem 1rem;
            background: #10b981;
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .new-chat-btn:hover { background: #059669; }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message {
            max-width: 80%;
            padding: 1rem;
            border-radius: 0.75rem;
            line-height: 1.5;
        }
        .message.user {
            background: #3b82f6;
            align-self: flex-end;
            border-bottom-right-radius: 0.25rem;
        }
        .message.assistant {
            background: #1e293b;
            align-self: flex-start;
            border-bottom-left-radius: 0.25rem;
            border: 1px solid #334155;
        }
        .message.error {
            background: #7f1d1d;
            border-color: #991b1b;
        }
        .mock-warning {
            background: #991b1b;
            color: #fecaca;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            border: 1px solid #dc2626;
        }
        .message pre {
            background: #0f172a;
            padding: 0.75rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin-top: 0.5rem;
            font-size: 0.875rem;
        }
        .input-container {
            padding: 1rem 2rem;
            background: #1e293b;
            border-top: 1px solid #334155;
        }
        .input-wrapper {
            display: flex;
            gap: 0.75rem;
            max-width: 900px;
            margin: 0 auto;
        }
        input {
            flex: 1;
            padding: 0.875rem 1rem;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 1rem;
        }
        input:focus { outline: none; border-color: #3b82f6; }
        input::placeholder { color: #64748b; }
        button {
            padding: 0.875rem 1.5rem;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            font-weight: 500;
        }
        button:hover { background: #2563eb; }
        button:disabled { background: #475569; cursor: not-allowed; }
        .typing { color: #94a3b8; font-style: italic; }
        .meta { font-size: 0.75rem; color: #64748b; margin-top: 0.5rem; }
    </style>
</head>
<body>
    <header>
        <div class="header-left">
            <h1>Research Assistant</h1>
            <p>Ask about companies, stocks, financials, and market research</p>
        </div>
        <button class="new-chat-btn" id="newChat" title="Start a new conversation">
            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path d="M12 5v14M5 12h14"/>
            </svg>
            New Chat
        </button>
    </header>
    <div class="chat-container" id="chat"></div>
    <div class="input-container">
        <div class="input-wrapper">
            <input type="text" id="input" placeholder="Ask about a company..." autocomplete="off">
            <button id="send">Send</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        let threadId = null;

        function addMessage(content, type) {
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = content.replace(/\\n/g, '<br>');
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const query = input.value.trim();
            if (!query) return;

            input.value = '';
            sendBtn.disabled = true;
            addMessage(query, 'user');
            const typing = addMessage('Thinking...', 'assistant typing');

            try {
                const endpoint = threadId ? '/continue' : '/query';
                const body = threadId
                    ? { thread_id: threadId, query }
                    : { query, use_cache: true };

                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });

                const data = await res.json();
                typing.remove();

                if (data.error) {
                    addMessage(data.error, 'assistant error');
                } else if (data.interrupted && data.interrupt_info) {
                    threadId = data.thread_id;
                    const msg = data.interrupt_info.message || 'Please provide more information.';
                    addMessage(msg, 'assistant');
                } else {
                    threadId = data.thread_id;
                    const response = data.final_response || 'No response received.';

                    // Show mock data warning if not using live API
                    if (data.data_source && data.data_source === 'mock_data') {
                        const warning = document.createElement('div');
                        warning.className = 'mock-warning';
                        warning.innerHTML = '⚠️ WARNING: Results are from MOCK DATA, not live API data. Tavily API may be unavailable.';
                        chat.appendChild(warning);
                    }

                    addMessage(response, 'assistant');
                }
            } catch (err) {
                typing.remove();
                addMessage('Connection error: ' + err.message, 'assistant error');
            }

            sendBtn.disabled = false;
            input.focus();
        }

        function startNewChat() {
            threadId = null;
            chat.innerHTML = '';
            input.value = '';
            input.focus();
            // Add welcome message
            addMessage('Hello! I\\'m the Research Assistant. Ask me about any company - stocks, financials, news, leadership, and more.', 'assistant');
        }

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        document.getElementById('newChat').addEventListener('click', startNewChat);

        // Show welcome message on load
        addMessage('Hello! I\\'m the Research Assistant. Ask me about any company - stocks, financials, news, leadership, and more.', 'assistant');
        input.focus();
    </script>
</body>
</html>
"""


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat interface."""
    return CHAT_HTML


# SVG favicon - research/chart icon
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="6" fill="#3b82f6"/>
  <path d="M8 22V14M12 22V10M16 22V16M20 22V12M24 22V8" stroke="white" stroke-width="2" stroke-linecap="round"/>
  <circle cx="20" cy="12" r="4" fill="none" stroke="white" stroke-width="1.5"/>
  <line x1="23" y1="15" x2="26" y2="18" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


def run_server():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "src.research_assistant.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
