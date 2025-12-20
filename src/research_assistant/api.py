"""
FastAPI REST API for the Research Assistant.

Provides HTTP endpoints for interacting with the multi-agent workflow,
including query processing, conversation management, and exports.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
