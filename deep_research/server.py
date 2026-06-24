"""FastAPI HTTP server for the deep research agent.

Supports both profiles:
  POST /research         — start research (accepts profile: opus|multi)
  POST /research/batch   — batch research
  GET  /research/{id}    — get result
  GET  /tasks            — list all tasks
  GET  /health           — health check
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
import uuid
from typing import Optional
from datetime import datetime

from .core.loop import run_research

app = FastAPI(
    title="Deep Research Agent API",
    description="Iterative knowledge gap research — opus (premium) and multi (budget) profiles",
    version="2.0.0",
)

# In-memory task store (replace with Redis for production)
tasks = {}


class ResearchRequest(BaseModel):
    query: str
    profile: str = "opus"
    max_iterations: Optional[int] = None
    max_time_minutes: Optional[int] = None
    depth: str = "standard"


class BatchRequest(BaseModel):
    queries: list[str]
    profile: str = "opus"
    max_iterations: Optional[int] = None
    max_time_minutes: Optional[int] = None
    depth: str = "standard"


async def _run_task(task_id: str, query: str, profile: str = "opus", **kwargs):
    """Background task runner."""
    tasks[task_id]["status"] = "running"
    tasks[task_id]["started_at"] = datetime.now().isoformat()

    try:
        result = await run_research(query=query, profile=profile, **kwargs)
        tasks[task_id]["status"] = "complete"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        tasks[task_id]["result"] = {
            "report": result.report,
            "obsidian_path": result.obsidian_path,
            "iteration_count": result.iteration_count,
            "sources_count": len(result.sources_consulted),
            "sources": result.sources_consulted[:20],
            "profile": result.profile,
            "cost_summary": result.cost_summary,
        }
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.now().isoformat()


@app.get("/")
async def root():
    return {
        "service": "Deep Research Agent",
        "version": "2.0.0",
        "profiles": ["opus", "multi"],
        "endpoints": {
            "health": "/health",
            "start_research": "POST /research",
            "get_result": "GET /research/{task_id}",
            "batch_research": "POST /research/batch",
            "list_tasks": "GET /tasks",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "research-agent",
        "version": "2.0.0",
        "active_tasks": len([t for t in tasks.values() if t["status"] == "running"]),
    }


@app.post("/research")
async def start_research(req: ResearchRequest, bg: BackgroundTasks):
    """Start a research task (async)."""
    if req.profile not in ("opus", "multi"):
        raise HTTPException(400, f"Invalid profile '{req.profile}'. Use 'opus' or 'multi'.")

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "query": req.query,
        "profile": req.profile,
        "depth": req.depth,
        "created_at": datetime.now().isoformat(),
    }

    bg.add_task(
        _run_task,
        task_id,
        req.query,
        profile=req.profile,
        max_iterations=req.max_iterations,
        max_time_minutes=req.max_time_minutes,
        depth=req.depth,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "query": req.query,
        "profile": req.profile,
        "check_status": f"/research/{task_id}",
    }


@app.get("/research/{task_id}")
async def get_result(task_id: str):
    """Get research task status and results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.post("/research/batch")
async def batch_research(req: BatchRequest, bg: BackgroundTasks):
    """Start multiple research tasks in parallel."""
    task_ids = []
    for query in req.queries:
        task_id = str(uuid.uuid4())[:8]
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "query": query,
            "profile": req.profile,
            "depth": req.depth,
            "created_at": datetime.now().isoformat(),
        }
        bg.add_task(
            _run_task,
            task_id,
            query,
            profile=req.profile,
            max_iterations=req.max_iterations,
            max_time_minutes=req.max_time_minutes,
            depth=req.depth,
        )
        task_ids.append(task_id)

    return {
        "batch_id": str(uuid.uuid4())[:8],
        "task_ids": task_ids,
        "count": len(task_ids),
        "profile": req.profile,
        "check_status": "/tasks",
    }


@app.get("/tasks")
async def list_tasks():
    """List all research tasks."""
    return {
        "tasks": list(tasks.values()),
        "total": len(tasks),
        "running": len([t for t in tasks.values() if t["status"] == "running"]),
        "complete": len([t for t in tasks.values() if t["status"] == "complete"]),
        "error": len([t for t in tasks.values() if t["status"] == "error"]),
    }


# Run with: uvicorn deep_research.server:app --host 0.0.0.0 --port 3002
