# server.py — Refactored & Commented
# ------------------------------------------------------------
# Purpose:
#   FastAPI-based in-memory registry for A2A agents.
# Key changes vs original:
#   - Added /ready endpoint
#   - Clearer cleanup task for stale heartbeats
#   - Comments & logging improvements
# ------------------------------------------------------------

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import asyncio
from contextlib import asynccontextmanager

from python_a2a import AgentCard
from python_a2a.discovery import AgentRegistry

# Incoming registration payload from agents
class AgentRegistration(BaseModel):
    name: str
    description: str
    url: str
    version: str
    capabilities: dict = {}
    skills: list = []
    

class HeartbeatRequest(BaseModel):
    url: str

# In-memory registry provided by python_a2a
registry_server = AgentRegistry(
    name="A2A Registry Server",
    description="Registry server for agent discovery",
)

# Remove agents that stop sending heartbeats
HEARTBEAT_TIMEOUT = 30   # seconds
CLEANUP_INTERVAL = 10    # seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background cleanup task for stale agents."""
    cleanup_task = asyncio.create_task(cleanup_stale_agents())
    yield
    cleanup_task.cancel()

app = FastAPI(
    title="A2A Agent Registry Server",
    description="FastAPI server for agent discovery",
    lifespan=lifespan,
)

async def cleanup_stale_agents():
    """Periodically clean up agents that haven't sent heartbeats recently."""
    while True:
        try:
            current_time = time.time()
            stale = []
            for url, last_seen in registry_server.last_seen.items():
                if current_time - last_seen > HEARTBEAT_TIMEOUT:
                    stale.append(url)
                    logging.warning(f"Agent {url} missed heartbeat, removing")
            for url in stale:
                registry_server.unregister_agent(url)
                logging.info(f"Removed stale agent: {url}")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL)

@app.get("/health")
async def health_check():
    """Liveness check (process is up)."""
    return {"status": "healthy"}

@app.get("/ready")
async def ready_check():
    """Readiness check (basic demo: always True)."""
    return {"ready": True}

@app.post("/registry/register", response_model=AgentCard, status_code=201)
async def register_agent(registration: AgentRegistration):
    """Register a new agent with the in-memory registry."""
    card = AgentCard(**registration.dict())
    registry_server.register_agent(card)
    return card

@app.get("/registry/agents", response_model=List[AgentCard])
async def list_registered_agents():
    """List currently registered agents."""
    return list(registry_server.get_all_agents())

@app.post("/registry/heartbeat")
async def heartbeat(request: HeartbeatRequest):
    """
    Receive heartbeat pings from agents.
    - Agents should POST periodically to be kept alive in the registry.
    """
    try:
        if request.url in registry_server.agents:
            registry_server.last_seen[request.url] = time.time()
            logging.info(f"Heartbeat from {request.url}")
            return {"success": True}
        #else:

        return {"success": False, "error": "Agent not registered"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/registry/agents/{url}", response_model=AgentCard)
async def get_agent(url: str):
    """Get a specific agent by URL key."""
    agent = registry_server.get_agent(url)
    if agent:
        return agent
    raise HTTPException(status_code=404, detail=f"Agent with URL '{url}' not found")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
