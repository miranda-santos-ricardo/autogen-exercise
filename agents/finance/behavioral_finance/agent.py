# agentAG2.py — Refactored with Persistent Event Loop (Thread) + Comments
from __future__ import annotations

import atexit
import argparse
import asyncio
import os
import json
import socket
import logging
import threading
import requests
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# --- A2A server & discovery ---
from python_a2a import (
    AgentCard,
    AgentSkill,
    A2AServer,
    run_server,
    Message,
    TextContent,
    MessageRole,
)
from python_a2a.discovery import enable_discovery

# --- AutoGen (agents, models, tools) ---
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_core.tools import FunctionTool
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("A2A-DataAnalystAgent")

load_dotenv()


def find_free_port() -> int:
    """Find an available TCP port bound to 0.0.0.0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]

def tool_meta(t: Any) -> Dict[str, str]:
    """Extrai metadados mínimos para publicar ferramentas no AgentCard."""
    name = getattr(t, "name", getattr(t, "__name__", "tool"))
    desc = getattr(t, "description", "") or (t.__doc__ or "").strip()
    return {"name": name, "description": (desc or "")[:280]}


class AutoGenA2AAgent(A2AServer, AssistantAgent):
    """
    Agente AutoGen exposed with A2A with discovery/registry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        registry_url: Optional[str],
        model_name: str,
        model_api_key: str,
        version: str,
        tools: Optional[List[Any]] = None,
        registration_retries: int = 3,
        heartbeat_interval: int = 10,
        system_prompt: str = "You are a helpful assistant.",
        agent_domain_expertise: str = "general-purpose",
        agent_subdomain_expertise: str = "general-purpose"
    ) -> None:
        # 1) AgentCard com skills/capabilities
        skill = AgentSkill(
            name=name,
            description=description,
        )
        card = AgentCard(
            name=name,
            description=description,
            url=url,
            version=version,
            skills=[skill],
            capabilities={
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
                "google_a2a_compatible": True,
                "parts_array_format": True,
                "agent_domain_expertise":agent_domain_expertise,
                "agent_subdomain_expertise":agent_subdomain_expertise
            },
        )

        # 2) Bases: A2AServer + AssistantAgent
        A2AServer.__init__(self, agent_card=card, google_a2a_compatible=True)
        #TODO: refactor this to decide between OpenAI or AzureOpenAI based on env vars
        model_client = OpenAIChatCompletionClient(model=model_name, api_key=model_api_key)
        #model_client = AzureOpenAIChatCompletionClient(
        #    azure_deployment=model_name,
        #    model=model_name,
        #    api_version=os.getenv("OPENAI_API_VERSION","2024-12-01-preview"),
        #    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT","https://aifoundrymaven.openai.azure.com/"),
        #)
        
        AssistantAgent.__init__(
            self,
            name=name,
            description=description,
            model_client=model_client,
            tools=tools or [],
            system_message=system_prompt,
        )

        # 3) Descoberta/Registry
        self.registry_url = registry_url
        self._registration_retries = int(registration_retries)
        self._heartbeat_interval = int(heartbeat_interval)
        self._discovery_client = None
        self.url = url
        self._ready = False

        # 4) Publica metadados das ferramentas
        self._tools: List[Any] = tools or []
        self._publish_tools_metadata()

        # 5) LOOP PERSISTENTE em THREAD
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="agent-event-loop", daemon=True
        )
        self._loop_thread.start()

        # Clean Shutdown
        def _shutdown():
            try:
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread.is_alive():
                    self._loop_thread.join(timeout=2)
            except Exception:
                pass

        atexit.register(_shutdown)

    # -------------------------------------
    # Tools metadata into AgentCard
    # -------------------------------------
    def _publish_tools_metadata(self) -> None:
        meta = [tool_meta(t) for t in self._tools]
        #if not any(m.get("name") == "somar" for m in meta):
        #    meta.insert(0, tool_meta(SOMAR_TOOL))
        self.agent_card.capabilities["tools"] = meta
        logger.info("Published tools on AgentCard: %s", meta)

    # -------------------------------------
    # Registry setup
    # -------------------------------------
    async def setup(self) -> None:
        """Registra agente no registry e inicia heartbeats."""
        if not self.registry_url:
            logger.warning("No registry URL configured; discovery disabled.")
            self._ready = True
            return


        for attempt in range(self._registration_retries):
            try:
                self._discovery_client = enable_discovery(
                    self,
                    registry_url=self.registry_url,
                    heartbeat_interval=self._heartbeat_interval,
                )

                def heartbeat_callback(results):
                    for result in results:
                        if result.get("success"):
                            logger.info("Heartbeat OK → %s", result["registry"])
                        else:
                            logger.warning(
                                "Heartbeat FAIL → %s: %s",
                                result["registry"],
                                result.get("message", "Unknown error"),
                            )

                self._discovery_client.heartbeat_callback = heartbeat_callback

                await asyncio.sleep(1.5)

                resp = requests.get(f"{self.registry_url}/registry/agents", timeout=5)
                if resp.status_code == 200:
                    agents = resp.json()
                    if any(agent.get("url") == self.url for agent in agents):
                        logger.info("Agent '%s' registered at %s", self.agent_card.name, self.registry_url)
                        self._ready = True
                        return
                    else:
                        logger.warning("Registration not visible yet (attempt %d)", attempt + 1)
                else:
                    logger.warning("Verify registration failed: %s (attempt %d)", resp.status_code, attempt + 1)

                await asyncio.sleep(2)
            except Exception as e:
                logger.error("Registration error (attempt %d): %s", attempt + 1, e)
                if attempt < self._registration_retries - 1:
                    await asyncio.sleep(2)

        logger.error("Failed to register agent after %d attempts", self._registration_retries)

    # -------------------------------------
    # Helpers
    # -------------------------------------
    def _to_text(self, resp: Any) -> str:
        """Normaliza formatos de resposta para string."""
        try:
            if hasattr(resp, "chat_message") and hasattr(resp.chat_message, "content"):
                return str(resp.chat_message.content)
            if hasattr(resp, "content"):
                return str(resp.content)
            if isinstance(resp, (dict, list)):
                return json.dumps(resp, ensure_ascii=False)
            return str(resp)
        except Exception as e:
            return f"(unable to normalize response: {e})"

    def _json_envelope(self, ok: bool, data: Any = None, error: Optional[str] = None) -> str:
        """Contrato consistente para o orquestrador."""
        return json.dumps({"ok": ok, "data": data, "error": error}, ensure_ascii=False)

    # -------------------------------------
    # A2A message handling
    # -------------------------------------
    def handle_message(self, message: Message) -> Message:
        """
        Roda a pipeline do AssistantAgent num loop assíncrono PERSISTENTE (thread).
        Evita criar/fechar loops por requisição (que causava 'Task was destroyed ...').
        """
        user_text = (message.content.text or "").strip()

        #if the server request to resign should execute the setup again
        if user_text.lower == "re-sign":
            asyncio.run_coroutine_threadsafe(self.setup(), self._loop)

        async def assistant_run(prompt: str):
            try:
                resp = await self.on_messages(
                    [TextMessage(content=prompt, source="user")],
                    cancellation_token=CancellationToken(),
                )
                text = self._to_text(resp)
                return self._json_envelope(True, data=text)
            except Exception as e:
                logger.exception("Assistant error")
                return self._json_envelope(False, data=None, error=str(e))

        # Enfileira a coroutine no loop persistente
        fut = asyncio.run_coroutine_threadsafe(assistant_run(user_text), self._loop)
        try:
            response_text = fut.result(timeout=60)  # ajuste conforme necessidade
        except FuturesTimeout:
            fut.cancel()
            response_text = self._json_envelope(False, error="Timeout running assistant coroutine")
        except Exception as e:
            response_text = self._json_envelope(False, error=f"Execution error: {e}")

        return Message(
            content=TextContent(text=response_text or ""),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )


# -----------------------------------------------------------------------------
# Bootstrap & main
# -----------------------------------------------------------------------------
async def build_tools_from_mcp() -> List[Any]:
    """Implement MCP Server here."""
    #gcp_server = StdioServerParams(
    #    command="npx",
    #    args=["@cocal/google-calendar-mcp"],
    #    env={"GOOGLE_OAUTH_CREDENTIALS": os.getenv("GOOGLE_OAUTH_CREDENTIALS", "")},
    #    read_timeout_seconds=500,
    #)
    #tools = await mcp_server_tools(gcp_server)
    ## yield point para permitir que async generators internos avancem
    #await asyncio.sleep(0)
    #tools = None
    #return list(tools)
    return None


async def run_agent() -> None:
    """Sobe o agente com configuração robusta."""
    model_api_key = os.getenv("AGENT_MODEL_API_KEY") #or os.getenv("AZURE_OPENAI_API_KEY")
    if not model_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or AZURE_OPENAI_API_KEY")
    
    # Carrega tools do MCP
    mcp_tools = await build_tools_from_mcp()
    all_tools: List[Any] = None
    #all_tools: List[Any] = [*mcp_tools, SOMAR_TOOL]

    # Host/port/url
    port = os.getenv("AGENT_PORT")
    host = os.getenv("AGENT_HOST", "http://localhost").rstrip("/")
    final_port = int(os.getenv("AGENT_PORT") or (port if port else find_free_port()))
    url = f"{host}:{final_port}"

    # Instancia o agente
    agent = AutoGenA2AAgent(
        name=os.getenv("AGENT_NAME") or "AutoGenA2AAgent",
        description=os.getenv("AGENT_DESCRIPTION") or "AGENT_DESCRIPTION",
        url=url,
        registry_url=os.getenv("AGENT_REGISTRY_URL","http://localhost:8000"),
        model_name=os.getenv("AGENT_MODEL_NAME", "AGENT_MODEL_NAME"),
        model_api_key=model_api_key,
        version=os.getenv("AGENT_VERSION", "0.1.0"),
        tools=all_tools,
        registration_retries=int(os.getenv("REGISTRATION_RETRIES", "3")),
        heartbeat_interval=int(os.getenv("HEARTBEAT_INTERVAL", "10")),
        system_prompt=os.getenv("AGENT_SYSTEM_PROMPT") or "You are a helpful assistant.",
        agent_domain_expertise=os.getenv("AGENT_DOMAIN_EXPERTISE") or "general-purpose",
        agent_subdomain_expertise=os.getenv("AGENT_SUBDOMAIN_EXPERTISE") or "general-purpose"
    )

    await agent.setup()
    run_server(agent, host="0.0.0.0", port=final_port, debug=False)
    logger.info("Agent '%s' started on '%s':%d", agent.agent_card.name, host, final_port)


if __name__ == "__main__":
    asyncio.run(run_agent())
