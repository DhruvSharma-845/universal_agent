import asyncio
from typing import Optional

from model import getModel
from tools import getTools
from agent import getAgent


class _AgentManager:
    """Internal agent manager (not exposed directly)"""
    
    def __init__(self):
        self._agent = None
        self._model = None
        self._tools = None
        self._config = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the agent components"""
        async with self._lock:
            if not self._initialized:
                print("Initializing Agent...")
                
                self._model = getModel()
                self._tools = await getTools()
                self._agent = getAgent(self._model, self._tools)
                
                self._config = {"configurable": {"thread_id": "1"}}
                
                self._initialized = True
                print("Agent initialized successfully!")
    
    @property
    def agent(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._agent
    
    @property
    def model(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._model
    
    @property
    def tools(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._tools

    @property
    def config(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._config
    
    def is_initialized(self) -> bool:
        return self._initialized


# Module-level singleton instance
_manager = _AgentManager()


# Public API
async def initialize_agent():
    """Initialize the agent singleton"""
    await _manager.initialize()


def get_agent():
    """Get the agent instance"""
    return _manager.agent


def get_model():
    """Get the model instance"""
    return _manager.model


def get_tools():
    """Get the tools instance"""
    return _manager.tools


def get_config():
    """Get the config instance"""
    return _manager.config

def is_agent_initialized() -> bool:
    """Check if agent is initialized"""
    return _manager.is_initialized()