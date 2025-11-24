from agent_manager import cleanup_agent, initialize_agent
from memory_store import cleanup_memory_store, initialize_memory_store
from tools_manager import cleanup_tools, initialize_tools


async def bootstrap_app():
    await initialize_memory_store()
    await initialize_tools()
    await initialize_agent()

async def destroy_app():
    await cleanup_agent()
    await cleanup_tools()
    await cleanup_memory_store()