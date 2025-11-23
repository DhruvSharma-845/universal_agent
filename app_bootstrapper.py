from agent_manager import cleanup_agent, initialize_agent
from tools_manager import cleanup_tools, initialize_tools


async def bootstrap_app():
    await initialize_tools()
    await initialize_agent()

async def destroy_app():
    await cleanup_agent()
    await cleanup_tools()