from agent_manager import cleanup_agent, initialize_agent


async def bootstrap_app():
    await initialize_agent()

async def destroy_app():
    await cleanup_agent()
