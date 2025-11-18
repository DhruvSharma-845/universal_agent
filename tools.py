import tomllib
import os

from langchain.tools import tool

# MCP client
from langchain_mcp_adapters.client import MultiServerMCPClient

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

def getMCPServersConfig():
    config_path = "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    mcp_servers = config.get("mcp_servers", {})
    # Override environment variables with actual env vars if they exist
    for server_name, server_config in mcp_servers.items():
        if "env" in server_config:
            for env_key in server_config["env"].keys():
                # Check if environment variable is set, use it if available
                env_value = os.getenv(env_key)
                if env_value:
                    server_config["env"][env_key] = env_value
    
    return mcp_servers

async def getTools():
    mcpServersConfig = getMCPServersConfig()
    client = MultiServerMCPClient(mcpServersConfig)
    tools = await client.get_tools()
    tools.append(multiply)
    return tools