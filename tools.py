import tomllib
import os

from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# MCP client
from langchain_mcp_adapters.client import MultiServerMCPClient

# [START] Mathematical tools
@tool
def add_numbers(x: int, y: int) -> int:
    """Adds two numbers and returns the sum.
    
    Args:
        x: First int
        y: Second int
    """
    return x + y

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b` and returns the product.

    Args:
        a: First int
        b: Second int
    """
    return a * b

# [END] Mathematical tools

def getWikipediaTool():
    # Configure the wrapper to fetch 1 result with up to 1000 characters
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=4000
    )

    # Initialize the Wikipedia query tool
    return WikipediaQueryRun(api_wrapper=api_wrapper)


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

    mathematical_tools = [add_numbers, multiply]
    tools.extend(mathematical_tools)

    tools.append(getWikipediaTool())

    return tools