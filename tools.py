import tomllib
import os

from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# MCP client
from langchain_mcp_adapters.client import MultiServerMCPClient

from a2a_client import A2AClient
from dto import MessageDetail

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

# Initialize the A2A client for code-mentor-rag agent
code_mentor_client = A2AClient(
    base_url="http://localhost:8001",
    agent_id="code-mentor-agent"
)

@tool
async def analyze_codebase(query: str) -> str:
    """
    Analyzes a codebase using semantic search through git commits and code changes.
    Use this tool when the user asks questions about code implementations, git history,
    or wants to understand how specific features were implemented.
    
    Args:
        query: The analysis query or question about the codebase
        
    Returns:
        Analysis results from the code mentor agent
    """
    try:
        # Create a message for the code mentor agent
        messages = [
            MessageDetail(
                role="user",
                content=query
            )
        ]
        
        # Make A2A request to code-mentor-rag agent
        response = await code_mentor_client.invoke(
            messages=messages,
            thread_id="codebase_analysis",  # You can make this dynamic if needed
            user_id="system"  # Or use the actual user_id from context
        )
        
        # Check if the response is successful
        if response.error:
            return f"Error from code mentor agent: {response.error.get('message', 'Unknown error')}"
        
        # Extract the assistant's response
        if response.result and "messages" in response.result:
            result_messages = response.result["messages"]
            # Find the last assistant message
            for msg in reversed(result_messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "No response content")
        
        return "No response from code mentor agent"
        
    except Exception as e:
        return f"Error communicating with code mentor agent: {str(e)}"

async def getTools():
    # mcpServersConfig = getMCPServersConfig()
    # client = MultiServerMCPClient(mcpServersConfig)
    # tools = await client.get_tools()

    tools = []

    mathematical_tools = [add_numbers, multiply]
    tools.extend(mathematical_tools)

    # tools.append(getWikipediaTool())

    tools.append(analyze_codebase)

    return tools