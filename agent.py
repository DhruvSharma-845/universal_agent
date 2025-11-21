# For state management
from langchain.messages import AnyMessage
from langchain_core.messages import trim_messages
from typing_extensions import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

from typing import Literal

from langchain.messages import SystemMessage, ToolMessage

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def getLLMCallWithModel(model_with_tools):
    def llm_call(state: dict):
        """LLM decides whether to call a tool or not"""

        # Trim messages to stay under token limit
        # Keep the most recent messages that fit within max_tokens
        trimmed_messages = trim_messages(
            state["messages"],
            max_tokens=64000,  # Adjust based on your model's context window
            strategy="last",  # Keep the most recent messages
            token_counter=model_with_tools,  # Use the model to count tokens
            include_system=False,  # System message handled separately
            allow_partial=False,  # Don't cut messages in half
        )

        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing operations on a private documentation tool: Adobe Wiki. Use the tools provided to you to perform the operations."
                        )
                    ]
                    + trimmed_messages
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }
    return llm_call

def getToolNode(tools):
    tools_by_name = {tool.name: tool for tool in tools}
    
    async def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    return tool_node


def getAgent(model, tools, checkpointer):

    model_with_tools = model.bind_tools(tools)

    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", getLLMCallWithModel(model_with_tools))
    agent_builder.add_node("tool_node", getToolNode(tools))

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", END: END}
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile(checkpointer=checkpointer)

    return agent
