from langchain.messages import HumanMessage, ToolMessage
def get_messages_details(conv_messages, thread_id):
    # Extract and format messages
    messages = []
    for msg in conv_messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        else:
            role = "assistant"
        
        content = msg.content
        if getattr(msg, "tool_calls", None) is not None:
            for tool_call in msg.tool_calls:
                content += f"Tool call: {tool_call['name']} with args: {tool_call['args']}"

        messages.append({
            "id": thread_id,
            "role": role,
            "content": content,
        })
    return messages