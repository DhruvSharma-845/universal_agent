import json
from langchain.messages import HumanMessage

from agent_manager import get_agent
from dto import ConversationHistory
from utils import get_messages_details

async def get_state(config: dict) -> dict:
    return await get_agent().aget_state(config)

async def chat_with_agent(thread_id: str, input_messages: list[str]) -> ConversationHistory:
    conversation_messages = [HumanMessage(content=message) for message in input_messages]
    # Use thread_id from request for conversation tracking
    config = {"configurable": {"thread_id": thread_id}}
    result = await get_agent().ainvoke({"messages": conversation_messages}, config=config)
    # Get the state from the checkpointer
    state = await get_state(config)
    
    if not state or not state.values.get("messages"):
        return ConversationHistory(thread_id=thread_id, messages=[])

    filtered_messages = []
    for msg in state.values["messages"][::-1]:
        if isinstance(msg, HumanMessage):
            break
        filtered_messages.append(msg)

    messages = get_messages_details(filtered_messages[::-1], thread_id)
        
    return ConversationHistory(thread_id=thread_id, messages=messages)

async def chat_with_agent_stream_generator(thread_id: str, input_messages: list[str]):
    conversation_messages = [HumanMessage(content=message) for message in input_messages]
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async for chunk in get_agent().astream({"messages": conversation_messages}, config, stream_mode="updates"):
            print(chunk)
            messages = get_messages_details(chunk["llm_call"]["messages"] if "llm_call" in chunk else chunk["tool_node"]["messages"], thread_id)
            conv_history = ConversationHistory(thread_id=thread_id, messages=messages)
            yield f"data: {conv_history.model_dump_json()}\n\n"
        
    except Exception as e:
        print(e)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    

async def get_conversation_history_from_agent(thread_id: str) -> ConversationHistory:
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get the state from the checkpointer
    state = await get_state(config)
    
    if not state or not state.values.get("messages"):
        return ConversationHistory(thread_id=thread_id, messages=[])
    
    # Extract and format messages
    messages = get_messages_details(state.values["messages"], thread_id)
    
    return ConversationHistory(thread_id=thread_id, messages=messages)
    
        