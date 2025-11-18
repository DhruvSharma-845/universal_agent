from model import getModel
from tools import getTools
from agent import getAgent


from langchain.messages import HumanMessage

import asyncio


async def main():
    # Tools Definition end here

    model = getModel()

    tools = await getTools()

    agent = getAgent(model, tools)

    # messages = [HumanMessage(content="Get the content of the wiki page: https://wiki.corp.adobe.com/display/lc/Debugging+setup+for+IC+Editor")]
    # messages = await agent.ainvoke({"messages": messages})
    # for m in messages["messages"]:
    #     m.pretty_print()

    print("Welcome to the AI Agent!")
    print("Type 'exit', 'quit', or 'q' to end the conversation.\n")
    
    conversation_messages = []
    
    while True:
        # Get user input
        user_query = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            print("Please enter a query.")
            continue
        
        # Add user message to conversation
        conversation_messages.append(HumanMessage(content=user_query))
        
        # Get agent response
        result = await agent.ainvoke({"messages": conversation_messages})
        
        # Update conversation with all messages (including tool calls)
        conversation_messages = result["messages"]
        
        # Print only the last assistant message
        print("\nAssistant:", end=" ")
        last_message = conversation_messages[-1]
        last_message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
