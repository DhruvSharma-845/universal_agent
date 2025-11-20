from agent_manager import initialize_agent, get_agent, get_config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

from langchain.messages import HumanMessage

from contextlib import asynccontextmanager

from dto import ChatRequest

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    await initialize_agent()
    global agent
    agent = get_agent()
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/api/universal-agent/chat")
async def universal_agent_chat(request: ChatRequest):
    conversation_messages = [HumanMessage(content=message) for message in request.messages]
    result = await agent.ainvoke({"messages": conversation_messages}, config=get_config())
    last_message = result["messages"][-1]
    return {"messages": [last_message.content]}

def run_server():
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        timeout_keep_alive=300,  # Keep idle connections alive for 120 seconds (default: 5)
        timeout_graceful_shutdown=30,  # Wait up to 30 seconds for graceful shutdown (default: 0)
    )