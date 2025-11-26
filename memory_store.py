import asyncio
import uuid
import os
from langchain.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from prompt_templates import MEMORY_CREATE_PROMPT
from utils import getValueFromConfig

class _MemoryStore:
    """Internal memory store (not exposed directly)"""
    
    def __init__(self):
        self._memory_store = None
        self._initialized = False
        self._persist_dir = "./faiss_memory"
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the memory store"""
        async with self._lock:
            if not self._initialized:
                memory_embedding_model = getValueFromConfig("memory", "memory_embedding_model")
                embeddings = OllamaEmbeddings(model=memory_embedding_model)

                # Create persist directory if it doesn't exist
                os.makedirs(self._persist_dir, exist_ok=True)

                # Try to load existing FAISS index
                index_path = os.path.join(self._persist_dir, "index")
                if os.path.exists(f"{index_path}.faiss"):
                    print("Loading existing FAISS index...")
                    self._memory_store = FAISS.load_local(
                        self._persist_dir,
                        embeddings,
                        allow_dangerous_deserialization=True  # Required for pickle loading
                    )
                    print("FAISS memory store loaded successfully!")
                else:
                    print("Creating new FAISS index...")
                    # Create a new FAISS index with a dummy document
                    dummy_doc = Document(
                        page_content="initialization",
                        metadata={"user_id": "system", "namespace": "init"}
                    )
                    self._memory_store = FAISS.from_documents([dummy_doc], embeddings)
                    self._save_index()
                    print("FAISS memory store created successfully!")

                self._initialized = True
                print("Memory store initialized successfully!")
    
    def _save_index(self):
        """Save FAISS index to disk"""
        if self._memory_store:
            self._memory_store.save_local(self._persist_dir)
    
    async def cleanup(self):
        """Cleanup the memory store"""
        async with self._lock:
            if self._initialized:
                self._save_index()
                self._memory_store = None
                self._initialized = False
                print("Memory store cleaned up successfully!")
    
    @property
    def memory_store(self):
        """Get the memory store instance"""
        return self._memory_store

# Module-level singleton instance
_manager = _MemoryStore()

# Public API
async def initialize_memory_store():
    await _manager.initialize()

async def cleanup_memory_store():
    await _manager.cleanup()

def get_memory_store():
    return _manager.memory_store

def _save_memory_store():
    """Helper to save the FAISS index"""
    _manager._save_index()

def getMemoriesForUserBasedOnQuery(user_id: str, query: str, limit: int = 3):
    """Search memories using FAISS similarity search"""
    store = get_memory_store()
    if not store:
        return []
    
    # Search with more results to filter by user
    results = store.similarity_search_with_score(
        query=query,
        k=limit * 10,  # Fetch more to filter by user_id
        filter={"user_id": user_id, "namespace": "memories"}
    )
    
    # Filter and limit results
    filtered_results = [
        doc.page_content for doc, score in results
        if doc.metadata.get("user_id") == user_id and doc.metadata.get("namespace") == "memories"
    ][:limit]
    
    return filtered_results

async def updateMemoryForUser(user_id: str, messages, memory_creator):
    # Namespace the memory
    namespace = (user_id, "memories")

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # Creating the memory create prompt
    memory_create_prompt_template = PromptTemplate.from_template(MEMORY_CREATE_PROMPT)
    conversation_messages = "\n".join(["user: " + msg.content if isinstance(msg, HumanMessage) else "assistant: " + msg.content for msg in messages])
    
    memory_create_prompt = memory_create_prompt_template.invoke({"conversation_messages": conversation_messages})
    
    memory_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant designed to create concise summaries (short-term memories) of conversations."),
        ("human", "{input}")
    ])
    prompt = memory_prompt_template.format_messages(input=memory_create_prompt.to_string())
    # Creating the memory
    stringified_memory = await memory_creator(prompt)

    doc = Document(
        page_content=stringified_memory,
        metadata={
            "user_id": user_id,
            "namespace": "memories",
            "memory_id": memory_id
        }
    )
    get_memory_store().add_documents([doc])
    
    # Save to disk after adding
    _save_memory_store()
    print(f"Memory saved and persisted to disk: \n\n {stringified_memory} \n\n")