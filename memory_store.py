import asyncio
import uuid
from langchain_ollama import OllamaEmbeddings
from langgraph.store.memory import InMemoryStore

from utils import getValueFromConfig

class _MemoryStore:
    """Internal memory store (not exposed directly)"""
    
    def __init__(self):
        self._memory_store = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the memory store"""
        async with self._lock:
            if not self._initialized:
                memory_embedding_model = getValueFromConfig("memory", "memory_embedding_model")
                embeddings = OllamaEmbeddings(model=memory_embedding_model)
                self._memory_store = InMemoryStore(index={
                    "embed": embeddings,  # Embedding provider
                    "dims": 768,                              # Embedding dimensions
                    "fields": ["memory"]              # Fields to embed
                })
                self._initialized = True
                print("Memory store initialized successfully!")
    
    async def cleanup(self):
        """Cleanup the memory store"""
        async with self._lock:
            if self._initialized:
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

def getMemoriesForUserBasedOnQuery(user_id: str, query: str, limit: int = 3):
    # Namespace the memory
    namespace = (user_id, "memories")

    # Search based on the most recent message
    memories = get_memory_store().search(
        namespace,
        query=query,
        limit=limit
    )
    return [d.value["memory"] for d in memories]

def updateMemoryForUser(user_id: str, memory: str):
    # Namespace the memory
    namespace = (user_id, "memories")

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # We create a new memory
    get_memory_store().put(namespace, memory_id, {"memory": memory}, index=["memory"])