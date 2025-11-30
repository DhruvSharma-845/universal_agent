from datetime import datetime
from typing import Any, List, Literal, Optional, Dict
import uuid
from pydantic import BaseModel, Field

class MessageDetail(BaseModel):
    id: Optional[str] = None
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    messages: list[MessageDetail]
    thread_id: str
    user_id: str

class ConversationHistory(BaseModel):
    thread_id: str
    messages: list[MessageDetail]
    user_id: str

class A2ARequest(BaseModel):
    """
    A2A protocol request format (JSON-RPC style).
    Sent to /a2a/{assistant_id} endpoint.
    """
    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["invoke", "stream"] = "invoke"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request parameters
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request parameters including messages and config"
    )
    
    # Core fields within params
    def __init__(self, **data):
        super().__init__(**data)
        if "params" not in data or not data["params"]:
            self.params = {
                "messages": [],
                "config": {},
                "thread_id": None,
                "user_id": None
            }
    
    @property
    def messages(self) -> List[MessageDetail]:
        """Get messages from params"""
        msgs = self.params.get("messages", [])
        return [MessageDetail(**m) if isinstance(m, dict) else m for m in msgs]
    
    @property
    def thread_id(self) -> Optional[str]:
        """Get thread_id from params"""
        return self.params.get("thread_id")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get config from params"""
        return self.params.get("config", {})

    @property
    def user_id(self) -> Optional[str]:
        """Get user_id from params"""
        return self.params.get("user_id")

class A2AResponse(BaseModel):
    """A2A protocol response format (JSON-RPC style)"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    
    # Either result or error
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None