from datetime import datetime
from typing import Any, List, Literal, Optional, Dict
from pydantic import BaseModel

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