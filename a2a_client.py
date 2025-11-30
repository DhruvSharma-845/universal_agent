# a2a_client.py
import httpx
import uuid
from typing import List, Optional
from dto import MessageDetail, A2ARequest, A2AResponse

class A2AClient:
    """Client for making A2A requests to other agents"""
    
    def __init__(self, base_url: str, agent_id: str, timeout: int = 300):
        self.base_url = base_url
        self.agent_id = agent_id
        self.timeout = timeout
        # self._agent_card: Optional[AgentCard] = None
    
    # async def get_agent_card(self) -> AgentCard:
    #     """Fetch the agent card from the remote agent"""
    #     if self._agent_card:
    #         return self._agent_card
            
    #     async with httpx.AsyncClient(timeout=self.timeout) as client:
    #         response = await client.get(f"{self.base_url}/.well-known/agent-card.json")
    #         response.raise_for_status()
    #         self._agent_card = AgentCard(**response.json())
    #         return self._agent_card
    
    async def invoke(
        self, 
        messages: List[MessageDetail], 
        thread_id: str, 
        user_id: str,
        method: str = "invoke"
    ) -> A2AResponse:
        request = A2ARequest(
            method=method,
            id=str(uuid.uuid4()),
            params={
                "messages": [msg.model_dump() for msg in messages],
                "thread_id": thread_id,
                "user_id": user_id,
                "config": {}
            }
        )
        
        endpoint_url = f"{self.base_url}/a2a/{self.agent_id}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                endpoint_url,
                json=request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return A2AResponse(**response.json())
