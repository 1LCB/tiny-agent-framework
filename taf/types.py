from typing import TypedDict, Literal

class AgentChunkResponse(TypedDict):
    origin: str
    content: str
    type: Literal["response", "reasoning", "tool_call", "tool_call_result"]
    metadata: dict

    def __str__(self):
        return self.get("content", None)
