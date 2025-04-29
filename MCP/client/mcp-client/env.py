from dataclasses import dataclass
from mcp import ClientSession, ToolResponse

@dataclass
class MCPEnvResponse:
    state: str
    reward: float
    done: bool

class MCPEnvClient:
    """通用MCP环境客户端"""
    def __init__(self, session: ClientSession):
        self.session = session

    async def reset(self, task_type: str, idx: int) -> str:
        """初始化指定类型的环境"""
        resp: ToolResponse = await self.session.call_tool(
            "env_reset",
            {"task_type": task_type, "index": idx}
        )
        return resp.content[0].text

    async def step(self, task_type: str, action: str) -> MCPEnvResponse:
        """执行环境动作"""
        resp: ToolResponse = await self.session.call_tool(
            "env_step",
            {"task_type": task_type, "action": action}
        )
        return MCPEnvResponse(
            state=resp.content[0].text,
            reward=resp.metadata.get("reward", 0.0),
            done=resp.metadata.get("done", False)
        )
