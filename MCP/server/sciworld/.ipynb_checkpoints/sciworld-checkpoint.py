from typing import Any, Dict, Optional
import logging
import time
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import sys

# 初始化 MCP 服务器
mcp = FastMCP("science_world")

# 确保导入路径正确
sys.path.insert(0, str(Path(__file__).parent))
from agentenv_sciworld.agentenv_sciworld.model import *
from agentenv_sciworld.agentenv_sciworld.environment import server

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

@mcp.tool()
async def hello() -> str:
    """服务健康检查"""
    return "This is environment ScienceWorld."

@mcp.tool()
async def create_env() -> Dict[str, Any]:
    """创建新环境"""
    try:
        result = server.create()
        return {
            "status": "success",
            "data": result,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def step_env(env_id: int, action: str) -> Dict[str, Any]:
    """在环境中执行动作"""
    try:
        result = server.step(env_id, action)
        return {
            "status": "success",
            "data": result,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def reset_env(env_id: int, data_idx: Optional[int] = None) -> Dict[str, Any]:
    """重置环境"""
    try:
        result = server.reset(env_id, data_idx)
        return {
            "status": "success",
            "data": result,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def get_env_observation(env_id: int) -> Dict[str, Any]:
    """获取环境观察"""
    try:
        observation = server.get_observation(env_id)
        return {
            "status": "success",
            "observation": observation,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def get_action_hints(env_id: int) -> Dict[str, Any]:
    """获取动作提示"""
    try:
        hints = server.get_action_hint(env_id)
        return {
            "status": "success",
            "hints": hints,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def get_env_goals(env_id: int) -> Dict[str, Any]:
    """获取环境目标"""
    try:
        goals = server.get_goals(env_id)
        return {
            "status": "success",
            "goals": goals,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def get_env_details(env_id: int) -> Dict[str, Any]:
    """获取环境详细信息"""
    try:
        details = server.get_detailed_info(env_id)
        return {
            "status": "success",
            "details": details,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(transport='stdio')