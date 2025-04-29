from typing import Any, Dict, Optional
import logging
import time
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import sys

# 初始化 MCP 服务器
mcp = FastMCP("baby_ai")

# 确保导入路径正确
sys.path.insert(0, str(Path(__file__).parent))
from model import StepRequestBody, ResetRequestBody
from environment import server

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

@mcp.tool()
async def hello() -> str:
    """服务健康检查"""
    return "This is environment BabyAI."

@mcp.tool()
async def create_env() -> Dict[str, Any]:
    """创建新BabyAI环境"""
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
async def step_env(body: StepRequestBody) -> Dict[str, Any]:
    """在BabyAI环境中执行动作"""
    try:
        result = server.step(body.id, body.action)
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
async def reset_env(body: ResetRequestBody) -> Dict[str, Any]:
    """重置BabyAI环境"""
    try:
        result = server.reset(body.id, body.data_idx)
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

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(transport='stdio')
    