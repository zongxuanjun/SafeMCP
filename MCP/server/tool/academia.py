from typing import Any, Dict, List
import logging
import time
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

# 导入原有模块
from agentenv_academia.academia_environment import academia_env_server
from agentenv_academia.academia_model import *
from agentenv_academia.academia_utils import debug_flg

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent  # 根据实际结构调整
sys.path.append(str(project_root))

# 加载环境变量
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("academia")

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

logger = logging.getLogger(__name__)
# 速率限制配置
RATE_LIMIT = {
    "per_second": 5,
    "per_minute": 60
}

request_count = {
    "second": 0,
    "minute": 0,
    "last_reset": time.time(),
    "last_minute_reset": time.time()
}

def check_rate_limit():
    """检查并更新速率限制"""
    now = time.time()
    
    # 重置秒级计数器
    if now - request_count["last_reset"] > 1:
        request_count["second"] = 0
        request_count["last_reset"] = now
        
    # 重置分钟级计数器
    if now - request_count["last_minute_reset"] > 60:
        request_count["minute"] = 0
        request_count["last_minute_reset"] = now
    
    if (request_count["second"] >= RATE_LIMIT["per_second"] or 
        request_count["minute"] >= RATE_LIMIT["per_minute"]):
        raise Exception("Rate limit exceeded")
    
    request_count["second"] += 1
    request_count["minute"] += 1

@mcp.tool()
async def generate_ok() -> str:
    """Test connectivity"""
    return "ok"

@mcp.tool()
async def list_envs() -> List[int]:
    """List all environments"""
    try:
        check_rate_limit()
        return list(academia_env_server.env.keys())
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def create_env(create_query: CreateQuery) -> int:
    """Create a new environment"""
    try:
        check_rate_limit()
        env = academia_env_server.create(create_query.id)
        return env
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def step_action(step_query: StepQuery) -> StepResponse:
    """Execute an action in the environment"""
    try:
        check_rate_limit()
        observation, reward, done, _ = academia_env_server.step(
            step_query.env_idx, step_query.action
        )
        return StepResponse(observation=observation, reward=reward, done=done)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_observation(env_idx: int) -> str:
    """Get environment observation"""
    try:
        check_rate_limit()
        return academia_env_server.observation(env_idx)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def reset_env(reset_query: ResetQuery) -> str:
    """Reset environment"""
    try:
        check_rate_limit()
        academia_env_server.reset(reset_query.env_idx, reset_query.id)
        return academia_env_server.observation(reset_query.env_idx)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # 启动标准输入输出模式的MCP服务器
    mcp.run(transport='stdio')