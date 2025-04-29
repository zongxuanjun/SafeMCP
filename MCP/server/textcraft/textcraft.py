from typing import Any, Dict, List, Optional
import logging
import time
from pathlib import Path
import sys
from dotenv import load_dotenv
from agentenv_textcraft_main.agentenv_textcraft.model import *
from agentenv_textcraft_main.agentenv_textcraft.env_wrapper import server
# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv()

# 初始化MCP服务器
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("textcraft")

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 速率限制配置
RATE_LIMIT = {
    "per_second": 10,
    "per_minute": 100
}

request_count = {
    "second": 0,
    "minute": 0,
    "last_reset": time.time()
}

def check_rate_limit():
    """速率限制检查"""
    now = time.time()
    elapsed = now - request_count["last_reset"]
    
    if elapsed > 60:
        request_count["minute"] = 0
        request_count["second"] = 0
        request_count["last_reset"] = now
    elif elapsed > 1:
        request_count["second"] = 0
    
    if (request_count["second"] >= RATE_LIMIT["per_second"] or
        request_count["minute"] >= RATE_LIMIT["per_minute"]):
        raise RuntimeError("Rate limit exceeded")
    
    request_count["second"] += 1
    request_count["minute"] += 1

# 工具函数注册
@mcp.tool()
async def create_env(commands: List[str], goal: str) -> Dict[str, Any]:
    """创建新环境实例"""
    try:
        check_rate_limit()
        env_id = server.create(commands, goal)
        logger.info(f"Created environment {env_id}")
        return {"env_id": env_id, "status": "success"}
    except Exception as e:
        logger.error(f"Create failed: {str(e)}")
        return {"error": str(e), "status": "error"}

@mcp.tool()
async def step_action(env_id: int, action: str) -> Dict[str, Any]:
    """执行环境动作"""
    try:
        check_rate_limit()
        logger.debug(f"Step on {env_id} with action: {action}")
        result = server.step(env_id, action)
        return {
            "observation": result[0],
            "reward": result[1],
            "terminated": result[2],
            "truncated": result[3],
            "info": result[4]
        }
    except Exception as e:
        logger.error(f"Step failed on {env_id}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def reset_env(env_id: int, data_idx: Optional[int] = None) -> Dict[str, Any]:
    """重置环境状态"""
    try:
        check_rate_limit()
        logger.info(f"Resetting environment {env_id}")
        observation, info = server.reset(env_id, data_idx)
        return {
            "observation": observation,
            "info": info
        }
    except Exception as e:
        logger.error(f"Reset failed on {env_id}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_observation(env_id: int) -> Dict[str, Any]:
    """获取环境观察"""
    try:
        check_rate_limit()
        obs = server.get_observation(env_id)
        return {"observation": obs}
    except Exception as e:
        logger.error(f"Get observation failed on {env_id}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_commands(env_id: int) -> Dict[str, Any]:
    """获取可用命令"""
    try:
        check_rate_limit()
        commands = server.get_commands(env_id)
        return {"commands": commands}
    except Exception as e:
        logger.error(f"Get commands failed on {env_id}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_goal(env_id: int) -> Dict[str, Any]:
    """获取环境目标"""
    try:
        check_rate_limit()
        goal = server.get_goal(env_id)
        return {"goal": goal}
    except Exception as e:
        logger.error(f"Get goal failed on {env_id}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_detailed_info(env_id: int) -> Dict[str, Any]:
    """获取环境详细信息"""
    try:
        check_rate_limit()
        details = server.get_detailed_info(env_id)
        return {"details": details}
    except Exception as e:
        logger.error(f"Get details failed on {env_id}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def list_envs() -> Dict[str, Any]:
    """列出所有活跃环境"""
    try:
        check_rate_limit()
        envs = list(server.env.keys())
        return {"environments": envs}
    except Exception as e:
        logger.error(f"List environments failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # 使用相对导入从当前包导入server
    
    # 启动MCP服务器
    mcp.run(transport='stdio')