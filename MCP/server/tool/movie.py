from typing import List, Dict, Any
import logging
import time
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path
# 项目路径设置
current_dir = str(Path(__file__).parent.absolute())
sys.path.insert(0, current_dir)
from agentenv_movie.movie_environment import movie_env_server
from agentenv_movie.movie_model import *
from agentenv_movie.movie_utils import debug_flg
project_root_1 = Path(__file__).parent.parent
sys.path.append(str(project_root_1))
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("movie")

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# 速率限制配置
RATE_LIMIT = {
    "per_second": 10,
    "per_minute": 100
}

class RateLimiter:
    def __init__(self):
        self.counters = {
            "second": {"count": 0, "last_reset": time.time()},
            "minute": {"count": 0, "last_reset": time.time()}
        }
    
    def check(self) -> None:
        """检查并更新速率限制"""
        now = time.time()
        
        # 秒级检查
        if now - self.counters["second"]["last_reset"] > 1:
            self.counters["second"]["count"] = 0
            self.counters["second"]["last_reset"] = now
            
        # 分钟级检查
        if now - self.counters["minute"]["last_reset"] > 60:
            self.counters["minute"]["count"] = 0
            self.counters["minute"]["last_reset"] = now
        
        if (self.counters["second"]["count"] >= RATE_LIMIT["per_second"] or
            self.counters["minute"]["count"] >= RATE_LIMIT["per_minute"]):
            raise RuntimeError("Rate limit exceeded")
        
        self.counters["second"]["count"] += 1
        self.counters["minute"]["count"] += 1

rate_limiter = RateLimiter()

@mcp.tool()
async def health_check() -> str:
    """服务健康检查（原/端点）"""
    return "ok"

@mcp.tool()
async def list_environments() -> List[int]:
    """列出所有环境ID（原/list_envs端点）"""
    try:
        rate_limiter.check()
        return list(movie_env_server.env.keys())
    except Exception as e:
        logging.error(f"list_environments error: {str(e)}")
        raise

@mcp.tool()
async def create_environment(env_id: int) -> Dict[str, Any]:
    """创建新环境（原/create端点）"""
    try:
        rate_limiter.check()
        env_idx = movie_env_server.create(env_id)
        return {
            "status": "success",
            "env_idx": env_idx,
            "timestamp": time.time()
        }
    except Exception as e:
        logging.error(f"create_environment error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
async def step_environment(env_idx: int, action: str) -> Dict[str, Any]:
    """执行环境步进（原/step端点）"""
    try:
        rate_limiter.check()
        observation, reward, done, _ = movie_env_server.step(env_idx, action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "env_idx": env_idx
        }
    except Exception as e:
        logging.error(f"step_environment error: {str(e)}")
        raise

@mcp.tool()
async def get_observation(env_idx: int) -> Dict[str, Any]:
    """获取环境观察（原/observation端点）"""
    try:
        rate_limiter.check()
        obs = movie_env_server.observation(env_idx)
        return {
            "env_idx": env_idx,
            "observation": obs,
            "timestamp": time.time()
        }
    except Exception as e:
        logging.error(f"get_observation error: {str(e)}")
        raise

@mcp.tool()
async def reset_environment(env_idx: int, env_id: int) -> Dict[str, Any]:
    """重置环境（原/reset端点）"""
    try:
        rate_limiter.check()
        movie_env_server.reset(env_idx, env_id)
        obs = movie_env_server.observation(env_idx)
        return {
            "env_idx": env_idx,
            "new_observation": obs,
            "reset_id": env_id,
            "status": "success"
        }
    except Exception as e:
        logging.error(f"reset_environment error: {str(e)}")
        raise

if __name__ == "__main__":
    # 启动MCP服务器（支持stdio和websocket）
    mcp.run(
        transport='stdio'
    )