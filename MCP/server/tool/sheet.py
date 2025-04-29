from typing import List, Dict, Any
import logging
import time
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

# 导入原有模块
from agentenv_sheet.sheet_environment import sheet_env_server
from agentenv_sheet.sheet_model import *
from agentenv_sheet.sheet_utils import debug_flg

# 项目路径设置
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("sheet")

# 配置结构化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 增强版速率限制器
class RateLimiter:
    def __init__(self):
        self.counters = {
            'global': {'count': 0, 'last_reset': time.time()},
            'per_env': {}
        }
        self.limits = {
            'global': {'per_second': 20, 'per_minute': 200},
            'per_env': {'per_second': 5, 'per_minute': 50}
        }

    def check(self, env_idx: int = None) -> None:
        """检查全局和针对特定环境的速率限制"""
        now = time.time()
        
        # 全局限制检查
        if now - self.counters['global']['last_reset'] > 1:
            self.counters['global']['count'] = 0
            self.counters['global']['last_reset'] = now
        self.counters['global']['count'] += 1
        
        # 环境特定限制检查
        if env_idx is not None:
            if env_idx not in self.counters['per_env']:
                self.counters['per_env'][env_idx] = {'count': 0, 'last_reset': now}
            
            if now - self.counters['per_env'][env_idx]['last_reset'] > 1:
                self.counters['per_env'][env_idx]['count'] = 0
                self.counters['per_env'][env_idx]['last_reset'] = now
            self.counters['per_env'][env_idx]['count'] += 1

        # 验证限制
        if (self.counters['global']['count'] > self.limits['global']['per_second'] or
            (env_idx and self.counters['per_env'][env_idx]['count'] > self.limits['per_env']['per_second'])):
            raise RuntimeError("Rate limit exceeded")

rate_limiter = RateLimiter()

@mcp.tool()
async def health_check() -> Dict[str, str]:
    """服务健康检查（原/端点）"""
    return {"status": "ok", "service": "sheet_env"}

@mcp.tool()
async def list_active_environments() -> Dict[str, Any]:
    """列出所有活跃环境（原/list_envs端点）"""
    try:
        rate_limiter.check()
        env_ids = list(sheet_env_server.env.keys())
        return {
            "count": len(env_ids),
            "environments": env_ids,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to list environments: {str(e)}")
        raise

@mcp.tool()
async def create_sheet_environment(env_id: int) -> Dict[str, Any]:
    """创建新的表格环境（原/create端点）"""
    try:
        rate_limiter.check()
        env_idx = sheet_env_server.create(env_id)
        logger.info(f"Created new sheet environment {env_idx} for ID {env_id}")
        return {
            "status": "created",
            "env_idx": env_idx,
            "env_id": env_id,
            "active_envs": len(sheet_env_server.env)
        }
    except Exception as e:
        logger.error(f"Failed to create environment: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "error_type": e.__class__.__name__
        }

@mcp.tool()
async def execute_sheet_action(
    env_idx: int, 
    action: str,
    action_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """在表格环境中执行动作（原/step端点）"""
    try:
        rate_limiter.check(env_idx)
        observation, reward, done, info = sheet_env_server.step(env_idx, action)
        
        logger.debug(f"Action executed in env {env_idx}: {action}")
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "env_idx": env_idx,
            "action": action
        }
    except Exception as e:
        logger.error(f"Failed to execute action in env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_sheet_observation(env_idx: int) -> Dict[str, Any]:
    """获取表格环境当前状态（原/observation端点）"""
    try:
        rate_limiter.check(env_idx)
        obs = sheet_env_server.observation(env_idx)
        return {
            "env_idx": env_idx,
            "observation": obs,
            "observation_type": type(obs).__name__,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get observation for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def reset_sheet_environment(
    env_idx: int,
    new_env_id: int = None
) -> Dict[str, Any]:
    """重置表格环境（原/reset端点）"""
    try:
        rate_limiter.check(env_idx)
        sheet_env_server.reset(env_idx, new_env_id)
        obs = sheet_env_server.observation(env_idx)
        
        logger.info(f"Reset environment {env_idx} with new ID {new_env_id}")
        return {
            "status": "reset",
            "env_idx": env_idx,
            "new_env_id": new_env_id,
            "new_observation": obs,
            "active_envs": len(sheet_env_server.env)
        }
    except Exception as e:
        logger.error(f"Failed to reset env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_environment_stats() -> Dict[str, Any]:
    """获取环境服务统计信息（新增功能）"""
    try:
        rate_limiter.check()
        return {
            "active_environments": len(sheet_env_server.env),
            "uptime": time.time() - mcp.start_time,
            "rate_limits": rate_limiter.limits,
            "service": "sheet_env"
        }
    except Exception as e:
        logger.error(f"Failed to get service stats: {str(e)}")
        raise

if __name__ == "__main__":
    # 启动MCP服务器，支持多种传输协议
    mcp.run(
        transport='stdio'
    )