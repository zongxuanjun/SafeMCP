from typing import List, Dict, Any, Optional
import logging
import time
import subprocess
from pathlib import Path
import sys
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 项目路径设置
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv()

# 导入原有模块
from agentenv_webarena.agentenv_webarena.environment import webarena_env_server
from agentenv_webarena.agentenv_webarena.utils import debug_flg

# 初始化 MCP 服务器
mcp = FastMCP("webarena")

# 增强日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class WebArenaRateLimiter:
    """增强版速率限制器，支持环境级限流"""
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
        now = time.time()
        
        # 全局限制检查
        if now - self.counters['global']['last_reset'] > 1:
            self.counters['global']['count'] = 0
            self.counters['global']['last_reset'] = now
        self.counters['global']['count'] += 1
        
        # 环境级限制检查
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

rate_limiter = WebArenaRateLimiter()

@mcp.tool()
async def health_check() -> Dict[str, str]:
    """服务健康检查（对应原/端点）"""
    return {"status": "ok", "service": "webarena"}

@mcp.tool()
async def list_environments() -> Dict[str, Any]:
    """列出所有活跃环境（原/list_envs端点）"""
    try:
        rate_limiter.check()
        env_ids = list(webarena_env_server.env.keys())
        return {
            "count": len(env_ids),
            "environments": env_ids,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"List environments failed: {str(e)}")
        raise

@mcp.tool()
async def create_environment() -> Dict[str, Any]:
    """创建新环境（原/create端点）"""
    try:
        rate_limiter.check()
        # 保留原有的auto_login逻辑
        subprocess.run(["python", "browser_env/auto_login.py"], check=True)
        env_idx = webarena_env_server.create()
        logger.info(f"Created new environment {env_idx}")
        return {
            "status": "created",
            "env_idx": env_idx,
            "active_envs": len(webarena_env_server.env)
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Auto-login failed: {str(e)}")
        return {
            "status": "error",
            "error_type": "AutoLoginError",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Create environment failed: {str(e)}")
        raise

@mcp.tool()
async def execute_action(
    env_idx: int,
    action: str,
    action_metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """执行动作（原/step端点）"""
    try:
        rate_limiter.check(env_idx)
        step_data = webarena_env_server.step(env_idx, action)
        
        logger.debug(f"Executed action in env {env_idx}: {action}")
        return {
            "observation": step_data[0],
            "reward": step_data[1],
            "terminated": step_data[2],
            "truncated": step_data[3],
            "info": step_data[4],
            "env_idx": env_idx,
            "action": action
        }
    except Exception as e:
        logger.error(f"Action failed in env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_observation(
    env_idx: int,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """获取环境观察（原/observation端点）"""
    try:
        rate_limiter.check(env_idx)
        obs = webarena_env_server.observation(env_idx)
        response = {
            "env_idx": env_idx,
            "observation": obs,
            "timestamp": time.time()
        }
        
        if include_metadata:
            obs_meta = webarena_env_server.observation_metadata(env_idx)
            response["metadata"] = obs_meta.get("text", "")
            
        return response
    except Exception as e:
        logger.error(f"Get observation failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def reset_environment(
    env_idx: int,
    seed: int = 0,
    config_idx: Optional[int] = None
) -> Dict[str, Any]:
    """重置环境（原/reset端点）"""
    try:
        rate_limiter.check(env_idx)
        options = {
            "config_file": f"./config_files/{config_idx}.json"
        } if config_idx else {}
        
        obs, info = webarena_env_server.reset(env_idx, seed, options)
        
        logger.info(f"Reset environment {env_idx} with config {config_idx}")
        return {
            "status": "reset",
            "env_idx": env_idx,
            "observation": obs.get("text", ""),
            "config_used": options.get("config_file", "default"),
            "info": info
        }
    except Exception as e:
        logger.error(f"Reset failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def close_environment(env_idx: int) -> Dict[str, Any]:
    """关闭环境（原/close端点）"""
    try:
        rate_limiter.check(env_idx)
        webarena_env_server.close(env_idx)
        logger.info(f"Closed environment {env_idx}")
        return {
            "status": "closed",
            "env_idx": env_idx,
            "active_envs": len(webarena_env_server.env)
        }
    except Exception as e:
        logger.error(f"Close failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_service_metrics() -> Dict[str, Any]:
    """获取服务指标（新增功能）"""
    try:
        rate_limiter.check()
        return {
            "active_environments": len(webarena_env_server.env),
            "rate_limits": {
                "global": rate_limiter.limits['global'],
                "per_env": rate_limiter.limits['per_env']
            },
            "service": "webarena",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Get metrics failed: {str(e)}")
        raise

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(
        transport='stdio'
    )