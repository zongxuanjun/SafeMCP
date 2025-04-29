from typing import List, Dict, Any, Optional
import logging
import time
from dataclasses import asdict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

# 导入原有模块
from agentenv_todo.todo_environment import todo_env_server
from agentenv_todo.todo_model import *
from agentenv_todo.todo_utils import debug_flg

# --- 初始化配置 ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("todo")

# 结构化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 增强功能模块 ---
class EnhancedRateLimiter:
    """增强型速率限制器，支持分级限流"""
    def __init__(self):
        self.limits = {
            'global': {'second': 30, 'minute': 300},
            'env_specific': {'second': 5, 'minute': 50}
        }
        self.reset_counters()
    
    def reset_counters(self):
        self.counters = {
            'global': {'second': 0, 'minute': 0, 'last_second': time.time(), 'last_minute': time.time()},
            'env': {}
        }
    
    def check(self, env_idx: Optional[int] = None) -> None:
        now = time.time()
        
        # 全局计数器
        if now - self.counters['global']['last_second'] > 1:
            self.counters['global']['second'] = 0
            self.counters['global']['last_second'] = now
        if now - self.counters['global']['last_minute'] > 60:
            self.counters['global']['minute'] = 0
            self.counters['global']['last_minute'] = now
            
        self.counters['global']['second'] += 1
        self.counters['global']['minute'] += 1
        
        # 环境级计数器
        if env_idx is not None:
            if env_idx not in self.counters['env']:
                self.counters['env'][env_idx] = {'second': 0, 'minute': 0, 'last_second': now, 'last_minute': now}
                
            if now - self.counters['env'][env_idx]['last_second'] > 1:
                self.counters['env'][env_idx]['second'] = 0
                self.counters['env'][env_idx]['last_second'] = now
            if now - self.counters['env'][env_idx]['last_minute'] > 60:
                self.counters['env'][env_idx]['minute'] = 0
                self.counters['env'][env_idx]['last_minute'] = now
                
            self.counters['env'][env_idx]['second'] += 1
            self.counters['env'][env_idx]['minute'] += 1
        
        # 检查限制
        if (self.counters['global']['second'] > self.limits['global']['second'] or
            self.counters['global']['minute'] > self.limits['global']['minute']):
            raise RuntimeError("Global rate limit exceeded")
            
        if (env_idx and 
            (self.counters['env'][env_idx]['second'] > self.limits['env_specific']['second'] or
             self.counters['env'][env_idx]['minute'] > self.limits['env_specific']['minute'])):
            raise RuntimeError(f"Environment {env_idx} rate limit exceeded")

rate_limiter = EnhancedRateLimiter()

# --- MCP工具函数 ---
@mcp.tool()
async def service_health() -> Dict[str, Any]:
    """服务健康检查（原/端点）"""
    return {
        "status": "healthy",
        "service": "todo_env",
        "version": mcp.version,
        "timestamp": time.time()
    }

@mcp.tool()
async def list_environments() -> Dict[str, Any]:
    """列出所有环境（原/list_envs端点）"""
    try:
        rate_limiter.check()
        env_ids = list(todo_env_server.env.keys())
        return {
            "count": len(env_ids),
            "active_environments": env_ids,
            "memory_usage": f"{sys.getsizeof(todo_env_server.env)/1024:.2f} KB"
        }
    except Exception as e:
        logger.error(f"List environments failed: {str(e)}")
        raise

@mcp.tool()
async def create_environment(env_id: int) -> Dict[str, Any]:
    """创建新环境（原/create端点）"""
    try:
        rate_limiter.check()
        env_idx = todo_env_server.create(env_id)
        logger.info(f"Created new todo environment {env_idx} for ID {env_id}")
        return {
            "operation": "create",
            "env_idx": env_idx,
            "env_id": env_id,
            "status": "success",
            "created_at": time.time()
        }
    except Exception as e:
        logger.error(f"Create environment failed for ID {env_id}: {str(e)}")
        return {
            "operation": "create",
            "status": "error",
            "error": str(e),
            "error_type": e.__class__.__name__
        }

@mcp.tool()
async def perform_action(
    env_idx: int,
    action: str,
    action_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """执行动作（原/step端点）"""
    try:
        rate_limiter.check(env_idx)
        observation, reward, done, _ = todo_env_server.step(env_idx, action)
        
        logger.debug(f"Action '{action}' performed in env {env_idx}")
        return {
            "env_idx": env_idx,
            "action": action,
            "observation": observation,
            "reward": reward,
            "done": done,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Action failed in env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_environment_state(env_idx: int) -> Dict[str, Any]:
    """获取环境状态（原/observation端点）"""
    try:
        rate_limiter.check(env_idx)
        obs = todo_env_server.observation(env_idx)
        return {
            "env_idx": env_idx,
            "observation": obs,
            "status": "success",
            "observation_type": type(obs).__name__,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Get state failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def reset_environment(
    env_idx: int,
    new_env_id: Optional[int] = None
) -> Dict[str, Any]:
    """重置环境（原/reset端点）"""
    try:
        rate_limiter.check(env_idx)
        todo_env_server.reset(env_idx, new_env_id)
        obs = todo_env_server.observation(env_idx)
        
        logger.info(f"Reset env {env_idx} with new ID {new_env_id}")
        return {
            "operation": "reset",
            "env_idx": env_idx,
            "new_env_id": new_env_id,
            "new_observation": obs,
            "status": "success",
            "reset_at": time.time()
        }
    except Exception as e:
        logger.error(f"Reset failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_service_metrics() -> Dict[str, Any]:
    """获取服务指标（新增功能）"""
    try:
        rate_limiter.check()
        return {
            "service": "todo_env",
            "uptime": time.time() - mcp.start_time,
            "active_envs": len(todo_env_server.env),
            "rate_limits": rate_limiter.limits,
            "memory_usage": f"{sys.getsizeof(todo_env_server.env)/1024:.2f} KB",
            "python_version": sys.version
        }
    except Exception as e:
        logger.error(f"Get metrics failed: {str(e)}")
        raise

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(
        transport='stdio' # 支持 'stdio'/'websocket'/'http'
    )