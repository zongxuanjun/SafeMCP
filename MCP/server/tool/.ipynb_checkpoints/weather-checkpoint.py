from typing import List, Dict, Any, Optional
import logging
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 导入原有模块
from agentenv_weather.weather_environment import weather_env_server
from agentenv_weather.weather_model import *
from agentenv_weather.weather_utils import debug_flg

# --- 初始化配置 ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("weather_env")

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('weather_env.log')
    ]
)
logger = logging.getLogger(__name__)

# --- 增强功能模块 ---
class WeatherRateLimiter:
    """天气环境专用速率限制器"""
    def __init__(self):
        self.config = {
            'global': {'per_second': 20, 'per_minute': 200},
            'env': {'per_second': 5, 'per_minute': 50}
        }
        self.reset_counters()
    
    def reset_counters(self):
        self.counters = {
            'global': {'second': 0, 'minute': 0, 'last_reset': time.time()},
            'env': {}
        }
    
    def check(self, env_idx: Optional[int] = None) -> None:
        now = time.time()
        
        # 全局计数器
        if now - self.counters['global']['last_reset'] > 60:
            self.counters['global'] = {'second': 0, 'minute': 0, 'last_reset': now}
        
        self.counters['global']['second'] += 1
        self.counters['global']['minute'] += 1
        
        # 环境级计数器
        if env_idx is not None:
            if env_idx not in self.counters['env']:
                self.counters['env'][env_idx] = {'second': 0, 'minute': 0, 'last_reset': now}
            
            if now - self.counters['env'][env_idx]['last_reset'] > 60:
                self.counters['env'][env_idx] = {'second': 0, 'minute': 0, 'last_reset': now}
            
            self.counters['env'][env_idx]['second'] += 1
            self.counters['env'][env_idx]['minute'] += 1
        
        # 检查限制
        if (self.counters['global']['second'] > self.config['global']['per_second'] or
            self.counters['global']['minute'] > self.config['global']['per_minute']):
            raise RuntimeError("Global rate limit exceeded")
            
        if (env_idx and 
            (self.counters['env'][env_idx]['second'] > self.config['env']['per_second'] or
             self.counters['env'][env_idx]['minute'] > self.config['env']['per_minute'])):
            raise RuntimeError(f"Environment {env_idx} rate limit exceeded")

rate_limiter = WeatherRateLimiter()

# --- MCP工具函数 ---
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """服务健康检查（原/端点）"""
    return {
        "service": "weather_env",
        "status": "healthy",
        "version": mcp.version,
        "timestamp": time.time()
    }

@mcp.tool()
async def list_weather_environments() -> Dict[str, Any]:
    """列出所有天气环境（原/list_envs端点）"""
    try:
        rate_limiter.check()
        env_ids = list(weather_env_server.env.keys())
        return {
            "count": len(env_ids),
            "environments": env_ids,
            "memory_usage": f"{sys.getsizeof(weather_env_server.env)/1024:.2f} KB"
        }
    except Exception as e:
        logger.error(f"List environments failed: {str(e)}")
        raise

@mcp.tool()
async def create_weather_environment(location_id: int) -> Dict[str, Any]:
    """创建新天气环境（原/create端点）"""
    try:
        rate_limiter.check()
        env_idx = weather_env_server.create(location_id)
        logger.info(f"Created weather env {env_idx} for location {location_id}")
        return {
            "operation": "create",
            "env_idx": env_idx,
            "location_id": location_id,
            "status": "success",
            "created_at": time.time()
        }
    except Exception as e:
        logger.error(f"Create failed for location {location_id}: {str(e)}")
        return {
            "operation": "create",
            "status": "error",
            "error": str(e),
            "error_type": e.__class__.__name__
        }

@mcp.tool()
async def update_weather(
    env_idx: int,
    action: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """更新天气状态（原/step端点）"""
    try:
        rate_limiter.check(env_idx)
        observation, reward, done, _ = weather_env_server.step(env_idx, action)
        
        logger.debug(f"Weather update in env {env_idx}: {action}")
        return {
            "env_idx": env_idx,
            "action": action,
            "observation": observation,
            "reward": reward,
            "done": done,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Weather update failed in env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_weather_observation(env_idx: int) -> Dict[str, Any]:
    """获取天气观测数据（原/observation端点）"""
    try:
        rate_limiter.check(env_idx)
        obs = weather_env_server.observation(env_idx)
        return {
            "env_idx": env_idx,
            "observation": obs,
            "status": "success",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Get observation failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def reset_weather_environment(
    env_idx: int,
    new_location_id: Optional[int] = None
) -> Dict[str, Any]:
    """重置天气环境（原/reset端点）"""
    try:
        rate_limiter.check(env_idx)
        weather_env_server.reset(env_idx, new_location_id)
        obs = weather_env_server.observation(env_idx)
        
        logger.info(f"Reset env {env_idx} with new location {new_location_id}")
        return {
            "operation": "reset",
            "env_idx": env_idx,
            "new_location_id": new_location_id,
            "observation": obs,
            "status": "success",
            "reset_at": time.time()
        }
    except Exception as e:
        logger.error(f"Reset failed for env {env_idx}: {str(e)}")
        raise

@mcp.tool()
async def get_weather_metrics() -> Dict[str, Any]:
    """获取天气服务指标（新增功能）"""
    try:
        rate_limiter.check()
        return {
            "service": "weather_env",
            "uptime": time.time() - mcp.start_time,
            "active_envs": len(weather_env_server.env),
            "rate_limits": rate_limiter.config,
            "memory_usage": f"{sys.getsizeof(weather_env_server.env)/1024:.2f} KB",
            "python_version": sys.version
        }
    except Exception as e:
        logger.error(f"Get metrics failed: {str(e)}")
        raise

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(transport='stdio') # 支持 'stdio'/'websocket'/'http'
