from typing import Any, Dict, Optional
import logging
import time
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import sys

# Initialize MCP server
mcp = FastMCP("alf_world")

# Ensure proper import paths
sys.path.insert(0, str(Path(__file__).parent))
from agentenv_alfworld.agentenv_alfworld.model import StepRequestBody, ResetRequestBody
from agentenv_alfworld.agentenv_alfworld.env_wrapper import server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

@mcp.tool()
async def hello() -> str:
    """Service health check"""
    return "This is environment AlfWorld."

@mcp.tool()
async def create_env() -> Dict[str, Any]:
    """Create a new AlfWorld environment"""
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
    """Execute an action in the AlfWorld environment"""
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
    """Reset the AlfWorld environment"""
    try:
        result = server.reset(body.id, body.game, body.world_type)
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
async def get_available_actions(env_id: int) -> Dict[str, Any]:
    """Get available actions for the environment"""
    try:
        actions = server.get_available_actions(env_id)
        return {
            "status": "success",
            "actions": actions,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def get_observation(env_id: int) -> Dict[str, Any]:
    """Get current observation from the environment"""
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
async def get_detailed_info(env_id: int) -> Dict[str, Any]:
    """Get detailed information about the environment"""
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
    # Start MCP server with stdio transport
    mcp.run(transport='stdio')