import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, TypedDict, Any, Mapping, Callable
from tqdm import tqdm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from agentenv.agentenv.controller.task import BaseTask
from agentenv.agentenv.controller.env import BaseEnvClient

# 定义数据结构
ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)

@dataclass
class ExperienceOutput:
    conversation: List[ConversationMessage]
    reward: float
    text: str
    seq_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]

@dataclass
class EvaluationOutput:
    experiences: Sequence[ExperienceOutput]
    score: float
    success: float

class MCPWebshopClient(BaseEnvClient):
    """实现BaseEnvClient接口的MCP客户端"""
    def __init__(self, server_path: str, test_data_path: str):
        self.server_path = server_path
        self.test_data_path = test_data_path
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tool_name = "webshop"
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
            if not isinstance(self.test_data, list):
                raise ValueError("测试数据集必须是JSON数组")
        
        self.current_task = None
        self.current_conversation = []

    def __len__(self) -> int:
        """返回测试数据集长度"""
        return len(self.test_data)

    async def connect(self):
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await self.session.initialize()
        print("MCP Webshop连接已建立")

    @property
    def conversation_start(self) -> List[ConversationMessage]:
        """返回初始对话状态"""
        if self.current_task is None:
            return []
            
        if 'conversations' in self.current_task:
            return [
                {"from": msg["from"], "loss": None, "value": msg["value"]}
                for msg in self.current_task['conversations']
            ]
        else:
            initial_msg = self.current_task.get('instruction', '请开始任务')
            return [{"from": "human", "loss": None, "value": initial_msg}]

    async def observe(self) -> str:
        """获取当前环境状态"""
        if not self.current_conversation:
            return await self._get_initial_state()
        return self.current_conversation[-1]["value"]

    async def reset(self, idx: int) -> None:
        """重置环境"""
        if idx >= len(self.test_data):
            raise IndexError(f"索引{idx}超出测试数据范围")
            
        self.current_task = self.test_data[idx]
        self.current_conversation = list(self.conversation_start)
        
        # 确保对话中包含初始状态
        if not self.current_conversation:
            initial_state = await self._get_initial_state()
            self.current_conversation.append(
                {"from": "human", "loss": None, "value": initial_state}
            )
            # print(initial_
    async def _get_initial_state(self) -> str:
        """获取初始环境状态"""
        if self.current_task is None:
            return "请开始任务"
            
        if 'instruction' in self.current_task:
            return self.current_task['instruction']
        elif 'prompt' in self.current_task:
            return self.current_task['prompt']
        elif 'conversations' in self.current_task and len(self.current_task['conversations']) > 0:
            return self.current_task['conversations'][0]['value']
        else:
            for value in self.current_task.values():
                if isinstance(value, str):
                    return value
            return "请开始任务"

    async def step(self, action: str) -> Dict[str, Any]:
        """
        执行动作并返回结果
        返回: {
            "state": str,  # 环境返回的新状态
            "reward": float,  # 奖励值
            "done": bool  # 是否结束
        }
        """
        try:
            # 添加Agent响应到对话
            self.current_conversation.append(
                {"from": "gpt", "loss": True, "value": action}
            )
            print('action:',action)
            # 构造MCP请求
            request = {
                "item_id": self.current_task.get("item_id", ""),
                "instruction": action,
                "conversation_history": self.current_conversation
            }
            
            # 调用MCP工具
            result = await self.session.call_tool(self.tool_name, request)
            response = result.content[0].text if result.content else ""
            
            # 解析响应
            try:
                response_data = json.loads(response)
                state = response_data.get("state", "")
                reward = float(response_data.get("reward", 0.0))
                done = bool(response_data.get("done", False))
            except json.JSONDecodeError:
                state = response
                reward = 1.0 if "success" in response.lower() else 0.0
                done = reward > 0
            
            # 添加环境响应到对话
            self.current_conversation.append(
                {"from": "human", "loss": None, "value": state}
            )
            
            return {
                "state": state,
                "reward": reward,
                "done": done
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.current_conversation.append(
                {"from": "human", "loss": None, "value": error_msg}
            )
            return {
                "state": error_msg,
                "reward": 0.0,
                "done": True
            }

    async def close(self):
        """关闭连接"""
        await self.exit_stack.aclose()
        self.session = None

class MCPWebshopTask(BaseTask):
    """实现BaseTask接口的MCP任务"""
    env_client_cls = MCPWebshopClient
    env_name = "webshop"

    def __init__(
        self,
        server_path: str,
        test_data_path: str,
        n_clients: int = 1,
    ):
        client_args = {
            "server_path": server_path,
            "test_data_path": test_data_path
        }
        super().__init__(client_args, n_clients)

class MCPEvaluator:
    def __init__(self, task: MCPWebshopTask):
        self.task = task

    async def _generate_experience_one(
        self,
        idx: int,
        max_rounds: Optional[int] = None
    ) -> ExperienceOutput:
        """生成单个经验（完整对话流程）"""
        client = self.task.clients[0]
        await client.connect()
        
        try:
            # 1. 重置环境
            await client.reset(idx)
            
            # 2. 获取初始状态并初始化对话
            state = await client.observe()
            conversation = list(client.conversation_start)
            conversation.append(
                {"from": "human", "loss": None, "value": state}
            )
            
            rounds = 0
            reward = 0.0
            done = False
            
            while not done:
                # 3. 生成Agent响应
                agent_response = await self._get_agent_response(conversation)
                
                # 4. 执行动作并获取环境反馈
                step_result = await client.step(agent_response)
                new_state, reward, done = (
                    step_result["state"],
                    step_result["reward"],
                    step_result["done"],
                )
                
                # 5. 更新对话历史
                conversation = list(client.current_conversation)
                
                rounds += 1
                if max_rounds is not None and rounds >= max_rounds:
                    break
            
            # 6. 构造完整的ExperienceOutput
            return ExperienceOutput(
                conversation=conversation,
                reward=reward,
                text="\n".join(msg["value"] for msg in conversation),
                seq_ids=[],
                attention_mask=[],
                action_mask=[]
            )
        finally:
            await client.close()

    async def _get_agent_response(self, conversation: List[ConversationMessage]) -> str:
        """获取Agent响应（示例实现，需替换为实际Agent调用）"""
        last_user_msg = next(
            (msg["value"] for msg in reversed(conversation) if msg["from"] == "human"),
            ""
        )
        
        if "搜索" in last_user_msg or "search" in last_user_msg.lower():
            return "search"
        elif "购买" in last_user_msg or "buy" in last_user_msg.lower():
            return "buy_now"
        else:
            return "search"  # 默认动作

    async def eval(
        self,
        idxs: Sequence[int],
        max_rounds: Optional[int] = None
    ) -> EvaluationOutput:
        """执行评测并返回分数"""
        experiences = []
        for idx in tqdm(idxs, desc="评测进度"):
            try:
                exp = await self._generate_experience_one(idx, max_rounds)
                experiences.append(exp)
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {str(e)}")
                experiences.append(ExperienceOutput(
                    conversation=[],
                    reward=0.0,
                    text=f"Error: {str(e)}",
                    seq_ids=[],
                    attention_mask=[],
                    action_mask=[]
                ))
        
        # 计算指标
        rewards = [exp.reward for exp in experiences]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        success_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        
        return EvaluationOutput(
            experiences=experiences,
            score=score,
            success=success_rate
        )

async def main():
    # 配置参数
    SERVER_PATH = "/root/wrp/MCP-A2A/MCP/server/webshop/webshop.py"
    TEST_DATA_PATH = "/root/autodl-tmp/data/webshop_test.json"
    
    # 创建并运行评测
    task = MCPWebshopTask(
        server_path=SERVER_PATH,
        test_data_path=TEST_DATA_PATH
    )
    
    evaluator = MCPEvaluator(task)
    results = await evaluator.eval(
        idxs=list(range(10)),  # 测试前10条数据
        max_rounds=7
    )
    
    # 输出结果
    print("\n==== EVALUATION RESULTS ====")
    print(f"Average Score: {results.score:.4f}")
    print(f"Success Rate: {results.success:.2%}")
    
    # 保存结果
    timestamp = int(time.time())
    os.makedirs("eval_results", exist_ok=True)
    result_file = f"eval_results/mcp_webshop_eval_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "score": results.score,
            "success_rate": results.success,
            "details": [
                {
                    "conversation": exp.conversation,
                    "reward": exp.reward
                } for exp in results.experiences
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {result_file}")
    return results.score

if __name__ == "__main__":
    final_score = asyncio.run(main())
    print(f"\nFinal Evaluation Score: {final_score:.4f}")