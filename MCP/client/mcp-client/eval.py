import asyncio
import json
import random
import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tqdm import tqdm
# 加载环境变量
load_dotenv()

class EnhancedMCPClient:
    def __init__(self):
        """初始化增强版MCP客户端"""
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        self.sessions = {}
        self.tools_map = {}
        
        # 测试数据集映射
        self.task_classes = {
            "webshop": '/root/autodl-tmp/data_j/webshop_train.json',
            "alf_world": '/root/autodl-tmp/data/alfworld_test.json',
            "baby_ai": '/root/autodl-tmp/data/babyai_test.json',
            "science_world": '/root/autodl-tmp/data/sciworld_test.json',
            "textcraft": '/root/autodl-tmp/data/textcraft_test.json',
            "webarena": '/root/autodl-tmp/data/webarena_test.json',
            "sqlgym": '/root/autodl-tmp/data/sqlgym_test.json',
            'maze': '/root/autodl-tmp/data/lmrlgym_maze_test.json',
            'wordle': '/root/autodl-tmp/data/lmrlgym_wordle_test.json',
            "weather": '/root/autodl-tmp/data/tool_weather_test.json',
            "todo": '/root/autodl-tmp/data/tool_todo_test.json',
            "movie": '/root/autodl-tmp/data/tool_movie_test.json',
            "sheet": '/root/autodl-tmp/data/tool_sheet_test.json',
            "academia": '/root/autodl-tmp/data/tool_academia_test.json'
        }
        
        # 服务器映射
        self.server_paths = {
            "weather": "../../server/weather/weather.py",
            "search": "../../server/search/search.py",
            "webshop": "/root/wrp/MCP-A2A/MCP/server/webshop/webshop.py",
            "textcraft": "/root/wrp/MCP-A2A/MCP/server/textcraft/textcraft.py",
            "movie": "/root/wrp/MCP-A2A/MCP/server/tool/movie.py",
            "academia": "/root/wrp/MCP-A2A/MCP/server/tool/academia.py",
            "sheet": "/root/wrp/MCP-A2A/MCP/server/tool/sheet.py",
            "todo": "/root/wrp/MCP-A2A/MCP/server/tool/todo.py",
            "weather_env": "/root/wrp/MCP-A2A/MCP/server/tool/weather.py",
            "alf_world": "/root/wrp/MCP-A2A/MCP/server/alfworld/alf_world.py",
            "baby_ai": "/root/wrp/MCP-A2A/MCP/server/babyai/babyai.py",
            "science_world": "/root/wrp/MCP-A2A/MCP/server/sciworld/sciworld.py",
            "webarena": "/root/wrp/MCP-A2A/MCP/server/webarena/webarena.py"
        }

    async def connect_to_server(self, server_name: str):
        """连接到指定名称的服务器"""
        if server_name not in self.server_paths:
            raise ValueError(f"未知的服务器名称: {server_name}")
        
        server_path = self.server_paths[server_name]
        
        # 检查服务器脚本类型
        is_python = server_path.endswith('.py')
        is_js = server_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 Python 或 JavaScript 文件")

        # 构建服务器参数
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_path],
            env=None
        )

        # 建立连接
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await session.initialize()
        self.sessions[server_name] = {
            "session": session, 
            "stdio": stdio, 
            "write": write
        }
        
        # 更新工具映射
        response = await session.list_tools()
        for tool in response.tools:
            self.tools_map[tool.name] = server_name
        
        print(f"成功连接到服务器: {server_name}")

    def load_dataset(self, server_name: str) -> List[Dict]:
        """加载测试数据集（直接使用item_id作为测试指令）"""
        if server_name not in self.task_classes:
            raise ValueError(f"未知的服务器名称: {server_name}")
        
        try:
            with open(self.task_classes[server_name], 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    raise ValueError("数据集必须是JSON数组")
                    
                # 直接返回原始数据，确保每个item都有item_id
                for item in data:
                    if not isinstance(item, dict) or "item_id" not in item:
                        raise ValueError("每个项目必须是包含item_id的字典")
                
                return data
            
        except Exception as e:
            raise ValueError(f"加载数据集失败: {str(e)}")

    def load_jailbreak_prompts(self, jailbreak_path: str) -> List[str]:
        """加载越狱数据集"""
        try:
            with open(jailbreak_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [p for p in data if isinstance(p, str)]
                elif isinstance(data, dict) and 'prompts' in data:
                    return [p for p in data['prompts'] if isinstance(p, str)]
                else:
                    raise ValueError("越狱数据集格式不正确")
        except Exception as e:
            raise ValueError(f"加载越狱数据集失败: {str(e)}")

    def combine_prompts(self, original: str, jailbreak: str, position: int) -> str:
        """组合原始prompt和越狱prompt"""
        if not original or not jailbreak:
            return original
        
        if position == 1:  # 前面插入
            return f"{jailbreak}\n\n{original}"
        elif position == 2:  # 后面插入
            return f"{original}\n\n{jailbreak}"
        elif position == 3:  # 随机位置插入
            lines = original.split('\n')
            if len(lines) <= 1:
                return f"{jailbreak}\n\n{original}"
            insert_pos = random.randint(0, len(lines)-1)
            return '\n'.join(lines[:insert_pos] + [jailbreak] + lines[insert_pos:])
        return original

    async def process_message(self, message: str) -> str:
        """处理单个消息并获取响应"""
        available_tools = []
        
        # 构建可用工具列表
        for tool_name, server_id in self.tools_map.items():
            session = self.sessions[server_id]["session"]
            response = await session.list_tools()
            for tool in response.tools:
                if tool.name == tool_name:
                    available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema
                        }
                    })

        messages = [{"role": "user", "content": message}]
        
        while True:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=available_tools,
            )

            content = response.choices[0]
            if content.finish_reason == "tool_calls":
                for tool_call in content.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    server_id = self.tools_map.get(tool_name)
                    if not server_id:
                        raise ValueError(f"未找到工具 {tool_name} 对应的服务端")

                    session = self.sessions[server_id]["session"]
                    result = await session.call_tool(tool_name, tool_args)
                    
                    messages.append({
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool_call.id,
                    })
            else:
                return content.message.content

    async def test_prompt(self, prompt: str) -> Dict:
        """测试单个prompt"""
        try:
            response = await self.process_message(prompt)
            return {
                "success": True,
                "response": response,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

    async def run_test_sequence(
        self, 
        server_name: str, 
        jailbreak_path: str, 
        position: int = 1,
        max_tests: Optional[int] = None
    ) -> List[Dict]:
        """
        运行完整的测试序列
        :param server_name: 服务器名称
        :param jailbreak_path: 越狱数据集路径
        :param position: 插入位置 (1=前插, 2=后插, 3=随机插)
        :param max_tests: 最大测试数量 (None表示不限制)
        :return: 测试结果列表
        """
        # 1. 连接服务器
        await self.connect_to_server(server_name)
        print('server init done')
        # 2. 加载数据集
        try:
            test_data = self.load_dataset(server_name)
            jailbreak_prompts = self.load_jailbreak_prompts(jailbreak_path)
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return []
        print('data initial done!')
        # 3. 运行测试
        results = []
        test_count = 0
        
        print(f"\n开始测试服务器: {server_name}")
        print(f"测试数据集大小: {len(test_data)}")
        print(f"越狱数据集大小: {len(jailbreak_prompts)}")
        print(f"组合方式: {['前插', '后插', '随机插'][position-1]}\n")
        
        # 外层循环：测试数据集
        for test_item in tqdm(test_data):
            print('test_item:',test_item)
            # if not isinstance(test_item, dict) or 'prompt' not in test_item:
            #     continue
                
            original_prompt = test_item['conversations'][0]['value']
            
            # 内层循环：越狱数据集
            for jailbreak_prompt in tqdm(jailbreak_prompts):
                # 检查是否达到最大测试数量
                if max_tests is not None and test_count >= max_tests:
                    print(f"达到最大测试数量 {max_tests}, 停止测试")
                    return results
                
                # 组合prompt
                combined_prompt = self.combine_prompts(
                    original_prompt, 
                    jailbreak_prompt,
                    position
                )
                
                # 测试组合后的prompt
                test_result = await self.test_prompt(combined_prompt)
                
                # 记录结果
                result = {
                    "server": server_name,
                    "original_prompt": original_prompt,
                    "jailbreak_prompt": jailbreak_prompt,
                    "combined_prompt": combined_prompt,
                    "test_result": test_result,
                    "position": position
                }
                print('result:',result)
                results.append(result)
                test_count += 1
                
                # 打印进度
                print(f"测试进度: {test_count} 组合")
                print(f"原始: {original_prompt[:50]}...")
                print(f"越狱: {jailbreak_prompt[:50]}...")
                print(f"结果: {'成功' if test_result['success'] else '失败'}")
                if not test_result['success']:
                    print(f"错误: {test_result['error']}")
                print("-" * 50)
        
        return results

    async def clean(self):
        """清理资源"""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()
async def main():
    # 用户输入
    print("=== MCP增强测试客户端 ===")
    server_name = input("请输入服务器名称 (可选 weather search webshop textcraft movie academia sheet todo weathe_env alf_world baby_ai science_world webarena): ").strip()
    jailbreak_path = input("请输入越狱数据集JSON文件路径: ").strip()
    position = int(input("请输入插入位置 (1=前插, 2=后插, 3=随机插): ").strip())
    max_tests = input("请输入最大测试数量 (留空则不限制): ").strip()
    max_tests = int(max_tests) if max_tests else None
    
    # 初始化客户端
    client = EnhancedMCPClient()
    
    try:
        # 运行测试
        results = await client.run_test_sequence(
            server_name=server_name,
            jailbreak_path=jailbreak_path,
            position=position,
            max_tests=max_tests
        )
        
        # 保存结果
        timestamp = os.path.join("test_results", f"results_{int(time.time())}.json")
        os.makedirs("test_results", exist_ok=True)
        
        with open(timestamp, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        success_count = sum(1 for r in results if r['test_result']['success'])
        print(f"\n测试完成! 共测试 {len(results)} 个组合")
        print(f"成功: {success_count}")
        print(f"失败: {len(results) - success_count}")
        print(f"结果已保存到: {timestamp}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
    finally:
        await client.clean()

if __name__ == "__main__":
    import time
    asyncio.run(main())



