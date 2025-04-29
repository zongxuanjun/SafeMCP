import asyncio
import os
import json
import random
from typing import List, Union
from openai import OpenAI
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载环境变量
load_dotenv()

class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.exit_stack = AsyncExitStack()
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        
        # 越狱测试配置
        self.jailbreak_enabled = False
        self.jailbreak_mode = None  # 'single' 或 'batch'
        self.jailbreak_position = None
        self.jailbreak_content = ""  # 单文本模式
        self.jailbreak_contents = []  # 批量模式
        self.current_jailbreak_index = 0  # 当前测试的越狱文本索引

        if not self.api_key:
            raise ValueError("未找到 API KEY. 请在 .env 文件中配置 API_KEY")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.sessions = {}
        self.tools_map = {}

    async def connect_to_server(self, server_id: str, server_script_path: str):
        """连接到 MCP 服务器"""
        if server_id in self.sessions:
            raise ValueError(f"服务端 {server_id} 已经连接")

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 Python 或 JavaScript 文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command,
                                            args=[server_script_path],
                                            env=None)

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await session.initialize()
        self.sessions[server_id] = {"session": session, "stdio": stdio, "write": write}
        print(f"已连接到 MCP 服务器: {server_id}")

        # 更新工具映射
        response = await session.list_tools()
        for tool in response.tools:
            self.tools_map[tool.name] = server_id

    async def list_tools(self):
        """列出所有服务端的工具"""
        if not self.sessions:
            print("没有已连接的服务端")
            return

        print("已连接的服务端工具列表:")
        for tool_name, server_id in self.tools_map.items():
            print(f"工具: {tool_name}, 来源服务端: {server_id}")

    def _load_jailbreak_contents(self, data_path: str) -> List[str]:
        """从文件加载越狱内容"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"警告: 越狱数据文件 {data_path} 未找到")
            return []

    def _insert_jailbreak_text(self, content: str) -> Union[str, List[str]]:
        """根据配置插入越狱文本"""
        if not self.jailbreak_enabled:
            return content
            
        if self.jailbreak_mode == 'single':
            # 单文本模式
            if self.jailbreak_position == 1:  # 前面插入
                return f"{self.jailbreak_content}\n\n{content}"
            elif self.jailbreak_position == 2:  # 后面插入
                return f"{content}\n\n{self.jailbreak_content}"
            elif self.jailbreak_position == 3:  # 随机位置插入
                lines = content.split('\n')
                if len(lines) <= 1:
                    return f"{content}\n\n{self.jailbreak_content}"
                insert_pos = random.randint(1, len(lines)-1)
                return '\n'.join(lines[:insert_pos] + [self.jailbreak_content] + lines[insert_pos:])
        elif self.jailbreak_mode == 'batch':
            # 批量模式 - 返回所有组合文本
            results = []
            for jailbreak_text in self.jailbreak_contents:
                if self.jailbreak_position == 1:  # 前面插入
                    results.append(f"{jailbreak_text}\n\n{content}")
                elif self.jailbreak_position == 2:  # 后面插入
                    results.append(f"{content}\n\n{jailbreak_text}")
                elif self.jailbreak_position == 3:  # 随机位置插入
                    lines = content.split('\n')
                    if len(lines) <= 1:
                        combined = f"{content}\n\n{jailbreak_text}"
                    else:
                        insert_pos = random.randint(1, len(lines)-1)
                        combined = '\n'.join(lines[:insert_pos] + [jailbreak_text] + lines[insert_pos:])
                    results.append(combined)
            return results
            
        return content

    async def configure_jailbreak(self):
        """配置越狱测试参数"""
        print("\n=== 越狱测试配置 ===")
        print("0. 禁用越狱测试")
        print("1. 单文本模式 (手动输入)")
        print("2. 批量模式 (从文件读取)")
        
        mode_choice = input("请选择测试模式 (0-2): ").strip()
        if mode_choice == '0':
            self.jailbreak_enabled = False
            print("已禁用越狱测试模式")
            return
        
        if mode_choice not in ['1', '2']:
            print("无效选择，使用默认单文本模式")
            mode_choice = '1'
        
        if mode_choice == '1':
            self.jailbreak_mode = 'single'
            self.jailbreak_content = input("请输入越狱文本: ").strip()
            if not self.jailbreak_content:
                print("越狱文本不能为空，禁用测试模式")
                self.jailbreak_enabled = False
                return
        else:
            self.jailbreak_mode = 'batch'
            data_path = input("请输入越狱数据文件路径: ").strip()
            self.jailbreak_contents = self._load_jailbreak_contents(data_path)
            if not self.jailbreak_contents:
                print("无法加载越狱内容，禁用测试模式")
                self.jailbreak_enabled = False
                return
            print(f"已加载 {len(self.jailbreak_contents)} 条越狱文本")
        
        self.jailbreak_enabled = True
        
        print("\n=== 插入位置配置 ===")
        print("1. 前面插入")
        print("2. 后面插入")
        print("3. 随机位置插入")
        
        pos_choice = input("请选择插入位置 (1-3): ").strip()
        self.jailbreak_position = int(pos_choice) if pos_choice in ['1', '2', '3'] else 1
        
        print(f"\n已配置: {self.jailbreak_mode}模式, 插入位置: {self.jailbreak_position}")

    async def process_query(self, query: str) -> Union[str, List[str]]:
        """处理用户查询"""
        if not self.jailbreak_enabled:
            messages = [{"role": "user", "content": query}]
            return await self._process_messages(messages)
        
        if self.jailbreak_mode == 'single':
            # 单文本模式
            messages = [{"role": "user", "content": self._insert_jailbreak_text(query)}]
            return await self._process_messages(messages)
        else:
            # 批量模式 - 逐个测试所有越狱文本
            results = []
            modified_contents = self._insert_jailbreak_text(query)
            
            for content in modified_contents:
                print(f"\n正在测试越狱文本: {content[:50]}...")
                messages = [{"role": "user", "content": content}]
                try:
                    response = await self._process_messages(messages)
                    results.append({
                        "jailbreak_text": content,
                        "response": response
                    })
                except Exception as e:
                    results.append({
                        "jailbreak_text": content,
                        "error": str(e)
                    })
            
            # 打印汇总结果
            print("\n=== 批量测试结果 ===")
            for i, result in enumerate(results, 1):
                if "error" in result:
                    print(f"{i}. 失败: {result['error']}")
                else:
                    print(f"{i}. 成功: {result['response'][:50]}...")
            
            return results

    async def _process_messages(self, messages: list) -> str:
        """处理消息并获取响应"""
        available_tools = []
        
        # 构建工具列表
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

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
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
                    
                    tool_response = {
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool_call.id,
                    }
                    messages.append(tool_response)
            else:
                return content.message.content

    async def chat_loop(self):
        """交互式聊天循环"""
        print("MCP 客户端已启动！输入命令或问题，'exit'退出")
        print("特殊命令:")
        print("!config - 配置越狱测试")
        print("!tools - 列出可用工具")

        while True:
            try:
                query = input("\n问: ").strip()
                if query.lower() == 'exit':
                    break
                elif query.lower() == '!config':
                    await self.configure_jailbreak()
                    continue
                elif query.lower() == '!tools':
                    await self.list_tools()
                    continue

                response = await self.process_query(query)
                if isinstance(response, list):
                    print("\n批量测试完成，查看上方汇总结果")
                else:
                    print(f"\nAI回复: {response}")

            except Exception as e:
                print(f"发生错误: {str(e)}")

    async def clean(self):
        """清理资源"""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()

async def main():
    client = MCPClient()
    try:
        # 连接所有服务端
        servers = {
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

        print("正在连接服务端...")
        for server_id, path in servers.items():
            try:
                await client.connect_to_server(server_id, path)
            except Exception as e:
                print(f"连接 {server_id} 失败: {str(e)}")

        await client.list_tools()
        await client.chat_loop()
    finally:
        await client.clean()

if __name__ == "__main__":
    asyncio.run(main())