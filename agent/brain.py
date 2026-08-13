from agent.tools.implementations import registry
from agent.memory import ShortTermMemory
from agent.logger import AgentLogger
import asyncio

from typing import AsyncGenerator

class PlanningBrain:
    def __init__(self, llm_engine, memory_manager: ShortTermMemory, logger: AgentLogger):
        self.llm = llm_engine
        self.memory = memory_manager
        self.logger = logger
        self.tools = registry

    async def process(self, user_input: str) -> AsyncGenerator[str, None]:
        # We simulate the dual-path returning an AsyncGenerator for streaming
        self.logger.thought(f"Processing input: {user_input}")
        
        needs_tools = any(kw in user_input.lower() for kw in [
            "remember", "forget", "recall", "tail", "head", "emotion", "status", "stop", "remind"
        ])

        if not needs_tools:
            self.logger.thought("Taking Fast-Path")
            async for chunk in self.llm.generate_response(user_input):
                yield chunk
            return
            
        self.logger.thought("Taking ReAct Path")
        
        # Simplified ReAct loop for demonstration
        tool_call_result = None
        executed_tool = None
        for name in self.tools._tools.keys():
            if name.replace('_', ' ') in user_input.lower() or name in user_input.lower():
                self.logger.action(name, {})
                try:
                    # Simple heuristic mock execution
                    if name == 'remember_fact':
                        tool_call_result = await self.tools.execute(name, {'key': 'default_key', 'value': 'default_value'})
                    elif name == 'set_reminder':
                        tool_call_result = await self.tools.execute(name, {'message': 'default_msg', 'delay_seconds': 10})
                    elif name == 'express_emotion':
                        tool_call_result = await self.tools.execute(name, {'emotion': 'happy'})
                    else:
                        tool_call_result = await self.tools.execute(name, {})
                    self.logger.observation(name, tool_call_result)
                except Exception as e:
                    tool_call_result = f"Failed: {e}"
                    self.logger.observation(name, tool_call_result)
                executed_tool = name
                break
                
        prompt = f"{user_input}\nTool {executed_tool} Result: {tool_call_result}\nAnswer accordingly." if executed_tool else user_input
        async for chunk in self.llm.generate_response(prompt):
            yield chunk
