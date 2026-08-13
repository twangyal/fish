import unittest
import os
import json
import asyncio
from agent.memory import ShortTermMemory, LongTermMemory
from agent.tools.registry import ToolRegistry
from agent.tools.implementations import registry as impl_registry
from agent.logger import AgentLogger
from agent.brain import PlanningBrain
from agent.llm import MockLLM
from voice_loop import VoiceLoop, VoiceLoopState

class TestAssistantBrain(unittest.TestCase):
    def setUp(self):
        self.log_file = "test_agent.log"
        self.mem_file = "test_memory.json"
        self.logger = AgentLogger(self.log_file)
        self.short_mem = ShortTermMemory(window_size=5, system_prompt="Test Prompt")
        self.long_mem = LongTermMemory(self.mem_file)
        self.llm = MockLLM(response_text="Test response")
        self.brain = PlanningBrain(self.llm, self.short_mem, self.logger)
        
    def tearDown(self):
        self.logger.close()
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if os.path.exists(self.mem_file):
            os.remove(self.mem_file)

    def test_short_term_memory(self):
        self.short_mem.add_message("user", "Hello")
        ctx = self.short_mem.get_context()
        self.assertEqual(len(ctx), 2)
        self.assertEqual(ctx[0]["role"], "system")
        self.assertEqual(ctx[1]["content"], "Hello")

    def test_long_term_memory(self):
        async def run_mem():
            await self.long_mem.save_fact("color", "blue", ["ui"])
            facts = await self.long_mem.recall_facts("blue")
            self.assertIn("color", facts)
            await self.long_mem.delete_fact("color")
            self.assertEqual(len(await self.long_mem.recall_facts()), 0)
        asyncio.run(run_mem())

    def test_concurrent_memory_writes(self):
        async def run_concurrent():
            memory = self.long_mem
            await asyncio.gather(*[memory.save_fact(f"key_{i}", f"val_{i}") for i in range(10)])
            facts = await memory.recall_facts()
            for i in range(10):
                self.assertIn(f"key_{i}", facts)
                self.assertEqual(facts[f"key_{i}"]["value"], f"val_{i}")
        asyncio.run(run_concurrent())

    def test_tool_registry(self):
        schemas = impl_registry.get_schemas()
        self.assertTrue(len(schemas) > 0)
        tool_names = [s["name"] for s in schemas]
        self.assertIn("remember_fact", tool_names)
        self.assertIn("express_emotion", tool_names)

    def test_planning_brain_fast_path(self):
        async def run_fast():
            response = ""
            async for token in self.brain.process("Hello there"):
                response += token
            self.assertEqual(response, "Test response")
        asyncio.run(run_fast())

    def test_planning_brain_react_path(self):
        async def run_react():
            response = ""
            async for token in self.brain.process("Please remember my name"):
                response += token
            self.assertEqual(response, "Test response")
        asyncio.run(run_react())

    def test_logger(self):
        self.logger.thought("thinking")
        self.assertTrue(os.path.exists(self.log_file))

    def test_voice_loop_integration(self):
        loop = VoiceLoop({"brain": {"target": "mock"}})
        self.assertIsNotNone(loop.brain)
        self.assertIsNotNone(loop.memory)
        self.assertIsNotNone(loop.logger)

    def test_set_reminder_integration(self):
        async def run_reminder():
            loop = VoiceLoop({"brain": {"target": "mock"}})
            VoiceLoop._active_loop = loop
            
            await impl_registry.execute("set_reminder", {"message": "test_msg", "delay_seconds": 0.01})
            
            await asyncio.sleep(0.05)
            
            msg = await loop._notification_queue.get()
            self.assertEqual(msg, "test_msg")
            
            response = ""
            async for token in self.brain.process("remind me to do laundry"):
                response += token
            self.assertEqual(response, "Test response")
            
        asyncio.run(run_reminder())

if __name__ == "__main__":
    unittest.main()
