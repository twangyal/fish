import unittest
import asyncio
from agent.llm import MockLLM
from agent.conversation import DefaultConversationController

class TestLLM(unittest.IsolatedAsyncioTestCase):
    async def test_mock_llm(self):
        llm = MockLLM(token_delay=0.001, response_text='Test response.')
        tokens = []
        async for token in llm.generate_response(''):
            tokens.append(token)
        self.assertEqual(''.join(tokens).strip(), 'Test response.')

    async def test_conversation_controller(self):
        ctrl = DefaultConversationController('System prompt.')
        ctrl.add_user_message('Hello')
        ctrl.add_assistant_message('Hi')
        prompt = ctrl.get_prompt()
        self.assertIn('system: System prompt.', prompt)
        self.assertIn('user: Hello', prompt)
        self.assertIn('assistant: Hi', prompt)

from unittest.mock import patch, MagicMock
from agent.llm import LlamaLLM

class TestLlamaLLM(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_llama_instance = MagicMock()
        self.mock_llama_cpp = MagicMock()
        self.mock_llama_cpp.Llama.return_value = self.mock_llama_instance
        self.modules_patcher = patch.dict('sys.modules', {'llama_cpp': self.mock_llama_cpp})
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()

    async def test_llama_llm_generate(self):
        llm = LlamaLLM(model_path="dummy.gguf")
        
        self.mock_llama_instance.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": "test"}}]},
            {"choices": [{"delta": {"content": " response"}}]}
        ]
        
        tokens = []
        async for token in llm.generate_response("hello"):
            tokens.append(token)
            
        self.assertEqual("".join(tokens), "test response")
        self.mock_llama_instance.create_chat_completion.assert_called_once()
