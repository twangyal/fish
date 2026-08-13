import asyncio
from typing import AsyncGenerator, Optional
import openai
from agent.interface import LLM
import threading

class MockLLM(LLM):
    def __init__(self, token_delay: float = 0.01, response_text: str = 'Hello there! I am a fish.'):
        self.token_delay = token_delay
        self.response_text = response_text
        self.first_token_emitted = False

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        self.first_token_emitted = False
        words = self.response_text.split(' ')
        for i, word in enumerate(words):
            await asyncio.sleep(self.token_delay)
            self.first_token_emitted = True
            yield word + (' ' if i < len(words) - 1 else '')

class LlamaLLM(LLM):
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        self.lock = threading.Lock()
        try:
            from llama_cpp import Llama
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        except ImportError:
            self.llm = None
            print("Warning: llama_cpp not installed, LlamaLLM will return empty strings.")

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        if self.llm is None:
            yield ""
            return

        q = asyncio.Queue()
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()

        def generate_sync():
            with self.lock:
                try:
                    for output in self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    ):
                        if cancel_event.is_set():
                            break
                        if "choices" in output and len(output["choices"]) > 0:
                            delta = output["choices"][0].get("delta", {})
                            if "content" in delta:
                                token = delta["content"]
                                loop.call_soon_threadsafe(q.put_nowait, token)
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

        thread = loop.run_in_executor(None, generate_sync)

        try:
            while True:
                token = await q.get()
                if token is None:
                    break
                if isinstance(token, Exception):
                    raise token
                yield token
        except asyncio.CancelledError:
            cancel_event.set()
            raise

class OpenAILLM(LLM):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
