import asyncio
from typing import AsyncGenerator, Optional
import openai
import httpx
from agent.interface import TTS
import threading

class MockTTS(TTS):
    def __init__(self, chunk_delay: float = 0.01):
        self.chunk_delay = chunk_delay

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(self.chunk_delay)
        yield b'mock_audio_sample_for: ' + text.encode('utf-8')

class PiperTTS(TTS):
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.lock = threading.Lock()
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(model_path, config_path=config_path)
        except ImportError:
            self.voice = None
            print("Warning: piper not installed, PiperTTS will yield empty audio.")

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        if self.voice is None:
            yield b""
            return

        q = asyncio.Queue()
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()

        def synthesize_sync():
            with self.lock:
                try:
                    for audio_bytes in self.voice.synthesize_stream_raw(text):
                        if cancel_event.is_set():
                            break
                        loop.call_soon_threadsafe(q.put_nowait, audio_bytes)
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

        thread = loop.run_in_executor(None, synthesize_sync)

        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        except asyncio.CancelledError:
            cancel_event.set()
            raise

class OpenAITTS(TTS):
    def __init__(self, api_key: Optional[str] = None, model: str = "tts-1", voice: str = "alloy"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        async with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm"
        ) as response:
            async for chunk in response.iter_bytes():
                yield chunk

class ElevenLabsTTS(TTS):
    def __init__(self, api_key: Optional[str] = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM", model_id: str = "eleven_monolingual_v1"):
        import os
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.model_id = model_id
        self.url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_16000"

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        headers = {}
        if self.api_key:
            headers["xi-api-key"] = self.api_key
            
        data = {
            "text": text,
            "model_id": self.model_id
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.url, headers=headers, json=data) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
