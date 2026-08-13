import asyncio
import numpy as np
import io
import wave
import openai
from typing import AsyncGenerator, Optional
from agent.interface import ASR

class MockASR(ASR):
    def __init__(self, transcription: str = 'This is a mock transcription.'):
        self.transcription = transcription
        self.delay: float = 0.05

    async def transcribe(self, audio_stream: AsyncGenerator[bytes, None]) -> str:
        async for _ in audio_stream:
            pass
        await asyncio.sleep(self.delay)
        return self.transcription

class WhisperASR(ASR):
    def __init__(self, model_size_or_path: str = "base.en", device: str = "cpu", compute_type: str = "float32"):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
        except ImportError:
            self.model = None
            print("Warning: faster_whisper not installed, WhisperASR will return empty strings.")

    async def transcribe(self, audio_stream: AsyncGenerator[bytes, None]) -> str:
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
            
        if not chunks or self.model is None:
            return ""

        audio_bytes = b"".join(chunks)
        
        # Convert 16-bit PCM to float32 numpy array normalized to [-1.0, 1.0]
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        def run_transcription():
            segments, _ = self.model.transcribe(audio_array, beam_size=5)
            return " ".join([segment.text for segment in segments]).strip()

        text = await asyncio.to_thread(run_transcription)
        return text

class OpenAIWhisperASR(ASR):
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def transcribe(self, audio_stream: AsyncGenerator[bytes, None]) -> str:
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
            
        if not chunks:
            return ""

        audio_bytes = b"".join(chunks)
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_bytes)
            
        wav_io.seek(0)
        
        response = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", wav_io, "audio/wav")
        )
        return response.text
