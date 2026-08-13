import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio

from asr.engine import OpenAIWhisperASR
from agent.llm import OpenAILLM
from tts.synthesis import OpenAITTS, ElevenLabsTTS
from voice_loop import VoiceLoop

class TestCloudEngines(unittest.IsolatedAsyncioTestCase):

    @patch("asr.engine.openai.AsyncOpenAI")
    async def test_openai_whisper_asr(self, mock_openai):
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "hello world"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        
        asr = OpenAIWhisperASR(api_key="test", model="whisper-1")
        
        async def mock_audio_stream():
            yield b"empty"
            
        result = await asr.transcribe(mock_audio_stream())
        self.assertEqual(result, "hello world")
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch("agent.llm.openai.AsyncOpenAI")
    async def test_openai_llm(self, mock_openai):
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        async def async_gen():
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "hello"
            yield mock_chunk
            
            mock_chunk2 = MagicMock()
            mock_chunk2.choices = [MagicMock()]
            mock_chunk2.choices[0].delta.content = " world"
            yield mock_chunk2

        mock_client.chat.completions.create = AsyncMock(return_value=async_gen())
        
        llm = OpenAILLM(api_key="test", model="gpt-4")
        
        result = []
        async for chunk in llm.generate_response("say hi"):
            result.append(chunk)
            
        self.assertEqual("".join(result), "hello world")

    @patch("tts.synthesis.openai.AsyncOpenAI")
    async def test_openai_tts(self, mock_openai):
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response_ctx = AsyncMock()
        mock_response = AsyncMock()
        
        async def iter_bytes():
            yield b"audio"
            yield b"data"
            
        mock_response.iter_bytes = iter_bytes
        mock_response_ctx.__aenter__.return_value = mock_response
        
        mock_client.audio.speech.with_streaming_response.create = MagicMock(return_value=mock_response_ctx)
        
        tts = OpenAITTS(api_key="test", model="tts-1", voice="alloy")
        
        result = []
        async for chunk in tts.synthesize("hello"):
            result.append(chunk)
            
        self.assertEqual(b"".join(result), b"audiodata")

    @patch("tts.synthesis.httpx.AsyncClient")
    async def test_elevenlabs_tts(self, mock_httpx):
        mock_client_instance = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        mock_response_ctx = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        
        async def aiter_bytes():
            yield b"11labs"
            yield b"audio"
            
        mock_response.aiter_bytes = aiter_bytes
        mock_response_ctx.__aenter__.return_value = mock_response
        
        mock_client_instance.stream = MagicMock(return_value=mock_response_ctx)
        
        tts = ElevenLabsTTS(api_key="test")
        
        result = []
        async for chunk in tts.synthesize("hello"):
            result.append(chunk)
            
        self.assertEqual(b"".join(result), b"11labsaudio")

    @patch("voice_loop.LiveSpeaker")
    @patch("voice_loop.LiveMic")
    @patch("voice_loop.OpenAITTS")
    @patch("voice_loop.OpenAILLM")
    @patch("voice_loop.OpenAIWhisperASR")
    def test_voice_loop_cloud_target(self, mock_asr, mock_llm, mock_tts, mock_mic, mock_speaker):
        config = {
            "brain": {
                "target": "cloud",
                "asr_model": "whisper-test",
                "llm_model": "gpt-test",
                "tts_provider": "openai",
                "tts_model": "tts-test",
                "voice": "voice-test"
            }
        }
        
        loop = VoiceLoop(config=config)
        self.assertEqual(loop.state.name, "LISTENING")
        
        mock_asr.assert_called_once_with(model="whisper-test")
        mock_llm.assert_called_once_with(model="gpt-test")
        mock_tts.assert_called_once_with(model="tts-test", voice="voice-test")

    def test_elevenlabs_tts_env_key(self):
        import os
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env_key"}):
            tts = ElevenLabsTTS(api_key=None)
            self.assertEqual(tts.api_key, "env_key")
            
    @patch("voice_loop.LiveSpeaker")
    @patch("voice_loop.LiveMic")
    @patch("voice_loop.ElevenLabsTTS")
    @patch("voice_loop.OpenAILLM")
    @patch("voice_loop.OpenAIWhisperASR")
    def test_voice_loop_cloud_elevenlabs(self, mock_asr, mock_llm, mock_tts, mock_mic, mock_speaker):
        config = {
            "brain": {
                "target": "cloud",
                "tts_provider": "elevenlabs",
                "api_key": "test_elevenlabs_key",
                "voice_id": "test_voice",
                "tts_model": "test_model"
            }
        }
        
        loop = VoiceLoop(config=config)
        self.assertEqual(loop.state.name, "LISTENING")
        mock_tts.assert_called_once_with(api_key="test_elevenlabs_key", voice_id="test_voice", model_id="test_model")

if __name__ == "__main__":
    unittest.main()
