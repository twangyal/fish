import unittest
import asyncio
from tts.synthesis import MockTTS
from tts.streaming import TTSStreamBuffer

class TestTTS(unittest.IsolatedAsyncioTestCase):
    async def test_mock_tts(self):
        tts = MockTTS(chunk_delay=0.001)
        chunks = []
        async for chunk in tts.synthesize('test'):
            chunks.append(chunk)
        self.assertGreater(len(chunks), 0)
        self.assertIn(b'test', chunks[0])

    async def test_tts_stream_buffer(self):
        buffer = TTSStreamBuffer()
        async def token_stream():
            yield 'Hello'
            yield ' there!'
            yield ' How'
            yield ' are you?'
        phrases = []
        async for phrase in buffer.process_tokens(token_stream()):
            phrases.append(phrase)
        self.assertEqual(phrases, ['Hello there!', 'How are you?'])

from unittest.mock import patch, MagicMock
from tts.synthesis import PiperTTS

class TestPiperTTS(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_voice_instance = MagicMock()
        self.mock_piper = MagicMock()
        self.mock_piper.PiperVoice.load.return_value = self.mock_voice_instance
        self.modules_patcher = patch.dict('sys.modules', {'piper.voice': self.mock_piper})
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()

    async def test_piper_tts_synthesize(self):
        tts = PiperTTS(model_path="dummy.onnx")
        
        self.mock_voice_instance.synthesize_stream_raw.return_value = [b"audio", b"data"]
        
        chunks = []
        async for chunk in tts.synthesize("test"):
            chunks.append(chunk)
            
        self.assertEqual(b"".join(chunks), b"audiodata")
        self.mock_voice_instance.synthesize_stream_raw.assert_called_once_with("test")
