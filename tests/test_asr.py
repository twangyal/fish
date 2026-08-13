import unittest
import asyncio
from asr.vad import MockVAD, VADEvent
from asr.engine import MockASR

class TestASR(unittest.IsolatedAsyncioTestCase):
    async def test_mock_vad(self):
        vad = MockVAD()
        self.assertIsNone(vad.analyze(b'chunk'))

    async def test_mock_asr(self):
        asr = MockASR('hello world')
        async def stream():
            yield b'chunk1'
            yield b'chunk2'
        result = await asr.transcribe(stream())
        self.assertEqual(result, 'hello world')

from unittest.mock import patch, MagicMock
from asr.engine import WhisperASR

class TestWhisperASR(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_whisper_model = MagicMock()
        self.mock_faster_whisper = MagicMock()
        self.mock_faster_whisper.WhisperModel.return_value = self.mock_whisper_model
        self.modules_patcher = patch.dict('sys.modules', {'faster_whisper': self.mock_faster_whisper})
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()

    async def test_whisper_asr_transcribe(self):
        asr = WhisperASR()
        self.assertEqual(asr.model, self.mock_whisper_model)
        
        mock_segment = MagicMock()
        mock_segment.text = "test transcription"
        self.mock_whisper_model.transcribe.return_value = ([mock_segment], None)
        
        async def mock_stream():
            yield b'\x00\x00' * 16000
            
        result = await asr.transcribe(mock_stream())
        self.assertEqual(result, "test transcription")
        self.mock_whisper_model.transcribe.assert_called_once()
