import unittest
from unittest.mock import patch, MagicMock
import asyncio
from audio.input import LiveMic
from audio.output import LiveSpeaker

class TestAudio(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_sd = MagicMock()
        self.modules_patcher = patch.dict('sys.modules', {'sounddevice': self.mock_sd})
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()

    async def test_live_mic(self):
        mic = LiveMic()
        self.assertIsNotNone(mic.sd)
        
        mock_stream = MagicMock()
        mock_stream.closed = False
        self.mock_sd.RawInputStream.return_value = mock_stream
        
        await mic.start()
        self.mock_sd.RawInputStream.assert_called_once()
        mock_stream.start.assert_called_once()
        
        # Simulate callback
        mic.callback(b'test', 4, None, None)
        
        chunk = await asyncio.wait_for(mic.stream().__anext__(), timeout=1.0)
        self.assertEqual(chunk, b'test')
        
        await mic.stop()
        mock_stream.stop.assert_called_once()

    async def test_live_speaker(self):
        speaker = LiveSpeaker()
        self.assertIsNotNone(speaker.sd)
        
        mock_stream = MagicMock()
        mock_stream.closed = False
        self.mock_sd.RawOutputStream.return_value = mock_stream
        
        await speaker.play(b'audio')
        # Allow the background thread in play() to execute
        await asyncio.sleep(0.01)
        self.mock_sd.RawOutputStream.assert_called_once()
        mock_stream.start.assert_called_once()
        mock_stream.write.assert_called_once_with(b'audio')
        
        await speaker.stop()
        mock_stream.stop.assert_called_once()