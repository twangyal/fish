import unittest
import asyncio
from voice_loop import VoiceLoop, VoiceLoopState
from asr.vad import VADEvent

class TestVoiceLoop(unittest.IsolatedAsyncioTestCase):
    async def test_voice_loop_e2e_latency(self):
        loop = VoiceLoop()
        loop.vad.analyze = lambda c: None
        
        start_task = asyncio.create_task(loop.start())
        await asyncio.sleep(0.01)
        
        loop.vad.analyze = lambda c: VADEvent.SPEECH_START
        loop.mic.inject_audio(b'start')
        await asyncio.sleep(0.01)
        self.assertEqual(loop.state, VoiceLoopState.LISTENING)
        self.assertIsNotNone(loop.latency_tracker.t0)
        
        loop.vad.analyze = lambda c: VADEvent.SPEECH_END
        loop.mic.inject_audio(b'end')
        await asyncio.sleep(0.01)
        
        # Reset analyze so it doesn't keep emitting
        loop.vad.analyze = lambda c: None
        
        async def wait_for_listening():
            while loop.state != VoiceLoopState.LISTENING or loop.latency_tracker.t6 is None:
                await asyncio.sleep(0.01)
                
        await asyncio.wait_for(wait_for_listening(), timeout=2.0)
        
        self.assertEqual(loop.state, VoiceLoopState.LISTENING)
        summary = loop.latency_tracker.get_summary()
        
        self.assertIsNotNone(summary['speech_duration'])
        self.assertIsNotNone(summary['asr_latency'])
        self.assertIsNotNone(summary['ttft'])
        self.assertIsNotNone(summary['phrase_latency'])
        self.assertIsNotNone(summary['tts_latency'])
        self.assertIsNotNone(summary['playback_latency'])
        self.assertIsNotNone(summary['total_e2e_latency'])
        
        await loop.stop()
        start_task.cancel()
        await asyncio.sleep(0.01)
