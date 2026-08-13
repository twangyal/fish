import unittest
import asyncio
from voice_loop import VoiceLoop, VoiceLoopState
from asr.vad import VADEvent

class TestBargeIn(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = VoiceLoop()
        self.start_task = None
        
    async def asyncTearDown(self):
        await self.loop.stop()
        if self.start_task and not self.start_task.done():
            self.start_task.cancel()
        await asyncio.sleep(0.01)

    async def wait_for_state(self, state, timeout=2.0):
        async def _wait():
            while self.loop.state != state:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(_wait(), timeout=timeout)

    async def test_barge_in_processing_asr(self):
        asr_event = asyncio.Event()
        
        async def delayed_transcribe(*args, **kwargs):
            await asr_event.wait()
            return "hello"
            
        self.loop.asr.transcribe = delayed_transcribe
        
        self.start_task = asyncio.create_task(self.loop.start())
        await asyncio.sleep(0.01)
        
        self.loop.vad.analyze = lambda c: VADEvent.SPEECH_END
        self.loop.mic.inject_audio(b"fake_end")
        
        await self.wait_for_state(VoiceLoopState.PROCESSING_ASR)
        
        self.loop.vad.analyze = lambda c: VADEvent.SPEECH_START
        self.loop.mic.inject_audio(b"fake_start")
        await asyncio.sleep(0.01)
        
        self.assertEqual(self.loop.state, VoiceLoopState.LISTENING)
        self.assertTrue(self.loop.active_task.cancelled() or self.loop.active_task.done())
        
        asr_event.set()

    async def test_barge_in_generating(self):
        llm_event = asyncio.Event()
        
        async def delayed_generate(*args, **kwargs):
            yield "token1"
            await llm_event.wait()
            yield "token2"
            
        self.loop.llm.generate_response = delayed_generate
        
        self.start_task = asyncio.create_task(self.loop.start())
        await asyncio.sleep(0.01)
        
        self.loop.vad.analyze = lambda c: VADEvent.SPEECH_END
        self.loop.mic.inject_audio(b"fake_end")
        
        await self.wait_for_state(VoiceLoopState.GENERATING)
            
        self.loop.vad.analyze = lambda c: VADEvent.SPEECH_START
        self.loop.mic.inject_audio(b"fake_start")
        await asyncio.sleep(0.01)
        
        self.assertEqual(self.loop.state, VoiceLoopState.LISTENING)
        self.assertTrue(self.loop.active_task.cancelled() or self.loop.active_task.done())
        
        llm_event.set()

    async def test_barge_in_speaking(self):
        tts_event = asyncio.Event()
        
        async def delayed_synthesize(*args, **kwargs):
            yield b"chunk1"
            await tts_event.wait()
            yield b"chunk2"
            
        self.loop.tts.synthesize = delayed_synthesize
        
        self.start_task = asyncio.create_task(self.loop.start())
        await asyncio.sleep(0.01)
        
        self.loop.vad.analyze = lambda c: VADEvent.SPEECH_END
        self.loop.mic.inject_audio(b"fake_end")
        
        await self.wait_for_state(VoiceLoopState.SPEAKING)
            
        self.loop.vad.analyze = lambda c: VADEvent.SPEECH_START
        self.loop.mic.inject_audio(b"fake_start")
        await asyncio.sleep(0.01)
        
        self.assertEqual(self.loop.state, VoiceLoopState.LISTENING)
        self.assertTrue(self.loop.active_task.cancelled() or self.loop.active_task.done())
        
        tts_event.set()
