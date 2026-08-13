import asyncio
from enum import Enum
from typing import Dict, Any, Optional
from benchmarks.latency import LatencyTracker
from asr.vad import MockVAD, VADEvent
from asr.engine import MockASR, WhisperASR, OpenAIWhisperASR
from agent.prompts import BILLY_BASS_SYSTEM_PROMPT
from agent.llm import MockLLM, LlamaLLM, OpenAILLM
from agent.conversation import DefaultConversationController
from tts.synthesis import MockTTS, PiperTTS, OpenAITTS, ElevenLabsTTS
from tts.streaming import TTSStreamBuffer
from audio.input import MockMic, LiveMic
from audio.output import MockSpeaker, LiveSpeaker
from agent.brain import PlanningBrain
from agent.memory import ShortTermMemory
from agent.logger import AgentLogger


class VoiceLoopState(Enum):
    LISTENING = 1
    PROCESSING_ASR = 2
    GENERATING = 3
    SPEAKING = 4

class VoiceLoop:
    _active_loop = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        VoiceLoop._active_loop = self
        self.state = VoiceLoopState.LISTENING
        self.latency_tracker = LatencyTracker()
        
        self.vad = MockVAD()
        
        target = "mock"
        if config and "brain" in config and "target" in config["brain"]:
            target = config["brain"]["target"]
            
        if target == "local":
            asr_model = config["brain"].get("asr_model", "base.en") if config else "base.en"
            llm_model = config["brain"].get("llm_model", "model.gguf") if config else "model.gguf"
            tts_model = config["brain"].get("tts_model", "model.onnx") if config else "model.onnx"
            
            self.asr = WhisperASR(model_size_or_path=asr_model)
            self.llm = LlamaLLM(model_path=llm_model)
            self.tts = PiperTTS(model_path=tts_model)
            self.mic = LiveMic()
            self.speaker = LiveSpeaker()
        elif target == "cloud":
            asr_model = config["brain"].get("asr_model", "whisper-1") if config else "whisper-1"
            llm_model = config["brain"].get("llm_model", "gpt-4o-mini") if config else "gpt-4o-mini"
            tts_provider = config["brain"].get("tts_provider", "openai") if config else "openai"
            
            self.asr = OpenAIWhisperASR(model=asr_model)
            self.llm = OpenAILLM(model=llm_model)
            if tts_provider == "elevenlabs":
                api_key = config["brain"].get("api_key") if config else None
                voice_id = config["brain"].get("voice_id", "21m00Tcm4TlvDq8ikWAM") if config else "21m00Tcm4TlvDq8ikWAM"
                model_id = config["brain"].get("tts_model", "eleven_monolingual_v1") if config else "eleven_monolingual_v1"
                self.tts = ElevenLabsTTS(api_key=api_key, voice_id=voice_id, model_id=model_id)
            else:
                tts_model = config["brain"].get("tts_model", "tts-1") if config else "tts-1"
                voice = config["brain"].get("voice", "alloy") if config else "alloy"
                self.tts = OpenAITTS(model=tts_model, voice=voice)
            self.mic = LiveMic()
            self.speaker = LiveSpeaker()
        else:
            self.asr = MockASR()
            self.llm = MockLLM()
            self.tts = MockTTS()
            self.mic = MockMic()
            self.speaker = MockSpeaker()
            
        self.logger = AgentLogger()
        self.memory = ShortTermMemory(system_prompt=BILLY_BASS_SYSTEM_PROMPT)
        self.brain = PlanningBrain(self.llm, self.memory, self.logger)
        
        self.tts_buffer = TTSStreamBuffer()
        
        self.active_task = None
        self._running = False
        self._asr_queue = asyncio.Queue()
        self._notification_queue = asyncio.Queue() # out-of-band notification queue

    @classmethod
    def enqueue_notification(cls, msg: str):
        if cls._active_loop:
            cls._active_loop._notification_queue.put_nowait(msg)

    async def play_notification(self, msg: str):
        self.handle_barge_in()
        self.state = VoiceLoopState.SPEAKING
        audio_stream = self.tts.synthesize(msg)
        async for audio_chunk in audio_stream:
            await self.speaker.play(audio_chunk)
        self.state = VoiceLoopState.LISTENING

    async def notification_worker(self):
        while self._running:
            try:
                notif = await asyncio.wait_for(self._notification_queue.get(), timeout=1.0)
                self.logger.observation('background_task', notif)
                await self.play_notification(notif)
            except asyncio.TimeoutError:
                pass

    async def start(self):
        self._running = True
        self._notif_task = asyncio.create_task(self.notification_worker())
        await self.mic.start()
        
        try:
            async for chunk in self.mic.stream():
                if not self._running:
                    break
                    
                event = self.vad.analyze(chunk)
                
                if event == VADEvent.SPEECH_START:
                    if self.state in (VoiceLoopState.PROCESSING_ASR, VoiceLoopState.GENERATING, VoiceLoopState.SPEAKING):
                        self.handle_barge_in()
                    self.latency_tracker.reset()
                    self.latency_tracker.mark_t0()
                    self.state = VoiceLoopState.LISTENING
                    
                    while not self._asr_queue.empty():
                        self._asr_queue.get_nowait()
                        
                elif event == VADEvent.SPEECH_END and self.state == VoiceLoopState.LISTENING:
                    self.latency_tracker.mark_t1()
                    self.state = VoiceLoopState.PROCESSING_ASR
                    
                    await self._asr_queue.put(None)
                    self.active_task = asyncio.create_task(self.process_pipeline())
                
                if self.state == VoiceLoopState.LISTENING and self.latency_tracker.t0 is not None:
                    await self._asr_queue.put(chunk)
                    
        except asyncio.CancelledError:
            pass
        finally:
            await self.mic.stop()
            
    def handle_barge_in(self):
        if self.state in (VoiceLoopState.PROCESSING_ASR, VoiceLoopState.GENERATING, VoiceLoopState.SPEAKING):
            if self.active_task and not self.active_task.done():
                self.active_task.cancel()
            self.tts_buffer.clear()
            self.speaker.clear()
        
        asyncio.create_task(self.speaker.stop())
        self.latency_tracker.reset()
        self.state = VoiceLoopState.LISTENING
        
    async def stop(self):
        self._running = False
        if hasattr(self, '_notif_task'):
            self._notif_task.cancel()
        self.handle_barge_in()
        
    async def process_pipeline(self):
        try:
            async def asr_stream():
                while True:
                    chunk = await self._asr_queue.get()
                    if chunk is None:
                        break
                    yield chunk
                    
            transcription = await self.asr.transcribe(asr_stream())
            self.latency_tracker.mark_t2()
            
            self.memory.add_message('user', transcription)
            prompt = transcription
            
            self.state = VoiceLoopState.GENERATING
            
            self.active_task = asyncio.create_task(self.run_generation_and_tts(prompt))
            await self.active_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import logging
            logging.error(f"Error in process_pipeline: {e}")
            self.state = VoiceLoopState.LISTENING
            
    async def run_generation_and_tts(self, prompt: str):
        try:
            raw_token_stream = self.brain.process(prompt)
            
            async def wrapped_token_stream():
                first = True
                async for token in raw_token_stream:
                    if first:
                        if self.latency_tracker.t3 is None:
                            self.latency_tracker.mark_t3()
                        first = False
                    yield token
            
            token_stream = wrapped_token_stream()
            phrase_stream = self.tts_buffer.process_tokens(token_stream)
            
            first_phrase_marked = False
            full_response = ''
            
            async for phrase in phrase_stream:
                if not first_phrase_marked:
                    self.latency_tracker.mark_t4()
                    first_phrase_marked = True
                    
                audio_stream = self.tts.synthesize(phrase)
                
                first_audio = True
                async for audio_chunk in audio_stream:
                    if first_audio:
                        if self.latency_tracker.t5 is None:
                            self.latency_tracker.mark_t5()
                        first_audio = False
                        self.state = VoiceLoopState.SPEAKING
                        
                        if self.latency_tracker.t6 is None:
                            self.latency_tracker.mark_t6()
                            
                    await self.speaker.play(audio_chunk)
                    
                full_response += phrase + ' '
                
            self.memory.add_message('assistant', full_response.strip())
            self.state = VoiceLoopState.LISTENING
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import logging
            logging.error(f"Error in run_generation_and_tts: {e}")
            self.state = VoiceLoopState.LISTENING
