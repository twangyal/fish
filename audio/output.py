import asyncio
from typing import Callable, Optional
from audio.lip_sync import calculate_mouth_envelope

class MockSpeaker:
    def __init__(self, animation_callback: Optional[Callable[[float], None]] = None):
        self.animation_callback = animation_callback
        self.playing = False


    async def play(self, audio_chunk: bytes):
        self.playing = True
        envelope = calculate_mouth_envelope(audio_chunk)
        if self.animation_callback:
            self.animation_callback(envelope)
            
    async def stop(self):
        self.playing = False

    def clear(self):
        pass

class LiveSpeaker:
    def __init__(self, sample_rate: int = 22050, channels: int = 1, animation_callback: Optional[Callable[[float], None]] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.animation_callback = animation_callback
        self.stream_obj = None
        
        import threading
        self.lock = threading.Lock()

        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            self.sd = None
            print("Warning: sounddevice not installed.")

    def _ensure_stream(self):
        with self.lock:
            if self.sd and self.stream_obj is None:
                self.stream_obj = self.sd.RawOutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16'
                )
                self.stream_obj.start()

    async def play(self, audio_chunk: bytes):
        self._ensure_stream()
        
        envelope = calculate_mouth_envelope(audio_chunk)
        if self.animation_callback:
            self.animation_callback(envelope)
            
        if self.stream_obj:
            def write_chunk():
                with self.lock:
                    if self.stream_obj and not self.stream_obj.closed:
                        self.stream_obj.write(audio_chunk)
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, write_chunk)

    async def stop(self):
        self.clear()

    def clear(self):
        with self.lock:
            if self.stream_obj:
                self.stream_obj.stop()
                self.stream_obj.close()
                self.stream_obj = None
