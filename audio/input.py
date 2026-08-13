import asyncio
from typing import AsyncGenerator

class MockMic:
    def __init__(self, frame_delay: float = 0.01):
        self.frame_delay = frame_delay
        self._running = False
        self._queue = asyncio.Queue()

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False
        await self._queue.put(None)

    def inject_audio(self, chunk: bytes):
        self._queue.put_nowait(chunk)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=self.frame_delay)
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                yield b'\x00\x00'

class LiveMic:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._running = False
        self._queue = asyncio.Queue()
        self.stream_obj = None

        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            self.sd = None
            print("Warning: sounddevice not installed.")

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self._running:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))

    async def start(self):
        self._running = True
        self._loop = asyncio.get_running_loop()
        if self.sd:
            self.stream_obj = self.sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                blocksize=self.chunk_size,
                callback=self.callback
            )
            self.stream_obj.start()

    async def stop(self):
        self._running = False
        if self.stream_obj:
            self.stream_obj.stop()
            self.stream_obj.close()
            self.stream_obj = None
        await self._queue.put(None)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        while self._running:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
