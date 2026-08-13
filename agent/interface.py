from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

class ASR(ABC):
    @abstractmethod
    async def transcribe(self, audio_stream: AsyncGenerator[bytes, None]) -> str:
        pass

class LLM(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        pass

class TTS(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        pass

class ConversationController(ABC):
    @abstractmethod
    async def start(self) -> None:
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        pass

    @abstractmethod
    def add_assistant_message(self, content: str) -> None:
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        pass
