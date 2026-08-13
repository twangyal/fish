from enum import Enum
from typing import Optional

class VADEvent(Enum):
    SPEECH_START = 'SPEECH_START'
    SPEECH_END = 'SPEECH_END'

class MockVAD:
    def __init__(self):
        self.is_speaking = False
        
    def analyze(self, frame: bytes) -> Optional[VADEvent]:
        return None
