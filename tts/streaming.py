import re
from typing import AsyncGenerator

class TTSStreamBuffer:
    def __init__(self):
        self.buffer = ''
        self.punctuation = {'.', '?', '!', ',', ';'}

    def clear(self):
        self.buffer = ''

    async def process_tokens(self, token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for token in token_stream:
            self.buffer += token
            
            last_punc_idx = -1
            for i, char in enumerate(self.buffer):
                if char in self.punctuation:
                    last_punc_idx = i
            
            if last_punc_idx != -1:
                phrase = self.buffer[:last_punc_idx+1].strip()
                self.buffer = self.buffer[last_punc_idx+1:]
                if phrase:
                    yield phrase
                    
        if self.buffer.strip():
            yield self.buffer.strip()
            self.buffer = ''
