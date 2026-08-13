from typing import List, Dict
from agent.interface import ConversationController

class DefaultConversationController(ConversationController):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def add_user_message(self, text: str):
        self.history.append({'role': 'user', 'content': text})

    def add_assistant_message(self, text: str):
        self.history.append({'role': 'assistant', 'content': text})

    def get_prompt(self) -> str:
        return '\n'.join([f"{m['role']}: {m['content']}" for m in self.history])
