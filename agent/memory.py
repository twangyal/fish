import json
import os
import time
import asyncio
from typing import List, Dict, Any, Optional

class ShortTermMemory:
    def __init__(self, window_size: int = 10, system_prompt: str = ""):
        self.window_size = window_size
        self.system_prompt = system_prompt
        self.history = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def get_context(self) -> List[Dict[str, str]]:
        context = []
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})
        context.extend(self.history)
        return context
        
    def clear(self):
        self.history = []

class LongTermMemory:
    def __init__(self, file_path: str = "models/memory.json"):
        self.file_path = file_path
        self._lock = asyncio.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            with open(self.file_path, 'r') as f:
                try:
                    self.facts = json.load(f)
                except json.JSONDecodeError:
                    self.facts = {}
        else:
            self.facts = {}

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        tmp_file = self.file_path + '.tmp'
        with open(tmp_file, 'w') as f:
            json.dump(self.facts, f, indent=2)
        os.replace(tmp_file, self.file_path)

    async def _save_async(self):
        await asyncio.to_thread(self._save)

    async def _load_async(self):
        await asyncio.to_thread(self._load)

    async def save_fact(self, key: str, value: Any, tags: List[str] = None):
        async with self._lock:
            self.facts[key] = {
                "value": value,
                "tags": tags or [],
                "timestamp": time.time()
            }
            await self._save_async()

    async def recall_facts(self, query: str = None) -> Dict[str, Any]:
        async with self._lock:
            await self._load_async()
            if not query:
                return self.facts
                
            query_lower = query.lower()
            results = {}
            for k, v in self.facts.items():
                if query_lower in k.lower() or                any(query_lower in tag.lower() for tag in v["tags"]) or                query_lower in str(v["value"]).lower():
                    results[k] = v
            return results

    async def delete_fact(self, key: str):
        async with self._lock:
            if key in self.facts:
                del self.facts[key]
                await self._save_async()
                return True
            return False
