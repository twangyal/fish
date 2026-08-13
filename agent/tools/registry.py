from typing import Dict, Callable, Any, Optional
import inspect
import asyncio

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any]):
        def decorator(func: Callable):
            self._tools[name] = func
            self._schemas[name] = {
                "name": name,
                "description": description,
                "parameters": parameters
            }
            return func
        return decorator

    async def execute(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found.")
        func = self._tools[name]
        if inspect.iscoroutinefunction(func):
            return await func(**args)
        else:
            return await asyncio.to_thread(func, **args)

    def get_schemas(self) -> list:
        return list(self._schemas.values())
        
    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)
