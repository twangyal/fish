import asyncio
import json
from typing import Optional, Dict, Any

class FishMCPServer:
    def __init__(self, watchdog, tool_registry=None, config_path: str = "config/agent.yaml"):
        self.watchdog = watchdog
        self.tool_registry = tool_registry
        self.config_path = config_path
        self.state = "idle"
        self._active_tasks = set()
        
        if self.tool_registry:
            self._register_tools()

    def _register_tools(self):
        @self.tool_registry.register(
            name="fish.animate",
            description="Animate the fish with a given action.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "duration": {"type": "number", "default": 1.0}
                },
                "required": ["action"]
            }
        )
        async def animate(action: str, duration: float = 1.0):
            return await self.animate(action, duration)

        @self.tool_registry.register(
            name="fish.wiggle",
            description="Wiggle the fish.",
            parameters={
                "type": "object",
                "properties": {
                    "duration": {"type": "number", "default": 1.0}
                }
            }
        )
        async def wiggle(duration: float = 1.0):
            return await self.wiggle(duration)

        @self.tool_registry.register(
            name="fish.look",
            description="Move the head to look in a direction.",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {"type": "string"},
                    "duration": {"type": "number", "default": 1.0}
                },
                "required": ["direction"]
            }
        )
        async def look(direction: str, duration: float = 1.0):
            return await self.look(direction, duration)

        @self.tool_registry.register(
            name="fish.stop",
            description="Stop all motors.",
            parameters={"type": "object", "properties": {}}
        )
        async def stop():
            self.stop()
            return {"status": "stopped"}

        @self.tool_registry.register(
            name="fish.get_status",
            description="Get the status of the fish.",
            parameters={"type": "object", "properties": {}}
        )
        async def get_status():
            return {"status": self.get_status()}

    async def animate(self, action: str, duration: float = 1.0) -> bool:
        if action not in ["mouth", "body", "head"]:
            raise ValueError(f"Invalid action: {action}")
        duration = min(duration, self.watchdog.max_runtime)
        
        task = asyncio.current_task()
        if task: self._active_tasks.add(task)
        try:
            if not self.watchdog.start_motor(action, 1.0):
                return False
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                pass
            finally:
                self.watchdog.stop_motor(action)
            return True
        finally:
            if task: self._active_tasks.discard(task)

    async def wiggle(self, duration: float = 1.0) -> bool:
        duration = min(duration, self.watchdog.max_runtime)
        
        task = asyncio.current_task()
        if task: self._active_tasks.add(task)
        try:
            if not self.watchdog.start_motor("body", 1.0):
                return False
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                pass
            finally:
                self.watchdog.stop_motor("body")
            return True
        finally:
            if task: self._active_tasks.discard(task)

    async def look(self, direction: str, duration: float = 1.0) -> bool:
        mapping = {"left": -1.0, "right": 1.0, "center": 0.0, "forward": 0.5, "up": 1.0, "down": -1.0}
        if direction not in mapping:
            raise ValueError(f"Invalid direction: {direction}")
        state = mapping[direction]
        duration = min(duration, self.watchdog.max_runtime)
        
        task = asyncio.current_task()
        if task: self._active_tasks.add(task)
        try:
            if not self.watchdog.start_motor("head", state):
                return False
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                pass
            finally:
                self.watchdog.stop_motor("head")
            return True
        finally:
            if task: self._active_tasks.discard(task)

    def get_status(self) -> str:
        return self.state

    def stop(self) -> None:
        self.watchdog.stop_all()
        for task in list(self._active_tasks):
            task.cancel()

    async def handle_json_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict):
            return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}
            
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0", 
                "id": req_id, 
                "result": {
                    "serverInfo": {"name": "FishMCPServer", "version": "1.0.0"},
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}}
                }
            }
        elif method == "tools/list":
            if self.tool_registry:
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.tool_registry.get_schemas()}}
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": []}}
        elif method == "tools/call":
            if self.tool_registry:
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                try:
                    result = await self.tool_registry.execute(tool_name, tool_args)
                    return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}
                except Exception as e:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    async def run_stdio(self, reader, writer):
        writer_lock = asyncio.Lock()
        
        async def process_line(line):
            try:
                request = json.loads(line.decode().strip())
                response = await self.handle_json_rpc(request)
                async with writer_lock:
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
            except json.JSONDecodeError:
                error_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
                async with writer_lock:
                    writer.write((json.dumps(error_resp) + "\n").encode())
                    await writer.drain()
            except Exception as e:
                pass
                
        tasks = set()
        while True:
            line = await reader.readline()
            if not line:
                break
            task = asyncio.create_task(process_line(line))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_http(self, host: str = None, port: int = None):
        import yaml
        from aiohttp import web

        if host is None or port is None:
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                    mcp = config.get("mcp", {})
                    host = host or mcp.get("host", "127.0.0.1")
                    port = port or mcp.get("port", 8000)
            except Exception:
                host = host or "127.0.0.1"
                port = port or 8000
                
        async def handle_post(request):
            try:
                data = await request.json()
                response = await self.handle_json_rpc(data)
                return web.json_response(response)
            except json.JSONDecodeError:
                return web.json_response({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})
            
        async def handle_sse(request):
            response = web.StreamResponse(
                status=200,
                reason="OK",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            await response.prepare(request)
            await response.write(b"event: endpoint\ndata: /message\n\n")
            
            try:
                while True:
                    await asyncio.sleep(15)
                    await response.write(b": keepalive\n\n")
            except asyncio.CancelledError:
                pass
            return response

        app = web.Application()
        app.router.add_post("/message", handle_post)
        app.router.add_post("/", handle_post)
        app.router.add_get("/sse", handle_sse)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        while True:
            await asyncio.sleep(3600)
