import unittest
import asyncio
import json
import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

sys.modules["yaml"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["aiohttp.web"] = MagicMock()

from raspberry_pi.mcp_server import FishMCPServer
from raspberry_pi.safety.watchdog import MotorWatchdog
from agent.tools.registry import ToolRegistry

class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_driver = MagicMock()
        self.watchdog = MotorWatchdog(driver=self.mock_driver, max_runtime=5.0, cooldown=1.0)
        self.tool_registry = ToolRegistry()
        self.server = FishMCPServer(watchdog=self.watchdog, tool_registry=self.tool_registry)
        
    def test_tool_registration_and_schemas(self):
        schemas = self.tool_registry.get_schemas()
        names = [s["name"] for s in schemas]
        self.assertIn("fish.animate", names)
        self.assertIn("fish.wiggle", names)
        self.assertIn("fish.look", names)
        self.assertIn("fish.stop", names)
        self.assertIn("fish.get_status", names)
        
    async def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            await self.server.animate("invalid_action", 1.0)
            
        with self.assertRaises(ValueError):
            await self.server.look("invalid_direction", 1.0)
            
    async def test_watchdog_enforcement(self):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await self.server.animate("mouth", 10.0)
            mock_sleep.assert_called_with(5.0)
            
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await self.server.animate("mouth", 1.0)
        
        self.watchdog.active_motors["mouth"] = 1000
        self.watchdog.stop_all()
        result = await self.server.animate("mouth", 1.0)
        self.assertFalse(result)

    async def test_json_rpc_execution(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        res = await self.server.handle_json_rpc(req)
        self.assertEqual(res["result"]["serverInfo"]["name"], "FishMCPServer")
        self.assertEqual(res["result"]["protocolVersion"], "2024-11-05")
        self.assertIn("tools", res["result"]["capabilities"])
        
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        res = await self.server.handle_json_rpc(req)
        self.assertTrue(len(res["result"]["tools"]) > 0)
        
        with patch.object(self.server, "get_status", return_value="idle"):
            req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "fish.get_status"}}
            res = await self.server.handle_json_rpc(req)
            self.assertIn("status", res["result"]["content"][0]["text"])

    async def test_stdio_transport(self):
        reader = AsyncMock()
        writer = Mock()
        writer.drain = AsyncMock()
        
        reader.readline.side_effect = [
            b"{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"ping\"}\n",
            b""
        ]
        
        await self.server.run_stdio(reader, writer)
        
        writer.write.assert_called()
        written = writer.write.call_args[0][0]
        self.assertIn(b"\"id\": 1", written)

    async def test_http_transport(self):
        # Dynamically inject AsyncMock for the specific instances after mock object creation
        import aiohttp.web
        aiohttp.web.AppRunner.return_value.setup = AsyncMock()
        aiohttp.web.TCPSite.return_value.start = AsyncMock()
        
        task = asyncio.create_task(self.server.run_http(host="127.0.0.1", port=0))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_concurrent_task_cancellation(self):
        # Start a long animation
        animate_task = asyncio.create_task(self.server.animate("mouth", 10.0))
        await asyncio.sleep(0.01) # let it start
        self.assertTrue(len(self.server._active_tasks) > 0)
        
        # Call stop
        self.server.stop()
        
        # Await the animate task, should be cancelled cleanly
        await animate_task
        self.assertEqual(len(self.server._active_tasks), 0)
        self.mock_driver.stop_all.assert_called() # watchdog should have stopped the motor
        
    async def test_run_stdio_concurrent_stop(self):
        reader = AsyncMock()
        writer = Mock()
        writer.drain = AsyncMock()
        
        # First send animate, then stop
        reader.readline.side_effect = [
            b"{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"tools/call\", \"params\": {\"name\": \"fish.animate\", \"arguments\": {\"action\": \"head\", \"duration\": 5.0}}}\n",
            b"{\"jsonrpc\": \"2.0\", \"id\": 2, \"method\": \"tools/call\", \"params\": {\"name\": \"fish.stop\"}}\n",
            b""
        ]
        
        await self.server.run_stdio(reader, writer)
        
        # Check that responses were written
        self.assertTrue(writer.write.call_count >= 2)
        responses = [call[0][0].decode() for call in writer.write.call_args_list]
        self.assertTrue(any('"id": 2' in r for r in responses))
        self.assertTrue(any('"id": 1' in r for r in responses))

    async def test_look_direction_mapping(self):
        with patch.object(self.server.watchdog, "start_motor", return_value=True) as mock_start:
            await self.server.look("left", 0.1)
            mock_start.assert_called_with("head", -1.0)
            
            await self.server.look("right", 0.1)
            mock_start.assert_called_with("head", 1.0)

            await self.server.look("center", 0.1)
            mock_start.assert_called_with("head", 0.0)

    async def test_json_parse_error(self):
        reader = AsyncMock()
        writer = Mock()
        writer.drain = AsyncMock()
        
        reader.readline.side_effect = [
            b"invalid json\n",
            b""
        ]
        
        await self.server.run_stdio(reader, writer)
        
        writer.write.assert_called()
        written = writer.write.call_args[0][0].decode()
        self.assertIn("-32700", written)

if __name__ == "__main__":
    unittest.main()
