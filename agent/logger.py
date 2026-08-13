import json
import logging
from typing import Any, Dict

class AgentLogger:
    def __init__(self, log_file: str = "agent.log"):
        self.logger = logging.getLogger("AgentLogger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        event = {
            "type": event_type,
            "payload": payload
        }
        self.logger.info(json.dumps(event))

    def thought(self, content: str):
        self.log_event("THOUGHT", {"content": content})

    def action(self, tool_name: str, arguments: Dict[str, Any]):
        self.log_event("ACTION", {"tool": tool_name, "arguments": arguments})

    def observation(self, tool_name: str, result: Any):
        self.log_event("OBSERVATION", {"tool": tool_name, "result": result})

    def metric(self, name: str, value: float, unit: str = ""):
        self.log_event("METRIC", {"name": name, "value": value, "unit": unit})

    def state_change(self, old_state: str, new_state: str, reason: str = ""):
        self.log_event("STATE_CHANGE", {"old": old_state, "new": new_state, "reason": reason})

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
