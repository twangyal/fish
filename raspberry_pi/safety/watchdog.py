import time
from typing import Dict
from ..hardware.motor_driver import MotorDriver

class MotorWatchdog:
    def __init__(self, driver: MotorDriver, max_runtime: float, cooldown: float):
        self.driver = driver
        self.max_runtime = max_runtime
        self.cooldown = cooldown
        self.active_motors: Dict[str, float] = {}
        self.cooldown_motors: Dict[str, float] = {}

    def start_motor(self, name: str, state: float, start_time: float = None) -> bool:
        current_time = start_time if start_time is not None else time.time()
        
        if state == 0.0:
            self.stop_motor(name)
            return True

        # Check cooldown
        if name in self.cooldown_motors:
            if current_time - self.cooldown_motors[name] < self.cooldown:
                return False
            else:
                del self.cooldown_motors[name]

        if name not in self.active_motors:
            self.active_motors[name] = current_time
        self._set_motor(name, state)
        return True

    def stop_motor(self, name: str) -> None:
        self._set_motor(name, 0.0)
        if name in self.active_motors:
            del self.active_motors[name]

    def _set_motor(self, name: str, state: float):
        if name == 'mouth':
            self.driver.set_mouth(state)
        elif name == 'head':
            self.driver.set_head(state)
        elif name == 'body':
            self.driver.set_body(state)

    def check_and_enforce(self, current_time: float = None) -> None:
        current_time = current_time if current_time is not None else time.time()
        expired = []
        for name, start_time in self.active_motors.items():
            if current_time - start_time > self.max_runtime:
                expired.append(name)
        
        for name in expired:
            self._set_motor(name, 0.0)
            self.cooldown_motors[name] = current_time
            del self.active_motors[name]

    def stop_all(self, current_time: float = None) -> None:
        current_time = current_time if current_time is not None else time.time()
        self.driver.stop_all()
        for name in list(self.active_motors.keys()):
            self.cooldown_motors[name] = current_time
        self.active_motors.clear()
        
    def emergency_stop(self) -> None:
        self.stop_all()
