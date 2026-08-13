from abc import ABC, abstractmethod

class MotorDriver(ABC):
    @abstractmethod
    def set_mouth(self, state: float) -> None:
        pass
        
    @abstractmethod
    def set_head(self, state: float) -> None:
        pass
        
    @abstractmethod
    def set_body(self, state: float) -> None:
        pass
        
    @abstractmethod
    def stop_all(self) -> None:
        pass

class MockMotorDriver(MotorDriver):
    def __init__(self):
        self.mouth = 0.0
        self.head = 0.0
        self.body = 0.0

    def set_mouth(self, state: float) -> None:
        self.mouth = state

    def set_head(self, state: float) -> None:
        self.head = state

    def set_body(self, state: float) -> None:
        self.body = state

    def stop_all(self) -> None:
        self.mouth = 0.0
        self.head = 0.0
        self.body = 0.0
