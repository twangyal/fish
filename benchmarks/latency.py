import time
from typing import Dict, Optional

class LatencyTracker:
    def __init__(self):
        self.t0: Optional[float] = None
        self.t1: Optional[float] = None
        self.t2: Optional[float] = None
        self.t3: Optional[float] = None
        self.t4: Optional[float] = None
        self.t5: Optional[float] = None
        self.t6: Optional[float] = None

    def mark_t0(self):
        self.t0 = time.time()

    def mark_t1(self):
        self.t1 = time.time()

    def mark_t2(self):
        self.t2 = time.time()

    def mark_t3(self):
        self.t3 = time.time()

    def mark_t4(self):
        self.t4 = time.time()

    def mark_t5(self):
        self.t5 = time.time()

    def mark_t6(self):
        self.t6 = time.time()

    def reset(self):
        self.t0 = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.t5 = None
        self.t6 = None

    def get_summary(self) -> Dict[str, Optional[float]]:
        def diff(t_end, t_start):
            if t_end is not None and t_start is not None:
                return t_end - t_start
            return None
            
        return {
            'speech_duration': diff(self.t1, self.t0),
            'asr_latency': diff(self.t2, self.t1),
            'ttft': diff(self.t3, self.t2),
            'phrase_latency': diff(self.t4, self.t3),
            'tts_latency': diff(self.t5, self.t4),
            'playback_latency': diff(self.t6, self.t5),
            'total_e2e_latency': diff(self.t6, self.t1)
        }
