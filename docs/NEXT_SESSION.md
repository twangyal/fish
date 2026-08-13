# Next Session

## Current Status
- **Option A (Local AI Engines):** Complete & verified (`WhisperASR`, `LlamaLLM`, `PiperTTS`, `LiveMic`, `LiveSpeaker`).
- **Option B (Cloud AI Engines):** Complete & verified (`OpenAIWhisperASR`, `OpenAILLM`, `OpenAITTS`, `ElevenLabsTTS`, `.env` loader, `target: "cloud"`).
- **General AI Assistant Architecture:** Complete & verified (`PlanningBrain`, `ShortTermMemory`, `LongTermMemory`, `ToolRegistry`, `AgentLogger`, `VoiceLoop` notification worker).
- **FastMCP Server & Remote Tool Bridge:** Complete & verified (`FishMCPServer`, stdio/HTTP transports, JSON-RPC 2.0, `MotorWatchdog` integration).
- **Test Suite:** 44/44 unit tests passing cleanly.

## Recommended Next Milestones

1. **Physical Hardware & PCA9685 Servo Drivers (`raspberry_pi/hardware/`)**:
   - Implement `PCA9685MotorDriver` using I2C PWM servo control for physical head flap, tail wiggle, and mouth lip-sync motors.
   - Connect RMS `calculate_mouth_envelope` directly to mouth PWM servo duty cycles.

2. **Comparative Latency & Benchmark Suite (`benchmarks/latency.py`)**:
   - Run automated T0-T6 latency and throughput benchmarks comparing Local (Option A) vs Cloud (Option B) engine stacks.

3. **Raspberry Pi Deployment & Live Field Testing**:
   - Configure deployment scripts, systemd service definitions, and environment wrappers for running live on physical Raspberry Pi hardware.



