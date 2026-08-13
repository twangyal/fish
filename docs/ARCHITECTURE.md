# Architecture Overview

## Brain/Body Split
The system is divided into two main components:
- **Body**: The Raspberry Pi hardware interface handling motors, sensors, and basic safety.
- **Brain**: The logical controller running the LLM, TTS, ASR, and conversation flow. This can run locally or remotely.

## Process Communication
- **MCP for Tools**: The Brain uses Model Context Protocol (MCP) to control the Body's tools, such as `fish.animate`, `fish.wiggle`, `fish.look`, and `fish.stop`.
- **Sockets for Audio**: Audio streams (PCM data) are transmitted via TCP/UDP sockets for low-latency ASR and TTS playback.

## Deterministic Mouth Sync
Audio playback is synchronized with the motor mouth by analyzing the audio's RMS amplitude in chunks (lip sync). The Body uses `calculate_mouth_envelope` to turn audio chunks into normalized motor states (0.0 to 1.0) and updates the mouth motor appropriately.

## Safety Invariants
- **Motor Watchdog**: The `MotorWatchdog` monitors all motor runtimes.
- **Timeouts & Cooldowns**: Motors are automatically stopped after a max runtime. Motors placed in cooldown cannot be restarted immediately.
- **Emergency Stop**: A system-level `stop_all` can immediately zero out all motor signals.

## Voice-to-Voice Loop
The `VoiceLoop` manages the real-time audio pipeline:
- **State Machine**: Transitions through `LISTENING` -> `PROCESSING_ASR` -> `GENERATING` -> `SPEAKING`.
- **Barge-in / Interruption**: If the `MockVAD` detects `SPEECH_START` during `GENERATING` or `SPEAKING`, it cancels the active LLM/TTS async tasks, stops the speaker, flushes the audio buffers, and returns to `LISTENING`.

## Latency Breakdown (T0-T6)
- **T0**: Speech Start (VAD triggered)
- **T1**: Speech End (VAD triggered)
- **T2**: ASR Complete
- **T3**: LLM First Token (TTFT)
- **T4**: First Phrase Ready
- **T5**: TTS First Audio Sample
- **T6**: Audio Playback Start
