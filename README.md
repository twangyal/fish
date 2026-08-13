# 🐟 Reprogrammed AI Singing Fish Plaque

> **Giving a classic wall-mounted animatronic singing fish a modern AI brain.**

Welcome to the **Embedded Voice-to-Voice AI Singing Fish** project! We took a retro novelty singing fish plaque (think Big Mouth Billy Bass) and turned it into an interactive, voice-driven AI assistant. Instead of playing canned 90s songs, this fish listens to you, thinks with an LLM, speaks back in real time, and wiggles its head, body, and mouth in sync with its voice.

---

## 💡 How It Works

The system is built on a decoupled **Brain / Body** architecture:

```
                  +-------------------------------------------------+
                  |                   THE BRAIN                     |
                  |  (Runs locally or in the cloud - PC/Server/Pi)  |
                  +-----------------------+-------------------------+
                                          |
                        ASR (Whisper)     |     TTS Audio & RMS
                        VAD (Silero/Mock) |     Motor Commands (MCP)
                                          v
                  +-------------------------------------------------+
                  |                   THE BODY                      |
                  |             (Raspberry Pi Animatronics)         |
                  +-----------------------+-------------------------+
                                          |
                +-------------------------+-------------------------+
                |                         |                         |
                v                         v                         v
        Mouth Motor (Lip Sync)    Head Motor (Look)        Body/Tail (Wiggle)
```

### 1. Dual AI Engine Stacks
You can switch seamlessly between local offline execution and cloud APIs depending on your hardware and network setup:
- **Option A (Local AI Stack)**: 100% offline, zero-telemetry pipeline using Whisper for ASR, Llama via `picoLLM` for reasoning, and `Piper` for speech synthesis.
- **Option B (Cloud AI Stack)**: High-speed cloud pipeline using OpenAI Whisper, GPT-4o, and ElevenLabs / OpenAI TTS for hyper-realistic voices.

### 2. Real-Time Voice Loop & Barge-In
The central `VoiceLoop` orchestrates the audio lifecycle (`LISTENING` → `PROCESSING_ASR` → `GENERATING` → `SPEAKING`).
- **Barge-in Support**: Interrupt the fish anytime while it is talking! If speech is detected while the fish is generating or speaking, it instantly cancels active LLM/TTS tasks, halts the motors, flushes the audio queue, and switches back to listening.
- **Deterministic Lip-Sync**: Audio chunks are processed through an RMS (Root Mean Square) envelope filter (`calculate_mouth_envelope`) that maps voice volume directly to mouth motor position (0.0 to 1.0) in real time.

### 3. Remote Tool Bridge (FastMCP Server)
The fish exposes its physical controls over the **Model Context Protocol (MCP)** using standard JSON-RPC 2.0 over both `stdio` and `HTTP/SSE`:
- `fish.animate(action, duration)`: Triggers animatronic sequences across mouth, head, and body.
- `fish.wiggle(duration)`: Wiggles the fish body/tail.
- `fish.look(direction, duration)`: Moves the head (`left`, `right`, `center`, `forward`, `up`, `down`).
- `fish.stop()`: Emergency stop that halts all motors and cancels active animation tasks immediately.
- `fish.get_status()`: Returns live motor and watchdog safety status.

### 4. Safety First (Motor Watchdog)
Physical animatronic motors can burn out if held on indefinitely. The built-in `MotorWatchdog` enforces:
- **Runtime Caps**: Automatically cuts motor power after a configurable max runtime (e.g., 5 seconds).
- **Thermal Cooldowns**: Enforces a required rest period (e.g., 2 seconds) before a motor can be reactivated.
- **Emergency Stop**: Instantly zeroes out motor signals and cancels background animation timers.

---

## 📂 Codebase Structure

```text
├── agent/                  # The Brain logic
│   ├── brain.py            # PlanningBrain (Fast path & ReAct tool calling)
│   ├── conversation.py     # State management & turn history
│   ├── llm.py              # LLM interfaces (Local Llama & OpenAI GPT)
│   ├── memory.py           # ShortTermMemory & LongTermMemory (JSON fact store)
│   ├── logger.py           # Structured event logging
│   └── tools/              # Tool registry & built-in tool implementations
├── raspberry_pi/           # The Body hardware & animatronics
│   ├── animation/          # Lip-sync RMS envelope calculation
│   ├── hardware/           # Motor driver abstractions (MockMotorDriver, PCA9685)
│   ├── mcp_server.py       # FastMCP Server (stdio & HTTP/SSE JSON-RPC 2.0)
│   └── safety/             # MotorWatchdog (runtime caps & cooldown enforcement)
├── asr/                    # Speech-to-Text (Whisper Local & Cloud)
├── tts/                    # Text-to-Speech (Piper Local, OpenAI, ElevenLabs)
├── audio/                  # Audio input/output streams (LiveMic, LiveSpeaker)
├── config/                 # YAML configuration (agent.yaml)
├── tests/                  # Complete unit test suite (44/44 passing)
└── voice_loop.py           # Real-time voice-to-voice orchestrator & state machine
```

---

## 🛠️ Complete Setup & Getting Started

### 1. Prerequisites & System Dependencies

Make sure Python 3.10+ is installed. PortAudio and FFmpeg are required for microphone input, speaker output, and audio decoding.

#### macOS
```bash
brew install portaudio ffmpeg
```

#### Linux / Raspberry Pi OS
```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pyaudio portaudio19-dev ffmpeg libasound2-dev
```

---

### 2. Repository & Virtual Environment Setup

1. Clone the repository and navigate into the project root:
   ```bash
   git clone https://github.com/your-username/fish.git
   cd fish
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade `pip` and install project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### 3. Configuration & API Keys Setup

1. **Environment Variables (`.env`)**:
   If using **Option B (Cloud AI Stack)**, create a `.env` file in the project root:
   ```bash
   cp .env.example .env  # or create a new .env file
   ```
   Add your API keys:
   ```env
   OPENAI_API_KEY=sk-proj-your-openai-api-key-here
   ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
   ```

2. **Agent Configuration (`config/agent.yaml`)**:
   Adjust host, ports, safety thresholds, and target brain engine in `config/agent.yaml`:
   ```yaml
   mcp:
     host: "0.0.0.0"
     port: 8000
   audio:
     port: 8001
   safety:
     max_motor_runtime_sec: 5.0
     motor_cooldown_sec: 2.0
     emergency_stop_timeout_sec: 1.0
   brain:
     target: "cloud"  # Change to "local" for offline local stack (Option A)
   models:
     path: "./models"
   ```

---

### 4. Running the System

#### Option A: Run the Voice-to-Voice Loop
Launch the main conversational voice pipeline:
```bash
venv/bin/python voice_loop.py
```

#### Option B: Run the FastMCP Remote Tool Server
Launch the animatronics MCP tool server over HTTP/SSE or stdio for remote agent control:
```bash
# HTTP/SSE mode (listens on 0.0.0.0:8000)
venv/bin/python -m raspberry_pi.mcp_server

# stdio mode (for local subprocess integration)
venv/bin/python -c "import asyncio, sys; from raspberry_pi.mcp_server import FishMCPServer; from raspberry_pi.safety.watchdog import MotorWatchdog; from raspberry_pi.hardware.motor_driver import MockMotorDriver; s = FishMCPServer(MotorWatchdog(MockMotorDriver(), 5.0, 2.0)); asyncio.run(s.run_stdio_async())"
```

---

### 5. Running Verification & Unit Tests

Verify all 44 unit tests across the memory store, brain, engine stacks, safety watchdog, and FastMCP server:
```bash
venv/bin/python -m unittest discover tests
```

Expected output:
```text
Ran 44 tests in ~1.0s
OK
```

---

## 🗺️ Future Milestones

- [x] **Option A: Local AI Engine Stack** (Whisper ASR, Llama LLM, Piper TTS, Live Audio).
- [x] **Option B: Cloud AI Engine Stack** (OpenAI Whisper, GPT-4o, ElevenLabs TTS).
- [x] **Autonomous Agent Brain** (Short & Long-term memory, ReAct tool execution, structured logging).
- [x] **FastMCP Server & Remote Tool Bridge** (stdio & HTTP/SSE MCP endpoints with safety watchdog).
- [ ] **Physical PCA9685 Hardware Driver Integration**: Connecting I2C PWM servo drivers to physical Big Mouth Billy Bass motors on Raspberry Pi hardware.
- [ ] **Comparative Latency Benchmark Suite**: Automated T0–T6 latency profiling comparing Local vs. Cloud engine stacks under real network conditions.
- [ ] **Raspberry Pi Deployment**: Systemd daemon configuration and physical plaque mounting.

---

## 📄 License

MIT License.
