# Fish AI Agent

## Purpose
To build a scalable, modular AI agent within an animatronic fish body, providing engaging and humorous interactions.

## Core Concept
An LLM-driven voice agent embodied in a physical fish. It listens via a microphone, replies using TTS, and animates its body and mouth in sync with speech and intent.

## Scope
The M0 scope covers basic project structure, configuration, abstract hardware interfaces, watchdog safety mechanisms, and basic deterministic lip-sync logic.

## Success Criteria
- Validated configuration loader.
- Watchdog enforces max runtimes and cooldowns.
- Audio chunks translate to normalized mouth positions.
- Modular architecture defined (Brain/Body split).
