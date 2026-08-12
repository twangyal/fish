# Architecture

## System Overview

This repository implements a local on-device voice assistant pipeline.

## Components

- `wake_word.py`: Uses `pvporcupine` for wake word detection.
- `input_speech.py`: Uses `pvcheetah` for Speech-to-Text (STT).
- `llm_inference.py`: Uses `picollm` for local LLM text generation.
- `output_speech.py`: Uses `pvorca` for Text-to-Speech (TTS).

## Boundaries

Document important dependency, authority, persistence, I/O, or trust boundaries.

## Data Flow

The pipeline follows this sequence:
audio input -> wake word detection -> STT -> LLM -> TTS -> audio output

## Invariants

List rules that must remain structurally true.

## Testing Strategy

The current verification approach uses the manual test script `test.py`.
