---
title: StemForge
emoji: 🎛️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
---

# StemForge

Upload a song and generate:

- guitar WAV stem
- piano WAV stem
- bass WAV stem
- drums WAV stem
- MIDI transcription for each selected instrument
- ZIP containing all generated files

## Models

### Demucs

Uses `htdemucs_6s`, the experimental six-source Demucs model that exposes:

- drums
- bass
- other
- vocals
- piano
- guitar

The model documentation notes that guitar separation is usable while piano separation can contain significant bleed/artifacts.

### Basic Pitch

Each isolated instrument stem is passed independently to Spotify Basic Pitch.

Basic Pitch is polyphonic and instrument-agnostic and generally performs best when one instrument is isolated before transcription.

## Hardware

For a first test, use a GPU Space.

ZeroGPU can also work if the Space is configured for ZeroGPU and the `spaces.GPU` decorator is available to the runtime. If ZeroGPU gives compatibility or memory problems, switch the Space to a dedicated NVIDIA GPU.

## API

The Gradio Space automatically exposes an API endpoint for `process_song`.

The generated API can be inspected from the Space's "Use via API" panel.

A ChatGPT app can call the Space as its heavy-processing backend.

## Important limitation

The current drum MIDI output is experimental. Basic Pitch is a pitch transcription model, not a dedicated drum transcription system. A future version should replace the drum transcription stage with a dedicated automatic drum transcription model and map the result to General MIDI percussion notes.

## Privacy

Uploaded audio is processed by this Space. Do not deploy this publicly until you have decided how files are retained, deleted, and authenticated.
