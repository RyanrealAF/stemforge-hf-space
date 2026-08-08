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

Audio source separation + MIDI transcription.

## Current pipeline

1. Upload audio.
2. Demucs `htdemucs_6s` separates guitar, piano, bass, drums, vocals and other.
3. Each selected pitched stem is passed independently to Spotify Basic Pitch.
4. WAV stems and MIDI files are packaged into a ZIP.

## Important fix

Basic Pitch 0.4.x requires a model path as the seventh argument to
`predict_and_save`. This Space imports `ICASSP_2022_MODEL_PATH` and passes it
explicitly.

## Upload behavior

The processor accepts the filepath returned by `gr.Audio(type="filepath")`.
If no file path reaches the function, the UI reports that condition instead of
failing deep inside the GPU worker.

## Hardware

Use a GPU Space for practical Demucs processing. ZeroGPU can be used with the
`spaces.GPU` decorator if the Space's runtime supports the installed stack.

## API

The Gradio function is exposed as `process_song` for later connection to the
StemForge ChatGPT App.

## Drum MIDI

Drum separation is supported. Drum MIDI remains experimental because Basic
Pitch is a pitched-note transcription model. A dedicated automatic drum
transcription model should be added for production-quality GM drum MIDI.
