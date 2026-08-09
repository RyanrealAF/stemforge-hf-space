import os
from pathlib import Path
import gradio as gr

from worker import process_song_worker


def resolve_audio_file(audio_file):
    """
    Gradio Audio(type='filepath') normally returns a string path.
    This also handles dictionary-style file representations defensively.
    """

    if audio_file is None:
        return None

    if isinstance(audio_file, (str, os.PathLike)):
        path = Path(audio_file)

        if path.is_file():
            return path

        return None

    if isinstance(audio_file, dict):
        for key in ("path", "name"):
            value = audio_file.get(key)

            if value:
                path = Path(value)

                if path.is_file():
                    return path

    return None


def process_song(
    audio_file,
    guitar,
    piano,
    bass,
    drums,
    progress=gr.Progress(),
):

    source = resolve_audio_file(audio_file)

    if source is None:
        raise gr.Error(
            "No audio file was received. "
            "Upload an MP3, WAV, FLAC, or M4A file and wait for "
            "the waveform to appear before starting."
        )

    selected = []

    if guitar:
        selected.append("guitar")

    if piano:
        selected.append("piano")

    if bass:
        selected.append("bass")

    if drums:
        selected.append("drums")

    if not selected:
        raise gr.Error(
            "Select at least one instrument."
        )

    try:

        zip_path = process_song_worker(
            str(source),
            selected,
            progress,
        )

        status_msg = (
            f"### ✅ StemForge complete\n\n"
            f"Generated separated audio and MIDI for: "
            f"**{', '.join(selected)}**\n\n"
            f"Your single ZIP contains only the generated "
            f"stems and MIDI files."
        )

        return zip_path, status_msg

    except Exception as exc:

        raise gr.Error(
            f"Processing failed: {exc}"
        )


# =============================================================
# GRADIO UI
# =============================================================

with gr.Blocks(
    title="StemForge",
) as demo:

    gr.Markdown(
        """
# 🎛️ StemForge

### Song → Instrument Stems → MIDI

Upload a song and StemForge will:

1. Separate the instruments with **Demucs htdemucs_6s**
2. Extract guitar, piano, bass, and/or drums
3. Convert each selected stem to MIDI with **Basic Pitch**
4. Package everything into **one ZIP file**

The original song is **not** included in the results.

⚠️ **Drum MIDI is experimental.** The drum WAV separation is supported,
but Basic Pitch is primarily a pitched-instrument transcription system.
"""
    )

    # ---------------------------------------------------------
    # AUDIO INPUT
    # ---------------------------------------------------------

    audio = gr.Audio(
        label="Source Song",
        type="filepath",
        sources=["upload"],
    )

    # ---------------------------------------------------------
    # INSTRUMENT SELECTION
    # ---------------------------------------------------------

    gr.Markdown(
        "### Select instruments"
    )

    with gr.Row():

        guitar = gr.Checkbox(
            value=True,
            label="🎸 Guitar",
        )

        piano = gr.Checkbox(
            value=True,
            label="🎹 Piano",
        )

        bass = gr.Checkbox(
            value=True,
            label="🎸 Bass",
        )

        drums = gr.Checkbox(
            value=True,
            label="🥁 Drums",
        )

    # ---------------------------------------------------------
    # PROCESS BUTTON
    # ---------------------------------------------------------

    run_button = gr.Button(
        "🔥 Separate + Transcribe",
        variant="primary",
        size="lg",
    )

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    status = gr.Markdown()

    # ---------------------------------------------------------
    # ONLY DOWNLOAD
    # ---------------------------------------------------------

    zip_file = gr.File(
        label="📦 StemForge Results",
        file_count="single",
    )

    # ---------------------------------------------------------
    # EVENT
    # ---------------------------------------------------------

    run_button.click(
        fn=process_song,

        inputs=[
            audio,
            guitar,
            piano,
            bass,
            drums,
        ],

        outputs=[
            zip_file,
            status,
        ],

        api_name="process_song",
    )


# =============================================================
# LAUNCH
# =============================================================

if __name__ == "__main__":

    demo.queue(
        max_size=4
    ).launch()
