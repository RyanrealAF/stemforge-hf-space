import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import torch

try:
    import spaces

    GPU = spaces.GPU
except Exception:
    def GPU(fn):
        return fn

from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH


APP_ROOT = Path(__file__).parent
OUTPUT_ROOT = APP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)


def run_cmd(cmd):
    """Run a subprocess and raise a useful error if it fails."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stdout[-12000:])

    return process.stdout


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


def find_stem(root, instrument):
    """Find a Demucs-generated WAV stem."""

    matches = list(root.rglob(f"{instrument}.wav"))

    if not matches:
        available = [p.name for p in root.rglob("*.wav")]

        raise FileNotFoundError(
            f"Demucs did not produce {instrument}.wav. "
            f"Available WAV files: {available}"
        )

    return matches[0]


def transcribe_stem(stem_path, output_dir, instrument):
    """
    Convert one isolated instrument stem to MIDI using Basic Pitch.
    """

    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        [str(stem_path)],
        str(midi_dir),
        True,   # save_midi
        False,  # sonify_midi
        False,  # save_model_outputs
        False,  # save_notes
        ICASSP_2022_MODEL_PATH,
    )

    generated_midis = list(midi_dir.glob("*.mid"))

    if not generated_midis:
        raise RuntimeError(
            f"Basic Pitch did not produce MIDI for {instrument}."
        )

    target = midi_dir / f"{instrument}.mid"

    if generated_midis[0] != target:
        shutil.copy2(generated_midis[0], target)

    return target


@GPU
def process_song(
    audio_file,
    guitar,
    piano,
    bass,
    drums,
    progress=gr.Progress(),
):
    """
    Main StemForge pipeline:

    Song
      ↓
    Demucs htdemucs_6s
      ↓
    Selected instrument stems
      ↓
    Basic Pitch
      ↓
    MIDI
      ↓
    One ZIP
    """

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

    # Create unique temporary job directory.
    job_id = next(tempfile._get_candidate_names())

    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy source into temporary working directory.
    input_path = job_dir / source.name

    shutil.copy2(
        source,
        input_path,
    )

    progress(
        0.05,
        desc="Preparing audio...",
    )

    # ---------------------------------------------------------
    # DEMUCS
    # ---------------------------------------------------------

    demucs_output = job_dir / "demucs"

    demucs_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    progress(
        0.10,
        desc=f"Separating instruments on {device.upper()}...",
    )

    run_cmd(
        [
            "python",
            "-m",
            "demucs",

            "-n",
            "htdemucs_6s",

            "--device",
            device,

            "--out",
            str(demucs_output),

            "--float32",

            str(input_path),
        ]
    )

    # Locate Demucs output.
    model_dir = (
        demucs_output
        / "htdemucs_6s"
    )

    if not model_dir.exists():
        raise RuntimeError(
            "Demucs completed but its output directory "
            "could not be found."
        )

    song_dirs = [
        directory
        for directory in model_dir.iterdir()
        if directory.is_dir()
    ]

    if not song_dirs:
        raise RuntimeError(
            "Demucs completed without creating a song directory."
        )

    separated_dir = song_dirs[0]

    # ---------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ---------------------------------------------------------

    stems_dir = job_dir / "stems"
    midi_dir = job_dir / "midi"

    stems_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    midi_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = []

    # ---------------------------------------------------------
    # INSTRUMENT PROCESSING
    # ---------------------------------------------------------

    total = len(selected)

    for index, instrument in enumerate(selected):

        progress_value = (
            0.25
            + (
                index
                / total
            )
            * 0.60
        )

        progress(
            progress_value,
            desc=f"Processing {instrument}...",
        )

        # Find separated WAV.
        separated_stem = find_stem(
            separated_dir,
            instrument,
        )

        # Copy only the separated stem.
        final_stem = (
            stems_dir
            / f"{instrument}.wav"
        )

        shutil.copy2(
            separated_stem,
            final_stem,
        )

        generated_files.append(
            final_stem
        )

        # -----------------------------------------------------
        # MIDI
        # -----------------------------------------------------

        progress(
            progress_value + 0.04,
            desc=f"Creating {instrument} MIDI...",
        )

        midi_file = transcribe_stem(
            final_stem,
            job_dir,
            instrument,
        )

        # Move MIDI into clean final directory.
        final_midi = (
            midi_dir
            / f"{instrument}.mid"
        )

        if midi_file != final_midi:
            shutil.copy2(
                midi_file,
                final_midi,
            )

        generated_files.append(
            final_midi
        )

    # ---------------------------------------------------------
    # ZIP
    # ---------------------------------------------------------

    progress(
        0.90,
        desc="Packaging stems and MIDI...",
    )

    zip_path = (
        job_dir
        / "StemForge-results.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        for file_path in generated_files:

            archive.write(
                file_path,
                file_path.relative_to(
                    job_dir
                ),
            )

    progress(
        1.0,
        desc="Complete.",
    )

    return (
        str(zip_path),
        (
            f"### ✅ StemForge complete\n\n"
            f"Generated separated audio and MIDI for: "
            f"**{', '.join(selected)}**\n\n"
            f"Your single ZIP contains only the generated "
            f"stems and MIDI files."
        ),
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
    ).launch()        raise gr.Error(
            "No audio file reached the processor. "
            "Upload an MP3/WAV/FLAC/M4A and wait for the waveform to appear "
            "before pressing Separate + Transcribe."
        )

    selected = [
        name for name, enabled in (
            ("guitar", guitar),
            ("piano", piano),
            ("bass", bass),
            ("drums", drums),
        )
        if enabled
    ]
    if not selected:
        raise gr.Error("Select at least one instrument.")

    job_id = next(tempfile._get_candidate_names())
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / source.name
    shutil.copy2(source, input_path)

    progress(0.05, desc="Preparing audio")

    demucs_out = job_dir / "demucs"
    demucs_out.mkdir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress(0.12, desc=f"Separating stems on {device.upper()}")

    run_cmd([
        "python", "-m", "demucs",
        "-n", "htdemucs_6s",
        "--device", device,
        "--out", str(demucs_out),
        "--float32",
        str(input_path),
    ])

    model_dir = demucs_out / "htdemucs_6s"
    song_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not song_dirs:
        raise RuntimeError("Demucs finished without creating a song directory.")

    separated = song_dirs[0]
    files_for_download = []
    total = len(selected)

    for i, instrument in enumerate(selected):
        progress(
            0.25 + (i / total) * 0.65,
            desc=f"Transcribing {instrument}",
        )

        stem = find_stem(separated, instrument)

        stem_target = job_dir / "stems" / f"{instrument}.wav"
        stem_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stem, stem_target)

        midi = transcribe_stem(stem_target, job_dir, instrument)
        files_for_download.extend([stem_target, midi])

    progress(0.95, desc="Packaging results")

    zip_path = job_dir / "stemforge-results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files_for_download:
            z.write(path, path.relative_to(job_dir))
        z.write(input_path, Path("source") / input_path.name)

    progress(1.0, desc="Complete")

    return (
        str(zip_path),
        [str(p) for p in files_for_download],
        (
            f"Complete. Generated {len(selected)} instrument stem(s) "
            f"and MIDI file(s). Drum MIDI is experimental."
        ),
    )


with gr.Blocks(title="StemForge") as demo:
    gr.Markdown(
        """
        # 🎛️ StemForge
        ### Song → instrument stems → MIDI

        Upload a song, wait until its waveform appears, select the instruments,
        then run separation and transcription.

        **Pipeline:** Demucs `htdemucs_6s` → isolated instrument → Basic Pitch.

        **Drums:** the drum stem is separated correctly, but the current MIDI
        transcription is experimental. A dedicated drum transcription model
        should replace this stage in the next version.
        """
    )

    audio = gr.Audio(
        label="Source song",
        type="filepath",
        sources=["upload"],
    )

    with gr.Row():
        guitar = gr.Checkbox(value=True, label="🎸 Guitar")
        piano = gr.Checkbox(value=True, label="🎹 Piano")
        bass = gr.Checkbox(value=True, label="🎸 Bass")
        drums = gr.Checkbox(value=True, label="🥁 Drums")

    run = gr.Button("🔥 Separate + Transcribe", variant="primary")

    status = gr.Markdown()
    zip_file = gr.File(label="All results (.zip)")
    individual = gr.File(
        label="Individual WAV + MIDI files",
        file_count="multiple",
    )

    run.click(
        fn=process_song,
        inputs=[audio, guitar, piano, bass, drums],
        outputs=[zip_file, individual, status],
        api_name="process_song",
    )

if __name__ == "__main__":
    demo.queue(max_size=4).launch()
