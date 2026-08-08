import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import torch

# Optional on standard GPU Spaces. On ZeroGPU, this decorator requests a GPU
# only while the heavy function is running.
try:
    import spaces

    GPU = spaces.GPU
except Exception:
    def GPU(fn):
        return fn

from basic_pitch.inference import predict_and_save

APP_ROOT = Path(__file__).parent
OUTPUT_ROOT = APP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

INSTRUMENTS = ["guitar", "piano", "bass", "drums"]


def run_cmd(cmd):
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-12000:])
    return proc.stdout


def find_stem(root: Path, name: str) -> Path:
    matches = list(root.rglob(f"{name}.wav"))
    if not matches:
        raise FileNotFoundError(f"Demucs did not produce {name}.wav")
    return matches[0]


def transcribe_stem(stem: Path, out_dir: Path, instrument: str):
    # Basic Pitch is instrument-agnostic and works best on one instrument at a time.
    instrument_dir = out_dir / "midi" / instrument
    instrument_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        [str(stem)],
        str(instrument_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
    )

    mids = list(instrument_dir.glob("*.mid"))
    if not mids:
        raise RuntimeError(f"Basic Pitch did not produce MIDI for {instrument}")

    target = instrument_dir / f"{instrument}.mid"
    if mids[0] != target:
        shutil.copy2(mids[0], target)
    return target


@GPU
def process_song(audio_file, guitar, piano, bass, drums, progress=gr.Progress()):
    if not audio_file:
        raise gr.Error("Upload an audio file first.")

    selected = [
        name for name, enabled in [
            ("guitar", guitar),
            ("piano", piano),
            ("bass", bass),
            ("drums", drums),
        ] if enabled
    ]
    if not selected:
        raise gr.Error("Select at least one instrument.")

    job_id = next(tempfile._get_candidate_names())
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    source = Path(audio_file)
    input_path = job_dir / source.name
    shutil.copy2(source, input_path)

    progress(0.05, desc="Preparing audio")

    # htdemucs_6s is the Demucs model that exposes guitar and piano in
    # addition to drums, bass, vocals, and other.
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

        # Copy stem into a predictable results directory.
        stem_target = job_dir / "stems" / f"{instrument}.wav"
        stem_target.parent.mkdir(exist_ok=True)
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
        f"Complete. Generated {len(selected)} instrument stem(s) and MIDI file(s). "
        "Drum MIDI is experimental because the current transcription stage is pitch-oriented.",
    )


with gr.Blocks(title="StemForge") as demo:
    gr.Markdown(
        """
        # 🎛️ StemForge
        ### Song → instrument stems → MIDI

        Upload a song. StemForge uses **Demucs htdemucs_6s** to isolate
        guitar, piano, bass, and drums, then sends each isolated stem to
        **Spotify Basic Pitch** for MIDI transcription.

        **Note:** drum separation is supported, but drum MIDI is experimental.
        """
    )

    audio = gr.Audio(
        label="Source song",
        type="filepath",
        sources=["upload"],
    )

    with gr.Row():
        guitar = gr.Checkbox(True, label="🎸 Guitar")
        piano = gr.Checkbox(True, label="🎹 Piano")
        bass = gr.Checkbox(True, label="🎸 Bass")
        drums = gr.Checkbox(True, label="🥁 Drums")

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
