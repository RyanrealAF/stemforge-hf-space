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

INSTRUMENTS = ("guitar", "piano", "bass", "drums")


def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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


def resolve_audio_file(audio_file):
    """Gradio Audio(type='filepath') normally returns a string path."""
    if audio_file is None:
        return None
    if isinstance(audio_file, (str, os.PathLike)):
        path = Path(audio_file)
        return path if path.is_file() else None
    # Defensive handling for a few Gradio/client representations.
    if isinstance(audio_file, dict):
        for key in ("path", "name"):
            value = audio_file.get(key)
            if value and Path(value).is_file():
                return Path(value)
    return None


def find_stem(root: Path, name: str) -> Path:
    matches = list(root.rglob(f"{name}.wav"))
    if not matches:
        raise FileNotFoundError(
            f"Demucs did not produce {name}.wav. "
            f"Available WAV files: {[p.name for p in root.rglob('*.wav')]}"
        )
    return matches[0]


def transcribe_stem(stem: Path, out_dir: Path, instrument: str):
    """
    Basic Pitch v0.4.x predict_and_save signature requires:
      audio_paths, output_directory, save_midi, sonify_midi,
      save_model_outputs, save_notes, model_path
    """
    instrument_dir = out_dir / "midi" / instrument
    instrument_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        [str(stem)],
        str(instrument_dir),
        True,   # save_midi
        False,  # sonify_midi
        False,  # save_model_outputs
        False,  # save_notes
        ICASSP_2022_MODEL_PATH,
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
    source = resolve_audio_file(audio_file)
    if source is None:
        raise gr.Error(
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
