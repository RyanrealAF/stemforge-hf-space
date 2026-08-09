import os
import json
import shutil
import subprocess
import zipfile
import tempfile
import math
from pathlib import Path

try:
    import spaces
    GPU = spaces.GPU
except ImportError:
    def GPU(fn):
        return fn


INSTRUMENTS = [
    "guitar",
    "piano",
    "bass",
    "drums",
]

APP_ROOT = Path(__file__).parent
OUTPUT_ROOT = APP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)


def run_command(command):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    if process.returncode != 0:
        raise RuntimeError(
            process.stdout[-15000:]
        )

    return process.stdout


def find_stem(separated_dir, instrument):
    stem_name = f"{instrument}.wav"

    matches = list(
        separated_dir.rglob(stem_name)
    )

    if not matches:
        available = [
            str(path.name)
            for path in separated_dir.rglob("*.wav")
        ]

        raise RuntimeError(
            f"Missing {stem_name}. "
            f"Available stems: {available}"
        )

    return matches[0]


def db(value):
    value = max(float(value), 1e-12)
    return 20.0 * math.log10(value)


def analyze_audio(audio_path):
    import librosa
    import numpy as np

    y, sr = librosa.load(
        str(audio_path),
        sr=None,
        mono=True,
    )

    if len(y) == 0:
        raise RuntimeError(
            "Audio contains no readable samples."
        )

    duration = len(y) / sr

    rms = np.sqrt(
        np.mean(
            np.square(y)
        )
    )

    peak = np.max(
        np.abs(y)
    )

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
    )

    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sr,
    )

    tempo = float(
        np.asarray(tempo).flatten()[0]
    )

    harmonic, percussive = librosa.effects.hpss(y)

    harmonic_energy = np.mean(
        np.square(harmonic)
    )

    percussive_energy = np.mean(
        np.square(percussive)
    )

    total_energy = (
        harmonic_energy
        + percussive_energy
        + 1e-12
    )

    harmonic_ratio = (
        harmonic_energy /
        total_energy
    )

    percussive_ratio = (
        percussive_energy /
        total_energy
    )

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
    )

    flatness = librosa.feature.spectral_flatness(
        y=y
    )

    return {
        "duration_seconds": round(
            duration,
            3,
        ),

        "sample_rate": int(sr),

        "peak_dbfs": round(
            db(peak),
            2,
        ),

        "rms_dbfs": round(
            db(rms),
            2,
        ),

        "tempo_bpm": round(
            tempo,
            2,
        ),

        "beat_count": int(
            len(beats)
        ),

        "harmonic_ratio": round(
            float(harmonic_ratio),
            4,
        ),

        "percussive_ratio": round(
            float(percussive_ratio),
            4,
        ),

        "spectral_centroid": round(
            float(np.mean(centroid)),
            2,
        ),

        "spectral_bandwidth": round(
            float(np.mean(bandwidth)),
            2,
        ),

        "spectral_flatness": round(
            float(np.mean(flatness)),
            5,
        ),
    }


def build_processing_plan(
    analysis,
    selected_instruments,
):

    percussive = analysis[
        "percussive_ratio"
    ]

    harmonic = analysis[
        "harmonic_ratio"
    ]

    bpm = analysis[
        "tempo_bpm"
    ]

    dynamic_range = (
        analysis["peak_dbfs"]
        - analysis["rms_dbfs"]
    )

    if percussive > 0.65:
        overlap = 0.40
    elif harmonic > 0.75:
        overlap = 0.30
    else:
        overlap = 0.35

    if dynamic_range < 8:
        shifts = 2
    else:
        shifts = 1

    plan = {
        "demucs": {
            "model": "htdemucs_6s",
            "overlap": overlap,
            "shifts": shifts,
        },

        "midi": {},
    }

    for instrument in selected_instruments:

        if instrument == "guitar":

            plan["midi"]["guitar"] = {
                "minimum_note_ms": (
                    35 if bpm > 110 else 45
                ),
                "minimum_frequency": 70,
                "maximum_frequency": 1800,
                "polyphonic": True,
            }

        elif instrument == "piano":

            plan["midi"]["piano"] = {
                "minimum_note_ms": (
                    30 if bpm > 110 else 40
                ),
                "minimum_frequency": 27,
                "maximum_frequency": 5000,
                "polyphonic": True,
            }

        elif instrument == "bass":

            plan["midi"]["bass"] = {
                "minimum_note_ms": (
                    45 if bpm > 110 else 65
                ),
                "minimum_frequency": 28,
                "maximum_frequency": 600,
                "polyphonic": False,
            }

        elif instrument == "drums":

            plan["midi"]["drums"] = {
                "onset_threshold": (
                    0.35
                    + percussive * 0.15
                ),
                "minimum_hit_spacing_ms": (
                    25 if bpm > 120 else 35
                ),
            }

    return plan


def separate_audio(
    input_path,
    output_dir,
    plan,
):
    import torch

    demucs_dir = (
        output_dir /
        "demucs"
    )

    demucs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    settings = plan[
        "demucs"
    ]

    command = [
        "python",
        "-m",
        "demucs",

        "-n",
        settings["model"],

        "--device",
        device,

        "--out",
        str(demucs_dir),

        "--float32",

        "--overlap",
        str(settings["overlap"]),

        "--shifts",
        str(settings["shifts"]),

        str(input_path),
    ]

    run_command(command)

    model_dir = (
        demucs_dir /
        settings["model"]
    )

    if not model_dir.exists():
        raise RuntimeError(
            "Demucs completed without "
            "creating its output directory."
        )

    song_dirs = [
        directory
        for directory in model_dir.iterdir()
        if directory.is_dir()
    ]

    if not song_dirs:
        raise RuntimeError(
            "Demucs produced no song directory."
        )

    return song_dirs[0]


def transcribe_stem(
    stem_path,
    midi_dir,
    instrument,
):
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    temp_midi_dir = midi_dir / f"temp_{instrument}"
    temp_midi_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        predict_and_save(
            [str(stem_path)],
            str(temp_midi_dir),
            True,
            False,
            False,
            False,
            ICASSP_2022_MODEL_PATH,
        )

        midi_files = list(
            temp_midi_dir.glob("*.mid")
        )

        if not midi_files:
            raise RuntimeError(
                f"No MIDI generated for "
                f"{instrument}."
            )

        generated = midi_files[0]

        target = (
            midi_dir /
            f"{instrument}.mid"
        )

        shutil.copy2(
            generated,
            target,
        )
        return target
    finally:
        shutil.rmtree(temp_midi_dir, ignore_errors=True)


def create_report(
    analysis,
    plan,
):

    lines = [
        "# StemForge Processing Report",
        "",
        "## Audio Analysis",
        "",
        f"- Duration: "
        f"{analysis['duration_seconds']} seconds",

        f"- Sample rate: "
        f"{analysis['sample_rate']} Hz",

        f"- Peak: "
        f"{analysis['peak_dbfs']} dBFS",

        f"- RMS: "
        f"{analysis['rms_dbfs']} dBFS",

        f"- Estimated BPM: "
        f"{analysis['tempo_bpm']}",

        f"- Harmonic ratio: "
        f"{analysis['harmonic_ratio']}",

        f"- Percussive ratio: "
        f"{analysis['percussive_ratio']}",

        "",
        "## Adaptive Separation",
        "",

        f"- Model: "
        f"{plan['demucs']['model']}",

        f"- Overlap: "
        f"{plan['demucs']['overlap']}",

        f"- Shifts: "
        f"{plan['demucs']['shifts']}",
    ]

    return "\n".join(lines)


@GPU
def process_song_worker(
    audio_path,
    selected_instruments,
    progress=None,
):

    if progress is None:
        def progress(
            value,
            desc="",
        ):
            pass

    input_path = Path(
        audio_path
    )

    if not input_path.exists():
        raise RuntimeError(
            "Uploaded audio file "
            "does not exist."
        )

    selected = [
        instrument
        for instrument in INSTRUMENTS
        if instrument in selected_instruments
    ]

    if not selected:
        raise RuntimeError(
            "No instruments selected."
        )

    job_id = (
        next(
            tempfile._get_candidate_names()
        )
    )

    job_dir = OUTPUT_ROOT / job_id

    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        progress(
            0.02,
            desc="Analyzing audio..."
        )

        local_input = (
            job_dir /
            input_path.name
        )

        shutil.copy2(
            input_path,
            local_input,
        )

        analysis = analyze_audio(
            local_input
        )

        progress(
            0.10,
            desc="Building adaptive plan..."
        )

        plan = build_processing_plan(
            analysis,
            selected,
        )

        with open(
            job_dir /
            "analysis.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                analysis,
                file,
                indent=2,
            )

        with open(
            job_dir /
            "processing-plan.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                plan,
                file,
                indent=2,
            )

        progress(
            0.15,
            desc="Separating instruments..."
        )

        separated_dir = (
            separate_audio(
                local_input,
                job_dir,
                plan,
            )
        )

        stems_dir = (
            job_dir /
            "stems"
        )

        midi_dir = (
            job_dir /
            "midi"
        )

        stems_dir.mkdir(
            exist_ok=True
        )

        midi_dir.mkdir(
            exist_ok=True
        )

        generated_files = []

        total = len(selected)

        for index, instrument in enumerate(
            selected
        ):

            progress(
                0.25
                + (
                    index /
                    total
                ) * 0.60,

                desc=(
                    f"Processing "
                    f"{instrument}..."
                ),
            )

            source_stem = find_stem(
                separated_dir,
                instrument,
            )

            final_stem = (
                stems_dir /
                f"{instrument}.wav"
            )

            shutil.copy2(
                source_stem,
                final_stem,
            )

            generated_files.append(
                final_stem
            )

            if instrument != "drums":

                midi_file = (
                    transcribe_stem(
                        final_stem,
                        midi_dir,
                        instrument,
                    )
                )

                generated_files.append(
                    midi_file
                )

        progress(
            0.90,
            desc="Creating report..."
        )

        report = create_report(
            analysis,
            plan,
        )

        report_file = (
            job_dir /
            "processing-report.md"
        )

        report_file.write_text(
            report,
            encoding="utf-8",
        )

        progress(
            0.95,
            desc="Creating ZIP..."
        )

        zip_path = (
            job_dir /
            "StemForge-results.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:

            for file_path in (
                generated_files
            ):

                archive.write(
                    file_path,
                    file_path.relative_to(
                        job_dir
                    ),
                )

            archive.write(
                report_file,
                "processing-report.md",
            )

            archive.write(
                job_dir /
                "analysis.json",
                "analysis.json",
            )

            archive.write(
                job_dir /
                "processing-plan.json",
                "processing-plan.json",
            )

        progress(
            1.0,
            desc="Complete."
        )

        return str(zip_path)

    except Exception:

        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise
