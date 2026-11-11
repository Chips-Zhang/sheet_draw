"""Compare separation on a known stereo mix of real reeds and synthetic backing."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.evaluate import count_matches, scores
from sheet_draw.transcriber import MusicTranscriber, TranscriptionConfig
from sheet_draw.vocal_separation import preprocess_for_transcription


def mixture(target_db=None):
    """Use sustained sections of four CC0 samples as an equal-L/R melody."""
    directory = Path(__file__).parent / "recordings"
    melody, reference = [], []
    for index, pitch in [(1, 64), (3, 72), (2, 67), (4, 76)]:
        sample, rate = sf.read(directory / f"harmonica_{index}.flac")
        n = int(.36 * rate)
        reed = sample[int(.15 * rate):int(.15 * rate) + n].copy()
        envelope = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (.012 * rate), 0, 1)
        reed *= envelope * (.28 / np.max(np.abs(reed)))
        start = len(melody) / rate
        reference.append((pitch, start, start + n / rate))
        melody.extend(reed)
        melody.extend(np.zeros(int(.14 * rate)))
    lead = np.array(melody)
    time = np.arange(len(lead)) / rate
    phase = 2 * np.pi * 130.81 * time
    bass = (.42 * np.sin(phase) + .16 * np.sin(2 * phase)) * (.8 + .2 * np.cos(4 * np.pi * time))
    right = .3 * np.sin(2 * np.pi * 196 * time)
    backing = np.column_stack((bass, right))
    if target_db is not None:
        backing *= np.sqrt(np.mean(lead**2) / np.mean(backing**2)) * 10 ** (-target_db / 20)
    return lead[:, None] + backing, lead, reference, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demucs", action="store_true")
    parser.add_argument("--levels", nargs="+", type=float, default=[None],
                        help="target/backing RMS ratios in dB")
    args = parser.parse_args()
    for level in args.levels:
        evaluate(level, args.demucs)


def evaluate(level, demucs):
    stereo, lead, reference, rate = mixture(level)
    methods = ["dry", "none", "center", "hpss", "spectral", "repet"]
    if demucs:
        methods.extend(["demucs-vocals", "demucs-other"])
    for mode in methods:
        method, _, target = mode.partition("-")
        if method == "dry":
            audio = lead
        elif method == "none":
            audio = stereo.mean(axis=1)
        else:
            audio = preprocess_for_transcription(stereo, rate, method, target or "vocals")
        with contextlib.redirect_stdout(io.StringIO()):
            result = MusicTranscriber(TranscriptionConfig(
                sample_rate=rate, enable_quantization=False)).transcribe(audio)
        predicted = [(round(69 + 12 * np.log2(n.frequency / 440)), n.start_time,
                      n.start_time + n.duration) for n in result.notes]
        correct = count_matches(reference, predicted, False)
        # Fixed gain SNR penalizes both residual backing and missing lead.
        snr = 10 * np.log10(np.sum(lead**2) / max(1e-16, np.sum((audio - lead)**2)))
        print(json.dumps(dict(method=mode, target_db=level, snr_db=None if mode == "dry" else round(float(snr), 2), reference=len(reference),
                              predicted=len(predicted), **scores(correct, len(reference), len(predicted)))))


if __name__ == "__main__":
    main()
