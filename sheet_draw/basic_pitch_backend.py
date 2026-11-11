"""Optional reference backend for comparing Spotify Basic Pitch note events."""

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from .audio_utils import frequency_to_note
from .score import Note


HARMONICA_PROFILES = {
    "official": {},
    "short": {"minimum_note_length": 70.},
    "sensitive": {"minimum_note_length": 70., "onset_threshold": .40, "frame_threshold": .20},
    "selective": {"minimum_note_length": 70., "onset_threshold": .55, "frame_threshold": .40},
}


def predict_harmonica_notes(audio, rate, instrument, key, profile):
    from .harmonica import make_harp
    if profile not in HARMONICA_PROFILES:
        raise ValueError(f"未知口琴识别参数：{profile}")
    params = dict(HARMONICA_PROFILES[profile])
    if profile != "official":
        harp = make_harp(instrument, key)
        low, high = harp.get_range()
        params.update(minimum_frequency=low * .97, maximum_frequency=high * 1.03)
    return predict_notes(audio, rate, **params)


@lru_cache(maxsize=1)
def _load_model():
    try:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("请在评测环境安装 basic-pitch[onnx]==0.4.0") from exc
    return Model(build_icassp_2022_model_path(FilenameSuffix.onnx))


def predict_notes(audio, sample_rate, **thresholds):
    """Return raw events, preserving bends' base pitches and overlapping notes."""
    model = _load_model()
    from basic_pitch.inference import predict

    with TemporaryDirectory() as directory:
        path = Path(directory) / "input.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        _, _, events = predict(str(path), model, **thresholds)
    notes = []
    for start, end, midi, amplitude, bends in sorted(events, key=lambda event: event[0]):
        freq = 440 * 2 ** ((int(midi) - 69) / 12)
        notes.append(Note(frequency_to_note(freq), freq, float(start), float(end - start),
                          float(np.clip(amplitude, 0, 1)),
                          tuple(map(int, bends)) if bends is not None else ()))
    return notes


def predict_melody(audio, sample_rate):
    """Choose a single line of strong, sustained events for a playable solo part."""
    return select_melody(predict_notes(audio, sample_rate))


def select_melody(events):
    from bisect import bisect_right
    from dataclasses import replace

    def effective_end(note):
        # Model releases can extend a few frames into the next attack.
        return note.start_time + note.duration - min(.04, note.duration * .2)

    events = sorted(events, key=effective_end)
    ends = [effective_end(n) for n in events]
    scores = [0.]
    previous = []
    keep = []
    for i, note in enumerate(events):
        compatible = bisect_right(ends, note.start_time, 0, i)
        candidate = scores[compatible] + note.duration * note.velocity ** 2
        previous.append(compatible)
        keep.append(candidate > scores[i])
        scores.append(max(candidate, scores[i]))

    melody = []
    i = len(events)
    while i:
        if keep[i - 1]:
            note = events[i - 1]
            if melody:
                note = replace(note, duration=min(note.duration, melody[-1].start_time - note.start_time))
            melody.append(note)
            i = previous[i - 1]
        else:
            i -= 1
    return melody[::-1]


def select_duet(events, instrument, key):
    """Add a compatible adjacent-hole partner to near-simultaneous melody attacks."""
    from bisect import bisect_left, bisect_right
    from .harmonica import make_harp
    from .notation import hole_choices

    harp = make_harp(instrument, key)
    ordered = sorted(events, key=lambda n: n.start_time)
    starts = [n.start_time for n in ordered]
    positions = {n.pitch: hole_choices(harp, n.pitch) for n in ordered}
    melody = select_melody(ordered)
    used = {(n.start_time, n.pitch) for n in melody}
    selected = list(melody)
    for lead in melody:
        partners = []
        lo = bisect_left(starts, lead.start_time - .055)
        hi = bisect_right(starts, lead.start_time + .055)
        for candidate in ordered[lo:hi]:
            if (candidate.start_time, candidate.pitch) in used or candidate.pitch == lead.pitch:
                continue
            overlap = min(lead.start_time + lead.duration, candidate.start_time + candidate.duration)
            overlap -= max(lead.start_time, candidate.start_time)
            if overlap < .6 * max(lead.duration, candidate.duration):
                continue
            if any(a.action == b.action and a.button == b.button and not a.bend and not b.bend
                   and abs(a.hole - b.hole) == 1 for a in positions[lead.pitch]
                   for b in positions[candidate.pitch]):
                partners.append((overlap * candidate.velocity, candidate))
        if partners:
            partner = max(partners, key=lambda pair: pair[0])[1]
            used.add((partner.start_time, partner.pitch))
            selected.append(partner)
    return sorted(selected, key=lambda n: (n.start_time, n.frequency))
