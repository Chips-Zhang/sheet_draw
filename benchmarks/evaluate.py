"""Reproduce the note-detection benchmark from the project root."""
import argparse
import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from sheet_draw.transcriber import MusicTranscriber, TranscriptionConfig


@dataclass
class Clip:
    name: str
    group: str
    audio: np.ndarray
    sr: int
    # Each reference contains MIDI pitch, onset seconds, offset seconds.
    # None timestamps explicitly denote recordings without boundary annotations.
    reference: list


def synthetic(name, group, score, sr):
    pieces, reference = [], []
    position = 0
    for midi, duration in score:
        n = round(sr * duration)
        if midi is None:
            signal = np.zeros(n)
        else:
            time = np.arange(n) / sr
            phase = 2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * time
            envelope = np.minimum(1, np.minimum(time, duration - time) / .012)
            signal = (.6 * np.sin(phase) + .22 * np.sin(2 * phase)
                      + .12 * np.sin(3 * phase)) * envelope
            reference.append((midi, position / sr, (position + n) / sr))
        pieces.append(signal)
        position += n
    return Clip(name, group, np.concatenate(pieces), sr, reference)


def fixtures():
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    for direction, sequence, sr in [('up', scale, 22050), ('down', scale[::-1], 44100)]:
        yield synthetic(direction, 'scale', [(pitch, .31) for pitch in sequence], sr)
    phrase = [(None, .16), (64, .25), (67, .375), (None, .25), (67, .125),
              (None, .125), (62, .5), (72, .25), (None, .25), (69, .375), (None, .25)]
    yield synthetic('rest-and-leaps', 'rests', phrase, 44100)
    # A 120-BPM syncopation and a one-beat triplet; annotate played seconds.
    yield synthetic('offbeat-and-tie', 'syncopation', [
        (None, .25), (60, .75), (64, .25), (None, .25), (67, .75),
        (65, .25), (None, .25), (62, .5)], 44100)
    yield synthetic('eighth-triplets', 'triplets', [
        (60, 1/6), (62, 1/6), (64, 1/6), (67, .5),
        (65, 1/6), (64, 1/6), (62, 1/6), (60, .5)], 44100)
    # Root keys from the source SFZ. Each file contains one original, sustained note.
    for i, pitch in enumerate([64, 67, 72, 76, 79, 84, 88], 1):
        path = Path(__file__).parent / 'recordings' / f'harmonica_{i}.flac'
        audio, sr = sf.read(path, always_2d=True)
        yield Clip(path.stem, 'recordings', audio.mean(axis=1), sr, [(pitch, None, None)])


def predict(clip, engine):
    if engine == 'basic-pitch':
        from sheet_draw.basic_pitch_backend import predict_notes
        notes = predict_notes(clip.audio, clip.sr)
    else:
        config = TranscriptionConfig(sample_rate=clip.sr, engine=engine, enable_quantization=False)
        notes = MusicTranscriber(config).transcribe(clip.audio).notes
    return [(int(round(69 + 12 * np.log2(note.frequency / 440))),
             note.start_time, note.start_time + note.duration) for note in notes]


def count_matches(reference, predicted, check_offset):
    if not predicted:
        return 0
    eligible = np.zeros((len(reference), len(predicted)), dtype=int)
    for i, (pitch, start, end) in enumerate(reference):
        for j, (candidate, onset, offset) in enumerate(predicted):
            if candidate != pitch or (start is not None and abs(onset - start) > .050):
                continue
            if check_offset and abs(offset - end) > max(.050, (end - start) * .20):
                continue
            eligible[i, j] = 1
    rows, cols = linear_sum_assignment(-eligible)
    return int(eligible[rows, cols].sum())


def scores(tp, ref, pred):
    return dict(matches=tp, precision=tp / pred if pred else 0,
                recall=tp / ref, f1=2 * tp / (ref + pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basic-pitch', action='store_true')
    parser.add_argument('--json', type=Path, help='save metrics and note events for each clip')
    args = parser.parse_args()
    clips = list(fixtures())
    report = {}
    for engine in ['legacy', 'monophonic'] + (['basic-pitch'] if args.basic_pitch else []):
        rows, groups = [], {}
        for clip in clips:
            with contextlib.redirect_stdout(io.StringIO()):
                prediction = predict(clip, engine)
            n = len(clip.reference)
            matched = count_matches(clip.reference, prediction, False)
            timed = clip.reference[0][1] is not None
            full = count_matches(clip.reference, prediction, True) if timed else None
            rows.append(dict(clip=clip.name, reference=clip.reference, predicted=prediction))
            total = groups.setdefault(clip.group, [0, 0, 0, 0])
            for i, value in enumerate([n, len(prediction), matched, full or 0]):
                total[i] += value
        summary = {}
        for group, (n, p, tp, timed_tp) in groups.items():
            summary[group] = dict(reference=n, predicted=p, note=scores(tp, n, p))
            if group != 'recordings':
                summary[group]['with_offset'] = scores(timed_tp, n, p)
            print(engine, group, json.dumps(summary[group]))
        report[engine] = dict(summary=summary, clips=rows)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
