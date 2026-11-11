"""Score frozen, song-disjoint development and holdout performances."""
import argparse
from dataclasses import replace
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import urlopen

import mir_eval
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sheet_draw import MusicTranscriber, TranscriptionConfig
from sheet_draw.basic_pitch_backend import predict_notes, select_duet
from sheet_draw.notation import beat_coordinates, grid_events

REVISION = "25bf104035b3aac560324576103dc0c664e87256"
BASE = ("https://raw.githubusercontent.com/filipemlins/Automatic-Music-Transcription-Harmonica/"
        + REVISION + "/AnnotatedSetupFinalMidi/")
ENGINES = ("legacy", "monophonic", "raw", "basic-pitch", "basic-pitch-poly", "duet-official", "basic-pitch-duet", "duet-score")


def cached_file(cache, relative, blob):
    original = relative.startswith("Annotaded/")
    url_base = BASE.rsplit("AnnotatedSetupFinalMidi/", 1)[0] if original else BASE
    prefix = "original-" if original else ""
    path = cache / (prefix + Path(relative).name)
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(url_base + quote(relative), timeout=90) as response:
            data = response.read()
    digest = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if digest != blob:
        raise ValueError(f"Expected {blob} for {relative}; retrieved {digest}")
    if not path.exists():
        path.write_bytes(data)
    return path


def load_reference(clip, cache):
    path = cached_file(cache, clip['labels'], clip['labels_blob'])
    fmt = clip.get('label_format', 'midi')
    if fmt == 'midi':
        data = np.loadtxt(path, skiprows=1, ndmin=2)
    else:
        data = np.loadtxt(path, delimiter=',', ndmin=2)
        if fmt == 'start_hz_end':
            data = data[:, [0, 2, 1]]
        data[:, 2] = np.rint(69 + 12 * np.log2(data[:, 2] / 440))
    audio = cached_file(cache, clip['audio'], clip['audio_blob'])
    if not (data.shape[1] == 3 and np.all(np.isfinite(data)) and
            np.all(data[:, 0] >= 0) and np.all(data[:, 1] > data[:, 0]) and
            data[:, 1].max() <= sf.info(audio).duration + .05 and
            np.all((data[:, 2] >= 0) & (data[:, 2] <= 127))):
        raise ValueError(f"Invalid time or pitch label in {path}")
    return data


def overlap_mask(reference):
    # At least 30 ms of shared sounding time, ignoring nearly coincident releases.
    common = (np.minimum.outer(reference[:, 1], reference[:, 1])
              - np.maximum.outer(reference[:, 0], reference[:, 0]))
    np.fill_diagonal(common, 0.)
    return np.any(common >= .03, axis=1)


def score(reference, notes):
    intervals = np.array([(n.start_time, n.start_time + n.duration) for n in notes]).reshape(-1, 2)
    pitches = np.array([n.frequency for n in notes])
    ref_hz = 440 * 2 ** ((reference[:, 2] - 69) / 12)
    pairs = mir_eval.transcription.match_notes(reference[:, :2], ref_hz, intervals, pitches, offset_ratio=None)
    strict = mir_eval.transcription.match_notes(reference[:, :2], ref_hz, intervals, pitches)
    poly = overlap_mask(reference)
    # Classify remaining near-onset errors without assigning any event twice.
    matched_ref = {r for r, _ in pairs}
    matched_est = {p for _, p in pairs}
    ref_other = [i for i in range(len(reference)) if i not in matched_ref]
    est_other = [i for i in range(len(notes)) if i not in matched_est]
    midi = 69 + 12 * np.log2(pitches / 440)
    deltas = np.abs(np.subtract.outer(reference[ref_other, 2], midi[est_other]))
    times = np.abs(np.subtract.outer(reference[ref_other, 0], intervals[est_other, 0]))
    cost = np.where(times <= .05, np.minimum(deltas, 127) + times, 1e6)
    i, j = linear_sum_assignment(cost)
    mismatches = deltas[i, j][times[i, j] <= .05]
    # Include pitch differences > 50 cents only (in case of ambiguous simultaneous matches).
    mismatches = mismatches[mismatches > .5]
    return dict(reference=len(reference), predicted=len(notes), onset_matches=len(pairs),
                offset_matches=len(strict), poly_reference=int(poly.sum()),
                poly_matches=sum(int(poly[r]) for r, _ in pairs),
                wrong_pitch=len(mismatches), octave_errors=int((np.abs(mismatches - 12) <= .5).sum()),
                onset_abs_ms=sum(1000 * abs(reference[r, 0] - intervals[p, 0]) for r, p in pairs),
                offset_abs_ms=sum(1000 * abs(reference[r, 1] - intervals[p, 1]) for r, p in pairs))


def metrics(counts):
    n, p, hit = counts['reference'], counts['predicted'], counts['onset_matches']
    return dict(**counts, missed=n-hit, extra=p-hit, precision=hit/p if p else 0.,
                recall=hit/n, onset_f1=2*hit/(n+p), offset_f1=2*counts['offset_matches']/(n+p),
                overlap_recall=counts['poly_matches']/counts['poly_reference'] if counts['poly_reference'] else None,
                onset_mae_ms=counts['onset_abs_ms']/hit if hit else None,
                offset_mae_ms=counts['offset_abs_ms']/hit if hit else None)


def quantized_score_notes(result):
    if not result.notes:
        return []
    starts, stops, _, _, _ = grid_events(result.notes, result.bpm, result.beats, 'auto', None)
    ticks = np.zeros((len(result.notes), 2))
    for tick, indices in starts.items():
        ticks[indices, 0] = tick
    for index, tick in stops.items():
        ticks[index, 1] = tick
    # Invert the same monotone map used by the staff renderer. Ties are a single event.
    target = ticks.ravel() / 12
    upper = max(n.start_time + n.duration for n in result.notes) + 120 / result.bpm
    while beat_coordinates([upper], result.bpm, result.beats)[0] < target.max():
        upper *= 2
    lo, hi = np.zeros_like(target), np.full_like(target, upper)
    for _ in range(40):
        mid = (lo + hi) / 2
        before = beat_coordinates(mid, result.bpm, result.beats) < target
        lo, hi = np.where(before, mid, lo), np.where(before, hi, mid)
    times = ((lo + hi) / 2).reshape(-1, 2)
    return [replace(n, start_time=float(a), duration=float(b-a))
            for n, (a, b) in zip(result.notes, times)]


def evaluate(row, engine, audio, rate):
    with contextlib.redirect_stdout(io.StringIO()):
        if engine == 'raw':
            return predict_notes(audio, rate)
        key = row['audio'].split('/')[0].rsplit('_', 1)[1]
        if engine == 'duet-official':
            return select_duet(predict_notes(audio, rate), 'diatonic', key)
        backend = 'basic-pitch-duet' if engine == 'duet-score' else engine
        config = TranscriptionConfig(sample_rate=rate, engine=backend, enable_quantization=False,
                                     harmonica_type='diatonic', harmonica_key=key)
        result = MusicTranscriber(config).transcribe(audio)
        return quantized_score_notes(result) if engine == 'duet-score' else result.notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--split", choices=['dev', 'holdout', 'all'], default='all')
    parser.add_argument("--engines", nargs='+', choices=ENGINES, default=ENGINES)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'benchmarks/performances_manifest.json').read_text())
    assert not ({r['song'] for r in manifest if r['split'] == 'dev'}
                & {r['song'] for r in manifest if r['split'] == 'holdout'})
    rows, totals = [], {}
    for clip in manifest:
        if args.split != 'all' and clip['split'] != args.split:
            continue
        signal, rate = sf.read(cached_file(args.cache, clip['audio'], clip['audio_blob']), always_2d=True)
        ref = load_reference(clip, args.cache)
        for engine in args.engines:
            notes = evaluate(clip, engine, signal.mean(axis=1), rate)
            counts = score(ref, notes)
            rows.append(dict(clip=clip['audio'], song=clip['song'], split=clip['split'], engine=engine, **metrics(counts)))
            total = totals.setdefault((clip['split'], engine), {})
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
        print(clip['split'], clip['audio'], flush=True)
    summary = [dict(split=k[0], engine=k[1], **metrics(v)) for k, v in totals.items()]
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(dict(revision=REVISION, summary=summary, clips=rows), indent=2) + '\n')


if __name__ == "__main__":
    main()
