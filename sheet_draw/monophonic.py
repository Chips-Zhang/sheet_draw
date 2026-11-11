"""Track the pitch and voiced boundaries of a single melodic line."""

from math import gcd
from typing import List

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, resample_poly, sosfilt

from .audio_utils import frequency_to_note
from .score import Note


def _fundamental(frame, rate):
    """First strong periodic peak of McLeod's normalized square difference."""
    x = frame - frame.mean()
    power = np.r_[0., np.cumsum(x * x)]
    if power[-1] < 1e-16:
        return 0.
    size = next_fast_len(2 * len(x) - 1)
    spectrum = rfft(x, size)
    max_lag = min(len(x) // 2, int(np.ceil(rate / 50)))
    lags = np.arange(max_lag + 1)
    corr = irfft(spectrum * spectrum.conj(), size)[:max_lag + 1]
    denom = power[len(x) - lags] + power[-1] - power[lags]
    periodicity = 2 * corr / np.maximum(denom, 1e-16)
    peaks = find_peaks(periodicity)[0]
    peaks = peaks[peaks >= max(2, int(rate / 2400))]
    if not peaks.size:
        return 0.
    strong = peaks[periodicity[peaks] >= max(.75, .91 * periodicity[peaks].max())]
    if not strong.size:
        return 0.
    lag = strong[0]
    a, b, c = periodicity[lag - 1:lag + 2]
    shift = .5 * (a - c) / (a - 2 * b + c) if a + c != 2 * b else 0.
    return rate / (lag + shift)


def track_notes(audio, sample_rate, min_duration=.05, smooth=True) -> List[Note]:
    """Measure one pitch per 10 ms and emit contiguous voiced note events."""
    signal = np.asarray(audio, dtype=float)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or sample_rate <= 0 or not np.all(np.isfinite(signal)):
        raise ValueError("需要有限的音频采样和有效采样率")
    if len(signal) < sample_rate * min_duration or not np.any(signal):
        return []
    duration = len(signal) / sample_rate
    rate = min(sample_rate, 22050)
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        signal = resample_poly(signal, rate // divisor, sample_rate // divisor)
    # Very high overtones can dominate integer-lag correlation in bright reeds.
    # Retain the supported F0 range (up to 2.4 kHz) and its second harmonic.
    if rate > 9600:
        signal = sosfilt(butter(4, 4800, fs=rate, output="sos"), signal)
    hop = max(1, round(.010 * rate))
    half = max(2, round(.025 * rate))
    centers = np.arange(hop // 2, len(signal) + hop // 2, hop)
    padded = np.pad(signal, (half, half + hop))
    level = np.array([np.sqrt(np.mean(padded[c + half - hop // 2:
                                              c + half - hop // 2 + hop] ** 2))
                      for c in centers])
    gate = max(1e-8, level.max() * .012)
    freq = np.zeros(len(centers))
    for i in np.flatnonzero(level > gate):
        c = centers[i]
        freq[i] = _fundamental(padded[c:c + 2 * half], rate)
    labels = np.full(len(freq), -1, dtype=int)
    voiced = freq > 0
    labels[voiced] = np.rint(69 + 12 * np.log2(freq[voiced] / 440.)).astype(int)
    if smooth:
        labels = median_filter(labels, 3, mode='nearest')
        # Discard brief pitch excursions inside a sustained note (e.g. vibrato).
        # Keep unvoiced gaps, which can be the only marker of a repeated note.
        edges = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
        for first, last in zip(edges[1:-2], edges[2:-1]):
            if (last - first <= 3 and labels[first] >= 0
                    and labels[first - 1] == labels[last]):
                labels[first:last] = labels[last]
    # A strong dip with matching pitches on both sides signals re-articulation.
    db = 20 * np.log10(np.maximum(level, gate))
    dips, _ = find_peaks(-db, distance=5)
    splits = {i for i in dips if 2 <= i < len(labels) - 2
              and labels[i - 2] == labels[i] == labels[i + 2] and labels[i] >= 0
              and min(db[max(0, i - 6):i].max(), db[i + 1:i + 7].max()) - db[i] > 12}
    changes = np.flatnonzero(np.diff(labels)) + 1
    boundaries = sorted({0, len(labels), *changes, *splits})
    notes = []
    for first, last in zip(boundaries, boundaries[1:]):
        start = first * hop / rate
        end = min(duration, last * hop / rate)
        if labels[first] < 0 or end - start + 1e-9 < min_duration:
            continue
        segment = freq[first:last]
        voiced_segment = segment[segment > 0]
        if not voiced_segment.size:
            continue
        pitch = float(np.median(voiced_segment))
        notes.append(Note(pitch=frequency_to_note(pitch), frequency=pitch,
                          start_time=start, duration=end - start, velocity=1.))
    return notes
