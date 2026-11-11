"""Move a sung melodic line into one harmonica register without changing its intervals."""
from dataclasses import replace
from functools import lru_cache
from math import log2

from .audio_utils import frequency_to_note
from .harmonica import make_harp
from .notation import hole_choices


TONICS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


def arrange(notes, instrument, key, semitones=None, target_tonic=None, source_tonic=None):
    harp = make_harp(instrument, key)

    @lru_cache(None)
    def difficulty(midi):
        options = hole_choices(harp, frequency_to_note(440 * 2 ** ((midi-69)/12)))
        if not options:
            return 1, 0.
        # Natural notes, little bending, and a central playing position are easiest.
        cost = min({None: 0., "half": 1., "full": 2., "three_half": 3.}[p.bend]
                   + .15*p.button + .025*abs(p.hole-5) for p in options)
        return 0, cost

    pitches = [round(69 + 12*log2(n.frequency/440)) for n in notes]
    def cost(shift):
        unavailable = effort = 0.
        for midi, note in zip(pitches, notes):
            miss, hard = difficulty(midi+shift)
            weight = max(.08, min(1., note.duration))
            unavailable += miss * weight
            effort += hard * weight
        # On chromatic harps prefer the singer's key among fully playable options.
        return (round(unavailable, 5), bool(instrument == "chromatic" and shift % 12),
                effort/max(len(notes), 1) + .008 * abs(shift), abs(shift))
    if semitones is None:
        choices = range(-24, 37)
        if target_tonic is not None:
            delta = TONICS.index(target_tonic) - source_tonic
            choices = [s for s in choices if (s - delta) % 12 == 0]
        semitones = min(choices, key=cost) if notes else 0
    if not -36 <= semitones <= 36:
        raise ValueError("整体移调范围为 −36 到 +36 半音。")
    result = []
    for note in notes:
        freq = note.frequency * 2 ** (semitones / 12)
        result.append(replace(note, pitch=frequency_to_note(freq), frequency=freq))
    return result, semitones, sum(difficulty(p + semitones)[0] for p in pitches)


def estimate_reference_tonic(audio, rate, notes):
    """Estimate the pitch class represented by 1 in relative numbered notation."""
    import librosa
    import numpy as np
    from .key_detection import KeyDetector

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # Take dispersed windows on long tracks to avoid basing the key on the intro.
    windows = [audio]
    if len(audio) > 90 * rate:
        windows = [audio[i:i+30*rate] for i in np.linspace(0,len(audio)-30*rate,3,dtype=int)]
    profiles = []
    for window in windows:
        wave = librosa.resample(window, orig_sr=rate, target_sr=22050)
        if len(wave) < 22050:
            wave = np.pad(wave, (0, 22050-len(wave)))
        profiles.append(librosa.feature.chroma_cqt(y=wave,sr=22050).mean(axis=1))
    chroma = np.mean(profiles,axis=0)
    vocal = np.bincount([round(69+12*log2(n.frequency/440))%12 for n in notes],
                        weights=[min(n.duration,1.) for n in notes],minlength=12)
    distribution = chroma/max(chroma.sum(),1e-9) + .25*vocal/max(vocal.sum(),1e-9)
    detector = KeyDetector()
    detected = detector._krumhansl_schmuckler(distribution)
    tonic = detector.NOTE_NAMES.index(detected["tonic"])
    return (tonic + (3 if detected["mode"] == "minor" else 0)) % 12


def fit_missing_octaves(notes, instrument, key):
    """For unavailable notes only, try the nearest playable register of the same pitch class."""
    harp = make_harp(instrument, key)
    fitted = []
    moved = missing = 0
    for i,n in enumerate(notes):
        if hole_choices(harp,n.pitch):
            fitted.append(n)
            continue
        midi = round(69 + 12 * log2(n.frequency/440))
        neighbours = [round(69+12*log2(m.frequency/440)) for m in
                      (fitted[-1:] + notes[i+1:i+2])]
        candidates=[]
        for delta in (-36,-24,-12,12,24,36):
            freq=n.frequency * 2 ** (delta/12)
            name=frequency_to_note(freq)
            if hole_choices(harp,name):
                candidates.append((abs(delta),sum(abs(midi+delta-k) for k in neighbours),
                                   replace(n,frequency=freq,pitch=name)))
        if candidates:
            fitted.append(min(candidates,key=lambda c:c[:2])[2])
            moved+=1
        else:
            fitted.append(n)
            missing+=1
    return fitted,moved,missing


def vocal_pulse(bpm, beats):
    """Report the slower quarter-note pulse when the tracker follows fast subdivisions."""
    if bpm >= 150:
        return bpm / 2, beats[::2]
    return bpm, beats
