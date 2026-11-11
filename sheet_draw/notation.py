"""Convert performed note times to ABC staff notation with harmonica fingerings."""
from collections import defaultdict
from itertools import product
from math import ceil, floor, log2
import re
import unicodedata

import numpy as np

from .harmonica import ChromaticHarmonica, make_harp, HarmonicaNote


def beat_coordinates(times, bpm, beats):
    """Map seconds to beats, interpolating local tempo and extrapolating missing beats."""
    times = np.asarray(times)
    anchors = np.asarray(beats, dtype=float)
    period = 60 / bpm
    if len(anchors) < 3 or not np.all(np.isfinite(anchors)):
        return times / period
    gaps = np.diff(anchors)
    if np.any(gaps <= 0):
        return times / period
    counts = np.maximum(1, np.rint(gaps / period))
    local = gaps / counts
    if np.any((local < .65 * period) | (local > 1.5 * period)):
        return times / period
    positions = np.r_[round(anchors[0] / period), round(anchors[0] / period) + np.cumsum(counts)]
    values = np.interp(times, anchors, positions)
    values += np.minimum(0, times - anchors[0]) / local[0]
    values += np.maximum(0, times - anchors[-1]) / local[-1]
    return np.maximum(0, values)


def grid_events(notes, bpm, beats, subdivision, downbeat):
    """Use twelve ticks per beat so binary and ternary divisions share integer time."""
    times = [(n.start_time, n.start_time + n.duration) for n in notes]
    coordinates = beat_coordinates(np.asarray(times).ravel(), bpm, beats)
    # Align the beat lattice to the supplied downbeat, including fractional pickups.
    position = beat_coordinates([downbeat], bpm, beats)[0] if downbeat is not None else 0.
    origin = ceil(position) - position
    coordinates += origin
    within = defaultdict(list)
    for t in coordinates:
        within[floor(t)].append(t % 1)
    ternary = set()
    for beat, offsets in within.items():
        x = np.asarray(offsets)
        if subdivision == "triplet" or (subdivision == "auto" and
                np.sum(abs(x - np.rint(3*x)/3)) + .03 < np.sum(abs(x - np.rint(4*x)/4))):
            ternary.add(beat)

    def tick(x):
        div = 3 if floor(x) in ternary else 4
        return 12 * floor(x) + round((x % 1) * div) * (12 // div)

    starts, stops = defaultdict(list), {}
    for i, (a, b) in enumerate(coordinates.reshape(-1, 2)):
        left = tick(a)
        right = max(left + (4 if left // 12 in ternary else 3), tick(b))
        starts[left].append(i)
        stops[i] = right
    return starts, stops, ternary, tick(origin), ceil(position) * 12


def hole_choices(harp, pitch):
    choices = []
    for hole, row in harp.layout.items():
        options = [(row[0], "blow", False, None), (row[1], "draw", False, None)]
        if isinstance(harp, ChromaticHarmonica):
            options += [(row[2], "blow", True, None), (row[3], "draw", True, None)]
        else:
            action = "draw" if hole <= 6 else "blow"
            options += [(n, action, False, bend) for n, bend in
                        zip(row[2], ("half", "full", "three_half"))]
        choices.extend(HarmonicaNote(None, hole, action, bend, button)
                       for n, action, button, bend in options if n == pitch)
    return choices


def tab_label(choices):
    if any(not options for options in choices):
        return "×"
    chosen = [opts[0] for opts in choices]
    uncertain = len(chosen) > 2
    if len(chosen) == 2:
        pairs = [(a, b) for a, b in product(*choices)
                 if a.hole != b.hole and a.action == b.action and a.button == b.button]
        if pairs:
            chosen = min(pairs, key=lambda pair: (sum(bool(p.bend) for p in pair),
                                                   abs(pair[0].hole - pair[1].hole)))
            uncertain = abs(chosen[0].hole - chosen[1].hole) != 1 or any(p.bend for p in chosen)
        else:
            uncertain = True
    labels = []
    for pos in chosen:
        hole = str(pos.hole) if pos.action == "blow" else chr(0x245F + pos.hole)
        bend = {None: "", "half": "′", "full": "″", "three_half": "‴"}[pos.bend]
        labels.append(f"{hole}{bend}" + ("↑" if pos.button else ""))
    return "+".join(labels) + ("?" if uncertain else "")


PITCH_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
NATURALS = (0, 2, 4, 5, 7, 9, 11)
LETTERS = "CDEFGAB"


def tonic_pc(name):
    return (NATURALS[LETTERS.index(name[0])] + name.count("#") - name.count("b")) % 12


def relative_major(key):
    parts = (key or "C major").split()
    return PITCH_NAMES[(tonic_pc(parts[0]) + (3 if "minor" in parts else 0)) % 12]


def spell_notes(notes, tonic):
    """Return ABC pitches, key-signature accidentals, and movable-do syllables."""
    root = tonic_pc(tonic)
    first = LETTERS.index(tonic[0])
    signature = {}
    for degree, interval in enumerate(NATURALS):
        idx = (first + degree) % 7
        signature[LETTERS[idx]] = (root + interval - NATURALS[idx] + 6) % 12 - 6
    pitches = []
    for note in notes:
        midi = round(69 + 12 * log2(note.frequency / 440))
        octave, pc = divmod(midi - (60 + root), 12)
        degree = max(i for i, semitone in enumerate(NATURALS) if semitone <= pc)
        alteration = pc - NATURALS[degree]
        if alteration and (tonic.endswith("b") or tonic == "F"):
            degree += 1
            alteration = -1
        idx = (first + degree) % 7
        letter = LETTERS[idx]
        accidental = signature[letter] + alteration
        register = (midi - accidental - NATURALS[idx]) // 12 - 5
        symbol = letter + "'" * max(register, 0) + "," * max(-register, 0)
        number = ("♯" if alteration > 0 else "♭" if alteration < 0 else "") + str(degree + 1)
        number += "̇" * max(0, octave) + "̣" * max(0, -octave)
        pitches.append((symbol, accidental, number))
    return signature, pitches


def numbered_duration(number, length):
    if length >= 8:
        return number + " –" * (length // 4 - 1)
    if length in (1, 2, 3):
        number += "̳" if length == 1 else "̲"
    return number + ("·" if length in (3, 6) else "")


def wrap_score_bars(body):
    """Budget horizontal space for both annotation rows before engraving a system."""
    bars = [bar.strip() for bar in body.split("|") if bar.strip()]
    lines, line, used = [], [], 0
    for bar in bars:
        width = 12  # barline and margin
        for degree, hole in re.findall(r'"_([^"]*?)\\n([^"]*)"', bar):
            glyphs = max(sum(not unicodedata.combining(c) for c in text)
                         for text in (degree, hole))
            width += 12 + 11 * glyphs
        if line and (used + width > 625 or len(line) >= 4):
            lines.append(" | ".join(line) + " |")
            line, used = [], 0
        line.append(bar)
        used += width
    return "\n".join(lines + [" | ".join(line) + " |]"])


def score_abc(result, instrument, meter, bpm, key="C", subdivision="straight", beats=(), downbeat=None, tie_ends=None, tonic=None):
    """Split at changes of active pitches; tie individual held pitches through changes."""
    harp = make_harp(instrument, key)
    notes = result.notes
    starts, stops, ternary, first, phase = grid_events(notes, bpm, beats, subdivision, downbeat)
    finish = max(stops.values(), default=first)
    bar = meter * 12
    boundaries = set(starts) | set(stops.values()) | {first, finish}
    boundaries.update(t for t in range(phase % bar, finish, bar) if t >= first)
    # Only subdivide beats with a ternary attack or release inside them.
    # Sustains spanning a whole beat need no artificial triplet or tie.
    ternary &= {t // 12 for t in boundaries if t % 12 in (4, 8)}
    for beat in ternary:
        boundaries.update(t for t in range(beat*12, beat*12+13, 4) if first <= t < finish)
    if finish % 12 and finish // 12 in ternary:
        boundaries.update(range(finish, (finish//12+1)*12+1, 4))
    boundaries = sorted(boundaries)
    tonic = tonic or relative_major(result.key)
    signature, symbols = spell_notes(notes, tonic)
    options = {n.pitch: hole_choices(harp, n.pitch) for n in notes}
    active = set()
    accidentals = {}
    parts = []
    previous_held = set()
    beam_beat = None
    for left, right in zip(boundaries, boundaries[1:]):
        active = {i for i in active if stops[i] > left}
        active.update(starts[left])
        sounding = sorted(active, key=lambda i: notes[i].frequency)
        label = tab_label([options[notes[i].pitch] for i in sounding]) if sounding else " "
        t = left
        while t < right:
            triplet = t // 12 in ternary
            span = 4 if triplet else next(s for s in (48, 36, 24, 18, 12, 9, 6, 3) if s <= right-t)
            length = 2 if triplet else span // 3
            prefix = "(3" if triplet and t % 12 == 0 else ""
            if triplet and t == first and t % 12:
                prefix = f"(3:2:{(12 - t % 12) // 4}"
            number = "+".join(symbols[i][2] for i in sounding) or "0"
            prefix += f'"_{numbered_duration(number, length)}\\n{label}"'
            pitches = []
            for i in sounding:
                letter, alter, _ = symbols[i]
                mark = ""
                if alter != accidentals.get(letter, signature[letter[0]]):
                    mark = "^" * alter if alter > 0 else "_" * -alter if alter < 0 else "="
                accidentals[letter] = alter
                pitches.append(mark + letter)
            held = [stops[i] > t + span for i in sounding]
            if tie_ends is not None:
                # abcjs otherwise matches incoming partial ties by chord index.
                # An inner voice can change index or become the only note.
                tie_ends.append([i in previous_held for i in sounding])
                previous_held = {i for i, continues in zip(sounding, held) if continues}
            tie_all = bool(held) and all(held)
            if len(pitches) > 1:
                token = "[" + "".join(p + ("-" if tie and not tie_all else "")
                                      for p, tie in zip(pitches, held)) + "]"
            else:
                token = pitches[0] if pitches else "z"
            text = f"{prefix}{token}{length}" + ("-" if tie_all else "")
            beat = (t-phase) // 12 if length < 4 else None
            if beat is not None and beat == beam_beat:
                parts[-1] += text
            else:
                parts.append(text)
            beam_beat = beat
            t += span
            if (t - phase) % bar == 0:
                beam_beat = None
                accidentals.clear()
                parts.append("|")
    body = wrap_score_bars(" ".join(parts))
    return (f"X:1\n%%propagate-accidentals octave\n%%barnumbers 0\n"
            f"%%annotationfont Helvetica 14\nP:1={tonic}\nM:{meter}/4\nL:1/16\n"
            f"Q:1/4={round(bpm)}\nK:{tonic}\n" + body + "\n")
