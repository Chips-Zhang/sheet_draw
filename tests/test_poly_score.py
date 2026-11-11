import numpy as np
import pytest

from sheet_draw import DiatonicHarmonica, Note, TranscriptionResult, note_to_frequency
from sheet_draw.basic_pitch_backend import select_duet
from sheet_draw.notation import beat_coordinates, score_abc


def note(pitch, start, duration, volume=1.):
    return Note(pitch, note_to_frequency(pitch), start, duration, volume)


def render(notes, **kwargs):
    return score_abc(TranscriptionResult(notes, "C major", 120, []), "diatonic", 4, 120, **kwargs)


def test_reed_pair_preserves_staggered_release_and_alternate_hole():
    # G4 can be drawn on 2 or blown on 3. With E4 the latter is required.
    notes = [note("E4", 0, .5), note("G4", 0, .25, .85)]
    abc = render(notes)
    assert r'"_3+5̲\n2+3"[E-G]2"_3̲\n2"E2' in abc


def test_duet_accepts_adjacent_same_breath_and_rejects_cross_breath():
    lead = note("C5", .5, .5)
    candidates = [lead, note("E5", .505, .49, .8), note("D5", .502, .49, .9)]
    assert [n.pitch for n in select_duet(candidates, "diatonic", "C")] == ["C5", "E5"]


@pytest.mark.parametrize("key,pitch", [("G", "G3"), ("Bb", "A#3"), ("C", "C4")])
def test_key_shifts_hole_and_range(key, pitch):
    harp = DiatonicHarmonica(key)
    assert harp.note_to_hole(pitch).hole == 1
    assert harp.get_range()[0] == pytest.approx(note_to_frequency(pitch))


def test_triplet_then_straight_and_pickup():
    notes = [note(p, i/6, 1/6) for i, p in enumerate(["C4", "D4", "E4"])]
    notes += [note("G4", .5, .5)]
    abc = render(notes, subdivision="auto", downbeat=.5)
    assert abc.count("(3") == 1
    assert 'E2 |' in abc and 'G4' in abc


def test_tempo_map_slowdown_and_missing_beat():
    times = [0, .5, 1.05, 1.625, 2.2]
    np.testing.assert_allclose(beat_coordinates(times, 110, [0, .5, 1.05, 2.2]), np.arange(5))
