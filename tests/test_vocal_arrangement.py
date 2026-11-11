import numpy as np

from sheet_draw.audio_utils import note_to_frequency
from sheet_draw.harmonica import make_harp
from sheet_draw.notation import hole_choices
from sheet_draw.score import Note
from sheet_draw.vocal_arrangement import arrange, fit_missing_octaves
from sheet_draw.vocal_separation import preprocess_for_transcription


def note(pitch):
    return Note(pitch, note_to_frequency(pitch), .1, .25)


def test_paddy_third_blow_and_bend_range_follow_key():
    paddy = make_harp("paddy", "G")
    assert any(p.hole == 3 and p.action == "blow" for p in hole_choices(paddy, "E4"))
    assert not any(p.hole == 3 for p in hole_choices(paddy, "D4"))
    assert any(p.hole == 3 for p in hole_choices(make_harp("diatonic", "G"), "D4"))


def test_song_key_differs_from_harp_and_manual_shift_has_priority():
    melody = [note("C4"), note("D4")]
    fitted, shift, _ = arrange(melody, "diatonic", "C", target_tonic="G", source_tonic=0)
    assert shift % 12 == 7
    assert fitted[1].frequency / fitted[0].frequency == melody[1].frequency / melody[0].frequency
    manual, shift, _ = arrange(melody, "diatonic", "C", -2, target_tonic="G")
    assert (manual[0].pitch, shift) == ("A#3", -2)


def test_octave_repair_leaves_playable_notes_and_time_untouched():
    melody = [note("F3"), note("C4")]
    fitted, moved, missing = fit_missing_octaves(melody, "chromatic", "C")
    assert ([n.pitch for n in fitted], moved, missing) == (["F4", "C4"], 1, 0)
    assert fitted[1] is melody[1] and fitted[0].start_time == melody[0].start_time


def test_long_stems_are_stitched_on_input_timeline(monkeypatch):
    signal = np.linspace(-1,1,96001,dtype=np.float32)
    monkeypatch.setattr("sheet_draw.vocal_separation._neural_stems",
                        lambda data, sr, method: {"vocals": data})
    np.testing.assert_array_equal(preprocess_for_transcription(signal,1000,"demucs"),signal)
