from sheet_draw import ChromaticHarmonica, DiatonicHarmonica, Note, TranscriptionResult
from sheet_draw.web import score_abc


def test_combined_score_rest_and_barline_tie():
    notes = [Note("C4", 261.63, 0.25, 1.25, 1), Note("D4", 293.66, 1.5, 0.5, 1)]
    result = TranscriptionResult(notes, "C major", 120, [])
    score = score_abc(result, "diatonic", 2, 120)
    assert "M:2/4" in score
    assert r'"_0̲\n "z2 "_1·\n1"C6- | "_1\n1"C4 "_2\n①"D4' in score


def test_solo_and_richter_positions():
    chromatic = ChromaticHarmonica()
    b4 = chromatic.note_to_hole("B4")
    assert (b4.hole, b4.action, b4.button) == (4, "draw", False)
    harp = DiatonicHarmonica()
    high_bend = harp.note_to_hole("A#6")
    assert (high_bend.hole, high_bend.action, high_bend.bend) == (10, "blow", "full")
    assert harp.note_to_hole("G#4").bend == "three_half"


def test_partial_tie_follows_pitch_when_lower_voice_changes_or_drops_out():
    notes = [
        Note("C5", 523.25, 0, .5, 1), Note("E5", 659.25, 0, 1, 1),
        Note("D5", 587.33, .5, .25, 1), Note("E5", 659.25, 1.5, .25, 1),
    ]
    ends = []
    abc = score_abc(TranscriptionResult(notes, "C major", 120, []), "diatonic",
                    2, 120, tie_ends=ends)
    assert "[C'E'-]4" in abc
    assert ends == [[False, False], [False, True], [True], [], [False]]


def test_blue_lotus_key_signature_repeated_attacks_and_octaves():
    from sheet_draw import note_to_frequency
    notes = [Note(p, note_to_frequency(p), i/4, .25) for i,p in
             enumerate(["E5", "E5", "F#5", "A5"])]
    abc = score_abc(TranscriptionResult(notes, "A major", 120, []), "paddy", 4, 120, "A")
    assert "K:A\n" in abc and "P:1=A" in abc
    assert abc.count(r'''"_5̲\n6"E'2''') == 2
    assert r'''"_6̲\n⑥"F'2''' in abc and r'''"_1̲̇\n7"A'2''' in abc
    assert "'2-" not in abc


def test_chromatic_slide_keeps_breath_mark_and_stays_on_hole_row():
    from sheet_draw import note_to_frequency
    notes = [Note(p, note_to_frequency(p), i/4, .25)
             for i,p in enumerate(["C#4","D#4","C4","D4"])]
    abc = score_abc(TranscriptionResult(notes, "C major", 120, []), "chromatic", 4, 120)
    for label in ("1↑", "①↑", "1", "①"):
        assert r"\n" + label + '"' in abc
    assert "推" not in abc


def test_system_breaks_take_annotation_width_into_account():
    from sheet_draw.notation import wrap_score_bars
    bar = (r'"_♯2̇\n⑤↑"D2' * 12)
    systems = wrap_score_bars(" | ".join([bar]*4)).splitlines()
    assert len(systems) == 4
    assert sum(line.count("⑤↑") for line in systems) == 48
