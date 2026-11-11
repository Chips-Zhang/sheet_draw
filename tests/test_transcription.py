import unittest

import numpy as np

from sheet_draw import KeyDetector, MusicTranscriber, TranscriptionConfig, generate_note


class TestTranscription(unittest.TestCase):
    def setUp(self):
        self.rate = 16000
        self.transcriber = MusicTranscriber(TranscriptionConfig(
            sample_rate=self.rate, enable_quantization=False))

    def test_legato_leap_and_repeated_pitch_after_rest(self):
        pitches = ["B4", "D5", "A4", "A4"]
        audio = np.concatenate([
            np.zeros(1600), generate_note("B4", .35, self.rate),
            generate_note("D5", .18, self.rate), generate_note("A4", .22, self.rate),
            np.zeros(3200), generate_note("A4", .24, self.rate)])
        notes = self.transcriber.transcribe(audio).notes
        self.assertEqual([n.pitch for n in notes], pitches)
        np.testing.assert_allclose([n.start_time for n in notes], [.10, .45, .63, 1.05], atol=.035)
        np.testing.assert_allclose([n.duration for n in notes], [.35, .18, .22, .24], atol=.04)

    def test_short_silence(self):
        self.assertEqual(self.transcriber.transcribe(np.zeros(160)).notes, [])

    def test_non_c_tonic(self):
        pitches = ["G4"] * 8 + ["B4"] * 5 + ["D5"] * 6 + ["F#4"] * 2 + ["C5"] * 3
        self.assertEqual(KeyDetector().detect(pitches)["key"], "G major")
