import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from sheet_draw import MusicTranscriber, TranscriptionConfig
from sheet_draw.vocal_separation import SpleeterWrapper, VocalSeparator, preprocess_for_transcription


class TestSeparation(unittest.TestCase):
    def test_dsp_preserves_timeline_and_reconstructs_mixture(self):
        # A non-hop-aligned sample count and off-centre impulse expose padding/latency errors.
        x = np.zeros(32031, dtype=np.float32)
        x[13571] = .8
        for method in ["hpss", "spectral", "repet", "median_filtering", "center"]:
            with self.subTest(method=method):
                lead, backing = VocalSeparator(48000, method=method).separate(np.column_stack((x, x)))
                self.assertEqual(lead.shape, x.shape)
                np.testing.assert_allclose(lead + backing, x, atol=1e-6)
                if method in ("spectral", "center"):
                    self.assertLessEqual(abs(np.argmax(np.abs(lead)) - 13571), 1)

    def test_stereo_survives_file_loading_and_routes_target(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "input.wav"
            sr = 16000
            stereo = np.column_stack([np.full(sr, .1), np.full(sr, -.2)])
            sf.write(file, stereo, sr, subtype="FLOAT")
            with patch("sheet_draw.vocal_separation._neural_stems",
                       return_value={"other": np.zeros_like(stereo)}) as model:
                transcriber = MusicTranscriber(TranscriptionConfig(enable_vocal_separation=True,
                    vocal_separation_method="demucs", separation_target="other"))
                self.assertEqual(transcriber.transcribe_file(file).notes, [])
                samples, rate, method = model.call_args.args
                np.testing.assert_allclose(samples, stereo, atol=1e-6)
                self.assertEqual((rate, method), (sr, "demucs"))

    def test_spleeter_resampling_contract(self):
        # Exercise the adapter independently of its optional TensorFlow runtime.
        wrapper = object.__new__(SpleeterWrapper)
        wrapper.available = True
        wrapper.separator = Mock()
        wrapper.separator.separate.side_effect = lambda audio: {"vocals": audio}
        audio = np.zeros(16001, dtype=np.float32)
        audio[4000] = .5
        result = wrapper.extract_vocal(audio, 16000)
        model_input = wrapper.separator.separate.call_args.args[0]
        self.assertEqual(model_input.shape, (44103, 2))
        self.assertEqual(result.shape, audio.shape)
        self.assertEqual(np.argmax(result), 4000)

    def test_short_and_mono_input(self):
        self.assertEqual(len(preprocess_for_transcription(np.ones(32), 16000, "hpss")), 32)
        with self.assertRaisesRegex(ValueError, "双声道"):
            preprocess_for_transcription(np.ones(32), 16000, "center")
