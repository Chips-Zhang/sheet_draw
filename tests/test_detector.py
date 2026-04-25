"""
测试用例
"""

import unittest
import numpy as np

from sheet_draw import PitchDetector, ToneAnalyzer
from sheet_draw.audio_utils import (
    generate_sine_wave,
    generate_note,
    note_to_frequency,
    frequency_to_note
)


class TestPitchDetector(unittest.TestCase):
    """测试音高检测器"""
    
    def setUp(self):
        self.detector = PitchDetector()
    
    def test_detect_a4(self):
        """测试检测A4 (440Hz)"""
        audio = generate_sine_wave(440.0, duration=1.0)
        result = self.detector.detect(audio)
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['frequency'], 440.0, delta=2.0)
        self.assertEqual(result['note'], 'A4')
    
    def test_detect_c4(self):
        """测试检测C4 (约262Hz)"""
        audio = generate_sine_wave(261.63, duration=1.0)
        result = self.detector.detect(audio)
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['frequency'], 261.63, delta=2.0)
        self.assertEqual(result['note'], 'C4')
    
    def test_detect_g3(self):
        """测试检测G3 (约196Hz)"""
        audio = generate_note('G3', duration=1.0)
        result = self.detector.detect(audio)
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['frequency'], 196.0, delta=3.0)
        self.assertEqual(result['note'], 'G3')
    
    def test_different_methods(self):
        """测试不同算法"""
        audio = generate_note('A4', duration=1.0)
        
        for method in ['yin', 'autocorr', 'fft']:
            detector = PitchDetector(method=method)
            result = detector.detect(audio)
            
            self.assertIsNotNone(result, f"{method} should detect pitch")
            # 允许较大误差，因为不同算法精度不同
            self.assertAlmostEqual(result['frequency'], 440.0, delta=10.0)


class TestAudioUtils(unittest.TestCase):
    """测试音频工具函数"""
    
    def test_generate_sine_wave(self):
        """测试生成正弦波"""
        audio = generate_sine_wave(440.0, duration=0.5, sample_rate=44100)
        
        expected_length = int(0.5 * 44100)
        self.assertEqual(len(audio), expected_length)
        self.assertTrue(np.all(np.abs(audio) <= 0.5))
    
    def test_generate_note(self):
        """测试生成音符"""
        audio = generate_note('A4', duration=0.5)
        
        expected_length = int(0.5 * 44100)
        self.assertEqual(len(audio), expected_length)
    
    def test_frequency_to_note(self):
        """测试频率转音符"""
        self.assertEqual(frequency_to_note(440.0), 'A4')
        self.assertEqual(frequency_to_note(261.63), 'C4')
        self.assertEqual(frequency_to_note(523.25), 'C5')
    
    def test_note_to_frequency(self):
        """测试音符转频率"""
        self.assertAlmostEqual(note_to_frequency('A4'), 440.0, delta=0.1)
        self.assertAlmostEqual(note_to_frequency('C4'), 261.63, delta=1.0)


class TestToneAnalyzer(unittest.TestCase):
    """测试音调分析器"""
    
    def test_analyze_stable_tone(self):
        """测试分析稳定的音调"""
        audio = generate_note('E4', duration=2.0)
        
        analyzer = ToneAnalyzer()
        result = analyzer.analyze(audio)
        
        self.assertNotIn('error', result)
        self.assertEqual(result['note'], 'E4')
        self.assertGreater(result['stability'], 0.9)  # 非常稳定
        self.assertGreater(result['pitch_count'], 10)


if __name__ == '__main__':
    unittest.main()
