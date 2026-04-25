"""
音乐扒谱器 (Music Transcription) - 完整的音频到曲谱转换

整合以下功能:
- Onset检测 (音符分割)
- 音高检测
- 节奏分析 (Beat tracking, BPM)
- 调性检测
- 曲谱生成 (简谱, MIDI)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .detector import PitchDetector
from .onset import OnsetDetector
from .rhythm import BeatTracker
from .key_detection import KeyDetector
from .score import Note, TranscriptionResult, quantize_notes


@dataclass
class TranscriptionConfig:
    """扒谱配置"""
    sample_rate: int = 44100
    hop_length: int = 512
    
    # Onset检测
    onset_threshold: float = 0.3
    
    # BPM范围
    tempo_min: float = 60
    tempo_max: float = 180
    
    # 量化设置
    quantization: str = "1/16"  # 1/4, 1/8, 1/16
    
    # 音高平滑
    pitch_smoothing: bool = True
    
    # 最小音符持续时间 (秒)
    min_note_duration: float = 0.05
    
    # 是否量化
    enable_quantization: bool = True
    
    # 人声分离
    enable_vocal_separation: bool = False
    vocal_separation_method: str = "hpss"  # 'hpss', 'spectral', 'spleeter'
    
    # 口琴优化
    harmonica_optimization: bool = False
    harmonica_type: str = "chromatic"  # 'chromatic', 'diatonic'
    harmonica_key: str = "C"


class MusicTranscriber:
    """音乐扒谱器"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        初始化扒谱器
        
        Args:
            config: 配置对象
        """
        self.config = config or TranscriptionConfig()
        
        # 初始化各个模块
        self.onset_detector = OnsetDetector(
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length,
            threshold=self.config.onset_threshold
        )
        
        self.pitch_detector = PitchDetector(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_length
        )
        
        self.beat_tracker = BeatTracker(
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length,
            tempo_min=self.config.tempo_min,
            tempo_max=self.config.tempo_max
        )
        
        self.key_detector = KeyDetector()
    
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        扒谱主函数
        
        Args:
            audio: 音频数据 (单声道)
            
        Returns:
            扒谱结果
        """
        print("🎵 开始扒谱...")
        
        # Step 0: 人声分离 (可选)
        if self.config.enable_vocal_separation:
            print("  Step 0/6: 人声分离...")
            from .vocal_separation import VocalSeparator
            separator = VocalSeparator(
                sample_rate=self.config.sample_rate,
                method=self.config.vocal_separation_method
            )
            audio = separator.extract_vocal(audio)
            print("    人声分离完成")
        
        # Step 1: Onset检测 (找到每个音符的开始)
        print("  Step 1/6: 检测音符分割点...")
        onsets = self._detect_onsets(audio)
        print(f"    检测到 {len(onsets)} 个音符")
        
        # Step 2: 节奏分析
        print("  Step 2/6: 分析节奏...")
        rhythm_info = self.beat_tracker.analyze(audio)
        bpm = rhythm_info["bpm"]
        beats = rhythm_info["beats"]
        print(f"    BPM: {bpm}")
        print(f"    检测到 {len(beats)} 个节拍")
        
        # Step 3: 音高检测
        print("  Step 3/6: 检测音高...")
        notes = self._detect_notes(audio, onsets)
        print(f"    成功检测 {len(notes)} 个音符")
        
        # Step 4: 调性检测
        print("  Step 4/6: 分析调性...")
        key_info = self._detect_key(notes)
        key = key_info["key"]
        print(f"    调性: {key}")
        
        # Step 5: 量化和优化
        print("  Step 5/6: 量化音符...")
        if self.config.enable_quantization:
            notes = quantize_notes(notes, bpm, self.config.quantization)
        
        # Step 6: 口琴优化 (可选)
        if self.config.harmonica_optimization:
            print("  Step 6/6: 口琴优化...")
            from .harmonica import HarmonicaScoreGenerator, create_chromatic_harmonica, create_diatonic_harmonica
            if self.config.harmonica_type == "chromatic":
                harp = create_chromatic_harmonica(self.config.harmonica_key)
            else:
                harp = create_diatonic_harmonica(self.config.harmonica_key)
            
            unplayable = HarmonicaScoreGenerator(harp).get_unplayable_notes(notes)
            if unplayable:
                print(f"    注意: 有 {len(unplayable)} 个音符可能难以演奏")
                print(f"    不可演奏音符: {', '.join(list(set(unplayable))[:5])}")
        
        print("✅ 扒谱完成!")
        
        return TranscriptionResult(
            notes=notes,
            key=key,
            bpm=bpm,
            beats=beats
        )
    
    def transcribe_file(self, filepath: str) -> TranscriptionResult:
        """
        从音频文件扒谱
        
        Args:
            filepath: 音频文件路径
            
        Returns:
            扒谱结果
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(filepath)
            self.config.sample_rate = sr
            self._update_modules()
        except Exception as e:
            raise RuntimeError(f"无法读取音频文件: {e}")
        
        # 转换为单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return self.transcribe(audio)
    
    def _update_modules(self):
        """更新模块配置"""
        self.onset_detector.sample_rate = self.config.sample_rate
        self.pitch_detector.sample_rate = self.config.sample_rate
        self.beat_tracker.sample_rate = self.config.sample_rate
    
    def _detect_onsets(self, audio: np.ndarray) -> List[float]:
        """检测音符开始点"""
        return self.onset_detector.detect(audio)
    
    def _detect_notes(self, audio: np.ndarray, onsets: List[float]) -> List[Note]:
        """
        检测每个onset的音高
        
        Args:
            audio: 音频数据
            onsets: 音符开始时间列表
            
        Returns:
            音符列表
        """
        notes = []
        
        for i, onset in enumerate(onsets):
            # 计算音符结束时间 (下一个onset或音频结束)
            if i < len(onsets) - 1:
                end_time = onsets[i + 1]
            else:
                end_time = len(audio) / self.config.sample_rate
            
            duration = end_time - onset
            
            # 跳过太短的音符
            if duration < self.config.min_note_duration:
                continue
            
            # 截取音符片段 (取中间部分，避免attack和release)
            onset_sample = int(onset * self.config.sample_rate)
            
            # 取音符中间的一段来检测音高
            stable_start = onset_sample + int(0.05 * self.config.sample_rate)  # 跳过前50ms
            stable_end = min(
                onset_sample + int(0.3 * self.config.sample_rate),  # 最多取300ms
                int(end_time * self.config.sample_rate)
            )
            
            if stable_end - stable_start < self.config.hop_length:
                stable_end = min(onset_sample + int(duration * self.config.sample_rate), len(audio))
                stable_start = onset_sample
            
            note_audio = audio[stable_start:stable_end]
            
            # 检测音高
            pitch_result = self.pitch_detector.detect(note_audio)
            
            if pitch_result:
                note = Note(
                    pitch=pitch_result["note"],
                    frequency=pitch_result["frequency"],
                    start_time=onset,
                    duration=duration,
                    velocity=1.0
                )
                notes.append(note)
        
        # 音高平滑处理
        if self.config.pitch_smoothing and len(notes) > 1:
            notes = self._smooth_pitch(notes)
        
        return notes
    
    def _smooth_pitch(self, notes: List[Note]) -> List[Note]:
        """
        平滑音高 (去除噪声和颤音影响)
        """
        if len(notes) < 3:
            return notes
        
        # 使用中值滤波平滑
        window_size = min(5, len(notes))
        if window_size < 3:
            return notes
        
        # 提取MIDI note numbers
        midi_notes = []
        for note in notes:
            midi_num = int(round(69 + 12 * np.log2(note.frequency / 440.0)))
            midi_notes.append(midi_num)
        
        # 中值滤波
        from scipy.ndimage import median_filter
        smoothed = median_filter(midi_notes, size=window_size, mode='nearest')
        
        # 更新音符频率
        for i, note in enumerate(notes):
            if smoothed[i] != midi_notes[i]:
                # 频率被平滑，更新频率
                new_freq = 440.0 * (2 ** ((smoothed[i] - 69) / 12))
                # 更新音高名称
                from .key_detection import KeyDetector
                kd = KeyDetector()
                new_note_name = kd._freq_to_note_name(new_freq)
                
                notes[i] = Note(
                    pitch=new_note_name or note.pitch,
                    frequency=new_freq,
                    start_time=note.start_time,
                    duration=note.duration,
                    velocity=note.velocity
                )
        
        return notes
    
    def _detect_key(self, notes: List[Note]) -> Dict:
        """检测调性"""
        if not notes:
            return {"key": "C major", "confidence": 0}
        
        pitch_names = [note.pitch for note in notes]
        return self.key_detector.detect(pitch_names)
    
    def transcribe_simple(self, audio: np.ndarray) -> str:
        """
        简化的扒谱接口，直接返回简谱文本
        
        Args:
            audio: 音频数据
            
        Returns:
            简谱文本
        """
        result = self.transcribe(audio)
        return result.to_jianpu()


# 便捷函数
def transcribe_audio(audio: np.ndarray, sample_rate: int = 44100) -> TranscriptionResult:
    """
    快速扒谱函数
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        
    Returns:
        扒谱结果
    """
    config = TranscriptionConfig(sample_rate=sample_rate)
    transcriber = MusicTranscriber(config)
    return transcriber.transcribe(audio)


def transcribe_file(filepath: str) -> TranscriptionResult:
    """
    从文件快速扒谱
    
    Args:
        filepath: 音频文件路径
        
    Returns:
        扒谱结果
    """
    transcriber = MusicTranscriber()
    return transcriber.transcribe_file(filepath)
