"""
音高检测核心模块

实现了多种音高检测算法：
- YIN算法
- 自相关法
- FFT峰值法
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import wave
import struct


class PitchDetector:
    """音高检测器"""
    
    # 标准音符频率映射 (A4 = 440Hz)
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(
        self,
        sample_rate: int = 44100,
        frame_size: int = 2048,
        hop_size: int = 512,
        threshold: float = 0.1,
        method: str = "yin"
    ):
        """
        初始化音高检测器
        
        Args:
            sample_rate: 采样率
            frame_size: 分析帧大小
            hop_size: 帧移
            threshold: YIN算法阈值 (0.0-1.0)
            method: 检测方法 ('yin', 'autocorr', 'fft')
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.threshold = threshold
        self.method = method
        
        # 有效音高范围 (Hz)
        self.min_freq = 50   # 最低检测频率 (~C2)
        self.max_freq = 2000 # 最高检测频率 (~C7)
    
    def detect(self, audio: np.ndarray) -> Optional[Dict]:
        """
        检测音频的音高
        
        Args:
            audio: 音频数据 (1D numpy array)
            
        Returns:
            包含音高信息的字典，或 None (如果未检测到)
        """
        # 确保音频是1D数组
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # 归一化
        audio = audio.astype(np.float64)
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()
        
        # 选择检测方法
        if self.method == "yin":
            freq = self._yin_detect(audio)
        elif self.method == "autocorr":
            freq = self._autocorr_detect(audio)
        elif self.method == "fft":
            freq = self._fft_detect(audio)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if freq is None:
            return None
        
        # 转换为音符信息
        note_info = self._freq_to_note(freq)
        
        return {
            "frequency": freq,
            "note": note_info["note"],
            "octave": note_info["octave"],
            "tone": note_info["tone"],
            "cents": note_info["cents"],
            "midi": note_info["midi"]
        }
    
    def detect_from_file(self, filepath: str) -> Optional[Dict]:
        """
        从音频文件检测音高
        
        Args:
            filepath: 音频文件路径
            
        Returns:
            包含音高信息的字典
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(filepath)
            self.sample_rate = sr
            return self.detect(audio)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def _yin_detect(self, audio: np.ndarray) -> Optional[float]:
        """
        YIN算法实现
        
        YIN算法通过计算差分函数和累积均值归一化来检测基频
        对噪声有较好的鲁棒性
        """
        # 截取帧
        if len(audio) < self.frame_size:
            return None
        
        frame = audio[:self.frame_size]
        
        # 计算差分函数 (Difference Function)
        tau_max = min(self.frame_size // 2, int(self.sample_rate / self.min_freq))
        tau_min = max(1, int(self.sample_rate / self.max_freq))
        
        diff = np.zeros(tau_max)
        for tau in range(tau_max):
            diff[tau] = np.sum((frame[:-tau_max] - frame[tau:-tau_max+tau])**2) if tau > 0 else 0
        
        # 累积均值归一化差分函数 (CMNDF)
        cmndf = np.zeros(tau_max)
        cmndf[0] = 1
        for tau in range(1, tau_max):
            if np.sum(diff[1:tau+1]) > 0:
                cmndf[tau] = diff[tau] * tau / np.sum(diff[1:tau+1])
            else:
                cmndf[tau] = 1
        
        # 寻找第一个低于阈值的谷值
        tau_estimate = None
        for tau in range(tau_min, tau_max):
            if cmndf[tau] < self.threshold:
                # 寻找局部最小值
                while tau + 1 < tau_max and cmndf[tau+1] < cmndf[tau]:
                    tau += 1
                tau_estimate = tau
                break
        
        if tau_estimate is None:
            return None
        
        # 抛物线插值提高精度
        if tau_estimate > 0 and tau_estimate < tau_max - 1:
            alpha = cmndf[tau_estimate - 1]
            beta = cmndf[tau_estimate]
            gamma = cmndf[tau_estimate + 1]
            
            if alpha + gamma - 2 * beta != 0:
                p = 0.5 * (alpha - gamma) / (alpha + gamma - 2 * beta)
                tau_estimate = tau_estimate + p
        
        # 计算频率
        frequency = self.sample_rate / tau_estimate
        
        # 验证频率范围
        if frequency < self.min_freq or frequency > self.max_freq:
            return None
        
        return frequency
    
    def _autocorr_detect(self, audio: np.ndarray) -> Optional[float]:
        """
        自相关法检测音高
        
        通过寻找自相关函数的峰值来确定基音周期
        """
        if len(audio) < self.frame_size:
            return None
        
        frame = audio[:self.frame_size]
        
        # 计算自相关
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        # 只考虑有效范围
        min_lag = int(self.sample_rate / self.max_freq)
        max_lag = min(int(self.sample_rate / self.min_freq), len(corr) - 1)
        
        if min_lag >= max_lag:
            return None
        
        # 寻找峰值
        corr_range = corr[min_lag:max_lag]
        if len(corr_range) == 0:
            return None
        
        peak_idx = np.argmax(corr_range) + min_lag
        
        # 验证峰值是否足够显著
        if corr[peak_idx] < 0.5 * corr[0]:
            return None
        
        frequency = self.sample_rate / peak_idx
        return frequency
    
    def _fft_detect(self, audio: np.ndarray) -> Optional[float]:
        """
        FFT峰值法检测音高
        
        通过频域分析寻找基频
        精度较低但计算速度快
        """
        if len(audio) < self.frame_size:
            return None
        
        frame = audio[:self.frame_size]
        
        # 加窗
        window = np.hanning(len(frame))
        frame = frame * window
        
        # FFT
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)
        
        # 频率轴
        freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
        
        # 只考虑有效范围
        valid_idx = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(valid_idx):
            return None
        
        valid_mag = magnitude[valid_idx]
        valid_freqs = freqs[valid_idx]
        
        # 寻找峰值
        peak_idx = np.argmax(valid_mag)
        frequency = valid_freqs[peak_idx]
        
        return frequency
    
    def _freq_to_note(self, frequency: float) -> Dict:
        """
        将频率转换为音符信息
        
        Args:
            frequency: 频率 (Hz)
            
        Returns:
            包含音符信息的字典
        """
        # A4 = 440Hz 作为参考
        A4 = 440.0
        
        # 计算与A4的半音数
        semitones = 12 * np.log2(frequency / A4)
        
        # 计算MIDI音符编号 (A4 = 69)
        midi_note = int(round(69 + semitones))
        
        # 计算音符名称和八度
        note_idx = (midi_note - 12) % 12
        octave = (midi_note - 12) // 12
        note_name = self.NOTE_NAMES[note_idx]
        
        # 计算音分偏差 (cents)
        rounded_semitones = round(semitones)
        cents = (semitones - rounded_semitones) * 100
        
        # 计算唱名 (简谱数字)
        # C大调: C=1, D=2, E=3, F=4, G=5, A=6, B=7
        tone = (note_idx + 3) % 7 + 1  # +3 因为C是第3个从A开始的半音
        
        return {
            "note": f"{note_name}{octave}",
            "octave": octave,
            "tone": tone,
            "cents": round(cents, 2),
            "midi": midi_note
        }
    
    def detect_continuous(
        self,
        audio: np.ndarray,
        hop_size: Optional[int] = None
    ) -> List[Dict]:
        """
        连续音高检测 (用于分析整个音频)
        
        Args:
            audio: 音频数据
            hop_size: 帧移 (默认使用初始化时的值)
            
        Returns:
            音高信息列表
        """
        if hop_size is None:
            hop_size = self.hop_size
        
        results = []
        
        # 分帧处理
        for i in range(0, len(audio) - self.frame_size, hop_size):
            frame = audio[i:i + self.frame_size]
            result = self.detect(frame)
            if result:
                result["time"] = i / self.sample_rate
                results.append(result)
        
        return results


class ToneAnalyzer:
    """音调分析器 - 分析一段音频的整体音调特征"""
    
    def __init__(self, detector: Optional[PitchDetector] = None):
        """
        初始化音调分析器
        
        Args:
            detector: 音高检测器实例
        """
        self.detector = detector or PitchDetector()
    
    def analyze(self, audio: np.ndarray) -> Dict:
        """
        分析音频的音调特征
        
        Args:
            audio: 音频数据
            
        Returns:
            音调分析结果
        """
        # 连续检测
        pitches = self.detector.detect_continuous(audio)
        
        if not pitches:
            return {"error": "No pitch detected"}
        
        # 提取频率
        frequencies = [p["frequency"] for p in pitches]
        
        # 统计分析
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        # 稳定性分析
        stability = self._calculate_stability(frequencies)
        
        # 获取主要音符
        note_counts = {}
        for p in pitches:
            note = p["note"]
            note_counts[note] = note_counts.get(note, 0) + 1
        
        dominant_note = max(note_counts.items(), key=lambda x: x[1])[0]
        
        # 转换为音符信息
        note_info = self.detector._freq_to_note(mean_freq)
        
        return {
            "mean_frequency": round(mean_freq, 2),
            "std_frequency": round(std_freq, 2),
            "stability": round(stability, 2),
            "dominant_note": dominant_note,
            "note": note_info["note"],
            "tone": note_info["tone"],
            "octave": note_info["octave"],
            "cents": note_info["cents"],
            "pitch_count": len(pitches),
            "frequency_range": {
                "min": round(min(frequencies), 2),
                "max": round(max(frequencies), 2)
            }
        }
    
    def _calculate_stability(self, frequencies: List[float]) -> float:
        """
        计算音高稳定性
        
        返回 0-1 之间的值，1表示非常稳定
        """
        if len(frequencies) < 2:
            return 1.0
        
        # 使用变异系数 (CV) 作为稳定性指标
        mean = np.mean(frequencies)
        std = np.std(frequencies)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        
        # 转换为稳定性分数 (CV越小越稳定)
        stability = max(0, min(1, 1 - cv))
        
        return stability
