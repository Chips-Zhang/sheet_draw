"""
节奏分析 (Beat Tracking) - 检测节拍位置和BPM
"""

import numpy as np
from typing import List, Tuple, Optional
import scipy.signal
from scipy.ndimage import uniform_filter1d


class BeatTracker:
    """节拍追踪器"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        tempo_min: float = 60,    # 最小BPM
        tempo_max: float = 200    # 最大BPM
    ):
        """
        初始化节拍追踪器
        
        Args:
            sample_rate: 采样率
            hop_length: 帧移
            tempo_min: 最小检测速度 (BPM)
            tempo_max: 最大检测速度 (BPM)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
    
    def analyze(self, audio: np.ndarray) -> dict:
        """
        分析音频的节奏信息
        
        Args:
            audio: 音频数据
            
        Returns:
            包含节奏信息的字典:
            - bpm: 每分钟节拍数
            - beats: 节拍时间列表 (秒)
            - beat_frames: 节拍帧索引
            - onset_envelope: onset强度曲线
        """
        # 计算onset强度曲线
        onset_env = self._compute_onset_envelope(audio)
        
        # 估计BPM
        bpm = self._estimate_tempo(onset_env)
        
        # 追踪节拍位置
        beats = self._track_beats(onset_env, bpm)
        
        return {
            "bpm": bpm,
            "beats": [b * self.hop_length / self.sample_rate for b in beats],
            "beat_frames": beats,
            "onset_envelope": onset_env
        }
    
    def _compute_onset_envelope(self, audio: np.ndarray) -> np.ndarray:
        """计算onset强度包络"""
        # 分帧
        frame_length = 2048
        hop_length = self.hop_length
        
        num_frames = (len(audio) - frame_length) // hop_length + 1
        envelope = np.zeros(num_frames)
        
        prev_spec = None
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            
            # 加窗
            window = np.hanning(len(frame))
            frame = frame * window
            
            # FFT
            spec = np.abs(np.fft.rfft(frame))
            
            if prev_spec is not None and len(spec) == len(prev_spec):
                # 计算频谱通量
                diff = spec - prev_spec
                envelope[i] = np.sum(np.maximum(0, diff))
            
            prev_spec = spec
        
        # 归一化
        if envelope.max() > 0:
            envelope = envelope / envelope.max()
        
        # 平滑
        envelope = uniform_filter1d(envelope, size=3)
        
        return envelope
    
    def _estimate_tempo(self, onset_env: np.ndarray) -> float:
        """
        估计BPM (基于自相关)
        """
        # 计算自相关
        autocorr = np.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 转换为lag范围 (对应BPM范围)
        # BPM = 60 / (lag * hop_length / sample_rate)
        # lag = 60 * sample_rate / (BPM * hop_length)
        
        lag_max = int(60 * self.sample_rate / (self.tempo_min * self.hop_length))
        lag_min = int(60 * self.sample_rate / (self.tempo_max * self.hop_length))
        
        lag_max = min(lag_max, len(autocorr) - 1)
        
        if lag_min >= lag_max:
            return 120  # 默认值
        
        # 在有效范围内寻找峰值
        autocorr_range = autocorr[lag_min:lag_max]
        peak_idx = np.argmax(autocorr_range)
        best_lag = peak_idx + lag_min
        
        # 计算BPM
        bpm = 60 * self.sample_rate / (best_lag * self.hop_length)
        
        return round(bpm, 1)
    
    def _track_beats(self, onset_env: np.ndarray, bpm: float) -> List[int]:
        """
        追踪节拍位置
        
        使用动态规划找到最优节拍位置
        """
        # 计算理论节拍间隔 (帧)
        beat_interval = int(60 * self.sample_rate / (bpm * self.hop_length))
        
        # 简单的峰值检测方法
        # 找到onset_env中的周期性峰值
        
        # 首先找到所有局部最大值
        local_max = []
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1]:
                local_max.append((i, onset_env[i]))
        
        # 按强度排序
        local_max.sort(key=lambda x: x[1], reverse=True)
        
        # 选择节拍位置 (保持一定的间隔)
        beats = []
        tolerance = beat_interval // 4  # 允许的位置偏差
        
        for frame, strength in local_max:
            # 检查是否与已选节拍太接近
            too_close = any(abs(frame - b) < beat_interval - tolerance for b in beats)
            if not too_close:
                beats.append(frame)
        
        beats.sort()
        
        return beats
    
    def detect_bpm(self, audio: np.ndarray) -> float:
        """
        只检测BPM
        
        Args:
            audio: 音频数据
            
        Returns:
            BPM值
        """
        result = self.analyze(audio)
        return result["bpm"]


class SimpleTempoDetector:
    """
    简化的速度检测器
    
    使用能量包络的自相关检测BPM
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def detect(self, audio: np.ndarray) -> float:
        """
        检测BPM
        
        Args:
            audio: 音频数据
            
        Returns:
            BPM值
        """
        # 计算能量包络
        hop_size = 512
        num_frames = len(audio) // hop_size
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + hop_size
            frame = audio[start:end]
            energy[i] = np.sqrt(np.mean(frame ** 2))
        
        # 差分 (突出节奏变化)
        diff = np.diff(energy)
        diff = np.maximum(0, diff)  # 只保留正值
        
        # 自相关
        autocorr = np.correlate(diff, diff, mode='full')
        autocorr = autocorr[len(autocorr)//2 + 10:]  # 跳过前10个lag
        
        # 寻找峰值
        # BPM范围: 60-200
        # 对应lag范围 (假设hop_size=512, sample_rate=44100)
        # lag = 60s / BPM / (hop_size/sample_rate) = 60*44100/(BPM*512)
        # BPM=60: lag = 86
        # BPM=200: lag = 26
        
        lag_min = int(60 * self.sample_rate / (200 * hop_size))
        lag_max = int(60 * self.sample_rate / (60 * hop_size))
        
        lag_max = min(lag_max, len(autocorr))
        
        if lag_min >= lag_max or lag_max > len(autocorr):
            return 120  # 默认
        
        # 找到最大峰值
        peak_idx = np.argmax(autocorr[lag_min:lag_max])
        best_lag = peak_idx + lag_min
        
        # 计算BPM
        bpm = 60 * self.sample_rate / (best_lag * hop_size)
        
        return round(bpm)


def estimate_bpm_simple(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """
    简单的BPM检测函数
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        
    Returns:
        BPM值
    """
    detector = SimpleTempoDetector(sample_rate)
    return detector.detect(audio)
