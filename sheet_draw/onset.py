"""
音符分割 (Onset Detection) - 检测音符开始时间点
"""

import numpy as np
from typing import List, Tuple, Optional
import scipy.signal


class OnsetDetector:
    """音符开始点检测器"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_fft: int = 2048,
        threshold: float = 0.5,
        pre_max: float = 0.03,      # 峰值前最大时间 (秒)
        post_max: float = 0.03,     # 峰值后最大时间 (秒)
        pre_avg: float = 0.1,       # 均值计算前时间 (秒)
        post_avg: float = 0.1,      # 均值计算后时间 (秒)
        wait: float = 0.03          # 最小音符间隔 (秒)
    ):
        """
        初始化Onset检测器
        
        Args:
            sample_rate: 采样率
            hop_length: 帧移
            n_fft: FFT大小
            threshold: 检测阈值 (0.0-1.0)
            pre_max: 峰值前搜索窗口 (秒)
            post_max: 峰值后搜索窗口 (秒)
            pre_avg: 均值计算前窗口 (秒)
            post_avg: 均值计算后窗口 (秒)
            wait: 最小音符间隔 (秒)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.threshold = threshold
        
        # 转换为帧数
        self.pre_max = int(pre_max * sample_rate / hop_length)
        self.post_max = int(post_max * sample_rate / hop_length)
        self.pre_avg = int(pre_avg * sample_rate / hop_length)
        self.post_avg = int(post_avg * sample_rate / hop_length)
        self.wait = int(wait * sample_rate / hop_length)
    
    def detect(self, audio: np.ndarray) -> List[float]:
        """
        检测音符开始时间点
        
        Args:
            audio: 音频数据
            
        Returns:
            音符开始时间列表 (秒)
        """
        # 计算onset强度曲线
        onset_env = self._compute_onset_strength(audio)
        
        # 峰值检测
        peaks = self._peak_pick(onset_env)
        
        # 转换为时间
        times = [p * self.hop_length / self.sample_rate for p in peaks]
        
        return times
    
    def _compute_onset_strength(self, audio: np.ndarray) -> np.ndarray:
        """
        计算onset强度曲线 (Spectral Flux)
        
        Spectral Flux 测量相邻帧之间频谱能量的变化
        """
        # 计算STFT
        stft = self._stft(audio)
        
        # 取幅度谱
        mag = np.abs(stft)
        
        # 计算频谱通量 (Spectral Flux)
        flux = np.zeros(mag.shape[1])
        
        for i in range(1, mag.shape[1]):
            # 只取正值差异 (只关心能量增加)
            diff = mag[:, i] - mag[:, i-1]
            flux[i] = np.sum(np.maximum(0, diff))
        
        # 归一化
        if flux.max() > 0:
            flux = flux / flux.max()
        
        return flux
    
    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """短时傅里叶变换"""
        # 使用scipy的stft
        f, t, Zxx = scipy.signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary='constant'
        )
        return Zxx
    
    def _peak_pick(self, onset_env: np.ndarray) -> List[int]:
        """
        峰值检测
        
        使用滑动窗口局部最大值检测
        """
        peaks = []
        n = len(onset_env)
        
        for i in range(n):
            # 计算局部均值
            start_avg = max(0, i - self.pre_avg)
            end_avg = min(n, i + self.post_avg + 1)
            local_mean = np.mean(onset_env[start_avg:end_avg])
            
            # 计算局部最大值
            start_max = max(0, i - self.pre_max)
            end_max = min(n, i + self.post_max + 1)
            
            # 峰值条件:
            # 1. 当前点是局部最大值
            # 2. 当前值大于局部均值 + 阈值
            is_peak = (
                onset_env[i] == np.max(onset_env[start_max:end_max]) and
                onset_env[i] > local_mean + self.threshold
            )
            
            if is_peak:
                # 检查与上一个peak的间隔
                if not peaks or (i - peaks[-1]) >= self.wait:
                    peaks.append(i)
        
        return peaks
    
    def get_onset_frames(self, audio: np.ndarray) -> List[int]:
        """
        获取onset帧索引
        
        Args:
            audio: 音频数据
            
        Returns:
            帧索引列表
        """
        onset_env = self._compute_onset_strength(audio)
        return self._peak_pick(onset_env)
    
    def visualize(self, audio: np.ndarray, output_path: str = None):
        """
        可视化onset检测结果
        
        Args:
            audio: 音频数据
            output_path: 输出图像路径
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualization")
            return
        
        onset_env = self._compute_onset_strength(audio)
        onsets = self.detect(audio)
        
        times = np.arange(len(onset_env)) * self.hop_length / self.sample_rate
        
        plt.figure(figsize=(12, 4))
        plt.plot(times, onset_env, label='Onset Strength')
        plt.vlines(onsets, 0, 1, colors='r', linestyles='--', label='Detected Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        plt.title('Onset Detection')
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


class EnergyBasedOnsetDetector:
    """
    基于能量的Onset检测器
    
    适用于鼓点、打击乐等有明显能量变化的音频
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        threshold: float = 0.1
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.threshold = threshold
    
    def detect(self, audio: np.ndarray) -> List[float]:
        """基于能量的onset检测"""
        # 计算每帧能量
        num_frames = len(audio) // self.hop_length
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.hop_length
            frame = audio[start:end]
            energy[i] = np.sum(frame ** 2)
        
        # 对数能量
        energy = np.log1p(energy)
        
        # 能量差分
        diff = np.diff(energy)
        
        # 归一化
        if diff.max() > 0:
            diff = diff / diff.max()
        
        # 峰值检测
        peaks = []
        for i in range(1, len(diff) - 1):
            if diff[i] > self.threshold and diff[i] > diff[i-1] and diff[i] > diff[i+1]:
                peaks.append(i + 1)  # +1因为diff比energy少一个
        
        # 转换为时间
        times = [p * self.hop_length / self.sample_rate for p in peaks]
        
        return times
