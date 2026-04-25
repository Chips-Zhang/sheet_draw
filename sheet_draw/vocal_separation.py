"""
人声分离 (Vocal Separation) - 从伴奏中提取人声或从混音中分离人声
"""

import numpy as np
from typing import Tuple, Optional
import scipy.signal
from scipy.ndimage import median_filter


class VocalSeparator:
    """人声分离器"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        method: str = "spectral_subtraction"
    ):
        """
        初始化人声分离器
        
        Args:
            sample_rate: 采样率
            n_fft: FFT大小
            hop_length: 帧移
            method: 分离方法 ('spectral_subtraction', 'hpss', 'median_filtering')
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.method = method
    
    def separate(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离人声和伴奏
        
        Args:
            audio: 输入音频 (混音)
            
        Returns:
            (vocal, accompaniment) - 人声和伴奏
        """
        if self.method == "spectral_subtraction":
            return self._spectral_subtraction(audio)
        elif self.method == "hpss":
            return self._hpss_separation(audio)
        elif self.method == "median_filtering":
            return self._median_filtering(audio)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def extract_vocal(self, audio: np.ndarray) -> np.ndarray:
        """只提取人声"""
        vocal, _ = self.separate(audio)
        return vocal
    
    def extract_accompaniment(self, audio: np.ndarray) -> np.ndarray:
        """只提取伴奏"""
        _, accompaniment = self.separate(audio)
        return accompaniment
    
    def _spectral_subtraction(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        频谱减法 - 基于人声和伴奏在频谱上的差异
        
        原理: 人声通常集中在200Hz-4kHz，伴奏有更宽的频谱
        """
        # STFT
        D = self._stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # 创建人声掩码
        # 人声主要集中在中频 (200Hz - 4000Hz)
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        
        vocal_mask = np.zeros_like(magnitude)
        accompaniment_mask = np.ones_like(magnitude)
        
        # 人声频段
        vocal_freq_min = 200
        vocal_freq_max = 4000
        
        vocal_freq_indices = (freqs >= vocal_freq_min) & (freqs <= vocal_freq_max)
        vocal_mask[vocal_freq_indices, :] = 1
        accompaniment_mask[vocal_freq_indices, :] = 0.3  # 伴奏在这些频段较弱
        
        # 应用掩码
        vocal_mag = magnitude * vocal_mask
        accompaniment_mag = magnitude * accompaniment_mask
        
        # ISTFT
        vocal = self._istft(vocal_mag * np.exp(1j * phase))
        accompaniment = self._istft(accompaniment_mag * np.exp(1j * phase))
        
        return vocal, accompaniment
    
    def _hpss_separation(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        HPSS (Harmonic-Percussive Source Separation)
        
        将音频分离为谐波成分(人声/旋律)和打击乐成分(鼓点)
        """
        # STFT
        D = self._stft(audio)
        magnitude = np.abs(D)
        
        # 中值滤波分离
        # 沿时间轴滤波 -> 提取谐波成分
        harmonic = median_filter(magnitude, size=(1, 15), mode='reflect')
        
        # 沿频率轴滤波 -> 提取打击乐成分
        percussive = median_filter(magnitude, size=(15, 1), mode='reflect')
        
        # 创建软掩码
        mask_harmonic = harmonic / (harmonic + percussive + 1e-10)
        mask_percussive = percussive / (harmonic + percussive + 1e-10)
        
        # 应用掩码
        phase = np.angle(D)
        harmonic_audio = self._istft(magnitude * mask_harmonic * np.exp(1j * phase))
        percussive_audio = self._istft(magnitude * mask_percussive * np.exp(1j * phase))
        
        # 人声主要是谐波成分，但包含部分打击乐
        # 简单起见，将谐波成分视为人声
        vocal = harmonic_audio * 0.8 + percussive_audio * 0.2
        accompaniment = percussive_audio * 0.8 + harmonic_audio * 0.2
        
        return vocal, accompaniment
    
    def _median_filtering(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于中值滤波的人声分离
        
        使用更激进的滤波来分离前景(人声)和背景(伴奏)
        """
        # STFT
        D = self._stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # 背景估计 - 使用中值滤波
        background = median_filter(magnitude, size=(15, 5), mode='reflect')
        
        # 前景掩码
        mask = np.maximum(0, magnitude - background * 0.8)
        mask = mask / (magnitude + 1e-10)
        
        # 平滑掩码
        mask = median_filter(mask, size=(3, 3), mode='reflect')
        
        # 应用掩码
        vocal_mag = magnitude * mask
        accompaniment_mag = magnitude * (1 - mask)
        
        vocal = self._istft(vocal_mag * np.exp(1j * phase))
        accompaniment = self._istft(accompaniment_mag * np.exp(1j * phase))
        
        return vocal, accompaniment
    
    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """短时傅里叶变换"""
        return scipy.signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary='constant'
        )[2]
    
    def _istft(self, spectrogram: np.ndarray) -> np.ndarray:
        """逆短时傅里叶变换"""
        _, audio = scipy.signal.istft(
            spectrogram,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=True
        )
        return audio


class SpleeterWrapper:
    """
    Spleeter 包装器
    
    如果安装了 Spleeter，使用深度学习模型进行高质量人声分离
    """
    
    def __init__(self, model: str = "2stems"):
        """
        初始化Spleeter包装器
        
        Args:
            model: 模型类型 ('2stems' - 人声+伴奏, '4stems' - 人声+鼓+贝斯+其他)
        """
        self.model = model
        self.separator = None
        self._check_spleeter()
    
    def _check_spleeter(self):
        """检查是否安装了Spleeter"""
        try:
            from spleeter.separator import Separator
            self.separator = Separator(f'spleeter:{self.model}')
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: Spleeter not installed. Using fallback methods.")
            print("Install Spleeter for better quality: pip install spleeter")
    
    def separate(self, audio: np.ndarray, sample_rate: int = 44100) -> dict:
        """
        使用Spleeter分离音轨
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            分离后的音轨字典
        """
        if not self.available:
            # 使用备用方法
            separator = VocalSeparator(sample_rate=sample_rate)
            vocal, accompaniment = separator.separate(audio)
            return {
                'vocals': vocal,
                'accompaniment': accompaniment
            }
        
        # 使用Spleeter
        # 需要将音频转换为Spleeter支持的格式
        # 这里简化处理，实际使用时可能需要保存为临时文件
        import tempfile
        import soundfile as sf
        from spleeter.separator import Separator
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_input = f.name
        
        sf.write(temp_input, audio, sample_rate)
        
        # 创建输出目录
        import tempfile
        output_dir = tempfile.mkdtemp()
        
        # 分离
        separator = Separator(f'spleeter:{self.model}')
        separator.separate_to_file(temp_input, output_dir)
        
        # 读取结果
        import os
        import glob
        
        result = {}
        for stem_file in glob.glob(os.path.join(output_dir, '*.wav')):
            stem_name = os.path.basename(stem_file).replace('.wav', '')
            stem_audio, _ = sf.read(stem_file)
            result[stem_name] = stem_audio
        
        # 清理临时文件
        import shutil
        os.remove(temp_input)
        shutil.rmtree(output_dir)
        
        return result
    
    def extract_vocal(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """提取人声"""
        result = self.separate(audio, sample_rate)
        return result.get('vocals', audio)


def preprocess_for_transcription(
    audio: np.ndarray,
    sample_rate: int = 44100,
    method: str = "auto"
) -> np.ndarray:
    """
    预处理音频用于扒谱
    
    自动选择最佳方法分离人声
    
    Args:
        audio: 输入音频
        sample_rate: 采样率
        method: 分离方法 ('auto', 'spleeter', 'hpss', 'spectral')
        
    Returns:
        预处理后的音频 (主要是人声)
    """
    if method == "auto":
        # 检查是否有Spleeter
        try:
            import spleeter
            method = "spleeter"
        except ImportError:
            method = "hpss"
    
    if method == "spleeter":
        wrapper = SpleeterWrapper()
        return wrapper.extract_vocal(audio, sample_rate)
    else:
        separator = VocalSeparator(sample_rate=sample_rate, method=method)
        return separator.extract_vocal(audio)


# 便捷函数
def separate_vocal(
    audio: np.ndarray,
    sample_rate: int = 44100,
    use_spleeter: bool = False
) -> np.ndarray:
    """
    快速人声分离
    
    Args:
        audio: 输入音频
        sample_rate: 采样率
        use_spleeter: 是否使用Spleeter (需要单独安装)
        
    Returns:
        人声部分
    """
    if use_spleeter:
        wrapper = SpleeterWrapper()
        return wrapper.extract_vocal(audio, sample_rate)
    else:
        separator = VocalSeparator(sample_rate=sample_rate)
        return separator.extract_vocal(audio)
