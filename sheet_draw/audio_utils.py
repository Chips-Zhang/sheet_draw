"""
音频工具函数
"""

import numpy as np
from typing import Tuple, Optional


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    生成正弦波音频（用于测试）
    
    Args:
        frequency: 频率 (Hz)
        duration: 持续时间 (秒)
        sample_rate: 采样率
        amplitude: 振幅 (0.0-1.0)
        
    Returns:
        音频数据
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def generate_note(
    note_name: str,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    生成指定音符的音频
    
    Args:
        note_name: 音符名称 (如 "A4", "C5", "F#3")
        duration: 持续时间
        sample_rate: 采样率
        amplitude: 振幅
        
    Returns:
        音频数据
    """
    # 解析音符名称
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 提取音符和八度
    if '#' in note_name:
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])
    
    # 计算MIDI编号
    note_idx = note_names.index(note)
    midi_note = (octave + 1) * 12 + note_idx
    
    # MIDI转频率: f = 440 * 2^((n-69)/12)
    frequency = 440.0 * (2 ** ((midi_note - 69) / 12))
    
    return generate_sine_wave(frequency, duration, sample_rate, amplitude)


def save_wav(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int = 44100
):
    """
    保存音频为WAV文件
    
    Args:
        audio: 音频数据
        filepath: 文件路径
        sample_rate: 采样率
    """
    import wave
    import struct
    
    # 转换为16位整数
    audio = (audio * 32767).astype(np.int16)
    
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())


def load_wav(filepath: str) -> Tuple[np.ndarray, int]:
    """
    加载WAV文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        (音频数据, 采样率)
    """
    import wave
    import struct
    
    with wave.open(filepath, 'r') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        
        raw_data = wav.readframes(n_frames)
        
        if sample_width == 2:
            fmt = f"{n_frames * n_channels}h"
            audio = np.array(struct.unpack(fmt, raw_data))
        elif sample_width == 4:
            fmt = f"{n_frames * n_channels}i"
            audio = np.array(struct.unpack(fmt, raw_data))
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # 转换为浮点数
        audio = audio.astype(np.float64) / 32768.0
        
        # 如果是立体声，转换为单声道
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        return audio, sample_rate


def frequency_to_note(frequency: float) -> str:
    """
    频率转音符名称
    
    Args:
        frequency: 频率 (Hz)
        
    Returns:
        音符名称
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # A4 = 440Hz
    A4 = 440.0
    semitones = 12 * np.log2(frequency / A4)
    midi_note = int(round(69 + semitones))
    
    note_idx = (midi_note - 12) % 12
    octave = (midi_note - 12) // 12
    
    return f"{note_names[note_idx]}{octave}"


def note_to_frequency(note_name: str) -> float:
    """
    音符名称转频率
    
    Args:
        note_name: 音符名称 (如 "A4", "C5")
        
    Returns:
        频率 (Hz)
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if '#' in note_name:
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])
    
    note_idx = note_names.index(note)
    midi_note = (octave + 1) * 12 + note_idx
    
    return 440.0 * (2 ** ((midi_note - 69) / 12))
