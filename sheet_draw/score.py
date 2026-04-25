"""
曲谱生成 (Score Generation) - 生成简谱、MIDI等格式
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Note:
    """音符数据类"""
    pitch: str           # 音符名称 (如 "C4", "A#5")
    frequency: float     # 频率 (Hz)
    start_time: float    # 开始时间 (秒)
    duration: float      # 持续时间 (秒)
    velocity: float = 1.0  # 力度 (0-1)
    
    # 可选的计算属性
    @property
    def octave(self) -> int:
        """获取八度"""
        return int(''.join(filter(str.isdigit, self.pitch)))
    
    @property
    def note_name(self) -> str:
        """获取音符名 (不含八度)"""
        return ''.join(filter(lambda c: not c.isdigit(), self.pitch))


class JianpuGenerator:
    """简谱生成器"""
    
    def __init__(self, key: str = "C major"):
        """
        初始化简谱生成器
        
        Args:
            key: 调性 (如 "C major", "A minor")
        """
        self.key = key
        self.notes = []
    
    def add_note(self, note: Note, jianpu_number: int):
        """添加音符"""
        self.notes.append({
            "note": note,
            "jianpu": jianpu_number
        })
    
    def generate(self, bpm: float = 120) -> str:
        """
        生成简谱文本
        
        Args:
            bpm: 速度 (BPM)
            
        Returns:
            简谱文本
        """
        lines = []
        
        # 标题
        lines.append("=" * 50)
        lines.append(f"简谱 | 调性: {self.key} | 速度: {bpm} BPM")
        lines.append("=" * 50)
        lines.append("")
        
        # 生成简谱行
        current_line = ""
        current_time = 0
        measure_duration = 60 / bpm * 4  # 4/4拍，每小节时长
        
        for i, note_data in enumerate(self.notes):
            note = note_data["note"]
            jianpu = note_data["jianpu"]
            
            # 检查是否需要添加小节线
            if note.start_time >= current_time + measure_duration:
                current_line += " | "
                current_time += measure_duration
            
            # 添加简谱数字
            octave_marker = ""
            if note.octave > 4:
                octave_marker = "'" * (note.octave - 4)  # 高音点
            elif note.octave < 4:
                octave_marker = "," * (4 - note.octave)  # 低音点
            
            # 时值标记
            duration_marker = self._get_duration_marker(note.duration, bpm)
            
            current_line += f"{jianpu}{octave_marker}{duration_marker} "
            
            # 每行最多16个音符
            if len(current_line) > 60:
                lines.append(current_line)
                current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def _get_duration_marker(self, duration: float, bpm: float) -> str:
        """获取时值标记"""
        beat_duration = 60 / bpm
        
        # 相对于四分音符的倍数
        ratio = duration / beat_duration
        
        if ratio >= 3.5:
            return " - - -"  # 全音符
        elif ratio >= 1.75:
            return " -"      # 二分音符
        elif ratio >= 0.875:
            return ""        # 四分音符
        elif ratio >= 0.4375:
            return "_"       # 八分音符
        elif ratio >= 0.21875:
            return "__"      # 十六分音符
        else:
            return "___"     # 三十二分音符
    
    def generate_with_beats(self, notes: List[Note], beats: List[float], bpm: float) -> str:
        """
        根据节拍位置生成简谱
        
        Args:
            notes: 音符列表
            beats: 节拍时间列表
            bpm: 速度
            
        Returns:
            简谱文本
        """
        # 计算每个音符相对于节拍的时值
        lines = []
        lines.append(f"1={self.key.split()[0]} {bpm} BPM")
        lines.append("")
        
        # 简化输出
        jianpu_line = ""
        for note in notes:
            # 简谱数字
            from .key_detection import KeyDetector
            kd = KeyDetector()
            jianpu_num = kd.get_jianpu_number(note.pitch, self.key)
            
            if jianpu_num == 0:
                jianpu_num = "0"  # 休止符或不在音阶内
            
            # 八度标记
            if note.octave > 4:
                marker = "'" * (note.octave - 4)
            elif note.octave < 4:
                marker = "," * (4 - note.octave)
            else:
                marker = ""
            
            jianpu_line += f"{jianpu_num}{marker} "
        
        lines.append(jianpu_line)
        
        return "\n".join(lines)


class MidiGenerator:
    """MIDI文件生成器"""
    
    def __init__(self, tempo: int = 120):
        """
        初始化MIDI生成器
        
        Args:
            tempo: 速度 (BPM)
        """
        self.tempo = tempo
        self.notes = []
    
    def add_note(self, note: Note):
        """添加音符"""
        self.notes.append(note)
    
    def generate(self, filepath: str):
        """
        生成MIDI文件
        
        Args:
            filepath: 输出文件路径
        """
        try:
            import mido
            from mido import Message, MidiFile, MidiTrack, MetaMessage
        except ImportError:
            print("mido library is required for MIDI generation")
            print("Install: pip install mido")
            return
        
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 添加 tempo meta message
        tempo_microseconds = int(60_000_000 / self.tempo)
        track.append(MetaMessage('set_tempo', tempo=tempo_microseconds))
        
        # 按时间排序音符
        sorted_notes = sorted(self.notes, key=lambda n: n.start_time)
        
        # 转换时间到ticks
        ticks_per_beat = mid.ticks_per_beat
        
        current_time = 0
        for note in sorted_notes:
            # 计算MIDI note number
            midi_note = self._freq_to_midi(note.frequency)
            
            # 计算delta time (从上一个事件到现在的时间差)
            note_start_ticks = int(note.start_time * ticks_per_beat * self.tempo / 60)
            note_duration_ticks = int(note.duration * ticks_per_beat * self.tempo / 60)
            
            delta_ticks = note_start_ticks - current_time
            
            # Note on
            velocity = int(note.velocity * 127)
            track.append(Message('note_on', note=midi_note, velocity=velocity, time=delta_ticks))
            
            # Note off
            track.append(Message('note_off', note=midi_note, velocity=0, time=note_duration_ticks))
            
            current_time = note_start_ticks + note_duration_ticks
        
        # 保存文件
        mid.save(filepath)
        print(f"MIDI file saved to: {filepath}")
    
    def _freq_to_midi(self, frequency: float) -> int:
        """频率转MIDI note number"""
        if frequency <= 0:
            return 0
        return int(round(69 + 12 * np.log2(frequency / 440.0)))


class TranscriptionResult:
    """扒谱结果容器"""
    
    def __init__(
        self,
        notes: List[Note],
        key: str,
        bpm: float,
        beats: List[float]
    ):
        self.notes = notes
        self.key = key
        self.bpm = bpm
        self.beats = beats
    
    def to_jianpu(self) -> str:
        """输出简谱"""
        generator = JianpuGenerator(self.key)
        
        # 计算每个音符的简谱数字
        from .key_detection import KeyDetector
        kd = KeyDetector()
        
        for note in self.notes:
            jianpu_num = kd.get_jianpu_number(note.pitch, self.key)
            generator.add_note(note, jianpu_num)
        
        return generator.generate(self.bpm)
    
    def to_midi(self, filepath: str):
        """输出MIDI"""
        generator = MidiGenerator(self.bpm)
        
        for note in self.notes:
            generator.add_note(note)
        
        generator.generate(filepath)
    
    def to_harmonica_tab(
        self,
        harmonica_type: str = "chromatic",
        harmonica_key: str = "C"
    ) -> str:
        """
        输出口琴TAB谱
        
        Args:
            harmonica_type: 口琴类型 ('chromatic', 'diatonic')
            harmonica_key: 口琴调性
            
        Returns:
            TAB谱文本
        """
        from .harmonica import (
            HarmonicaScoreGenerator,
            create_chromatic_harmonica,
            create_diatonic_harmonica
        )
        
        if harmonica_type == "chromatic":
            harp = create_chromatic_harmonica(harmonica_key)
        else:
            harp = create_diatonic_harmonica(harmonica_key)
        
        generator = HarmonicaScoreGenerator(harp)
        return generator.generate_tab(self.notes)
    
    def to_harmonica_jianpu(
        self,
        harmonica_type: str = "chromatic",
        harmonica_key: str = "C"
    ) -> str:
        """
        输出带TAB的口琴简谱
        
        Args:
            harmonica_type: 口琴类型 ('chromatic', 'diatonic')
            harmonica_key: 口琴调性
            
        Returns:
            曲谱文本
        """
        from .harmonica import (
            HarmonicaScoreGenerator,
            create_chromatic_harmonica,
            create_diatonic_harmonica
        )
        
        if harmonica_type == "chromatic":
            harp = create_chromatic_harmonica(harmonica_key)
        else:
            harp = create_diatonic_harmonica(harmonica_key)
        
        generator = HarmonicaScoreGenerator(harp)
        return generator.generate_jianpu_with_tab(self.notes, self.key)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "key": self.key,
            "bpm": self.bpm,
            "beats": self.beats,
            "notes": [
                {
                    "pitch": n.pitch,
                    "frequency": n.frequency,
                    "start_time": n.start_time,
                    "duration": n.duration,
                    "velocity": n.velocity
                }
                for n in self.notes
            ]
        }


def quantize_notes(
    notes: List[Note],
    bpm: float,
    quantization: str = "1/16"
) -> List[Note]:
    """
    量化音符时值
    
    Args:
        notes: 原始音符列表
        bpm: 速度
        quantization: 量化精度 ("1/4", "1/8", "1/16")
        
    Returns:
        量化后的音符列表
    """
    beat_duration = 60 / bpm
    
    # 根据量化精度计算最小单位
    if quantization == "1/4":
        grid_size = beat_duration
    elif quantization == "1/8":
        grid_size = beat_duration / 2
    else:  # 1/16
        grid_size = beat_duration / 4
    
    quantized = []
    for note in notes:
        # 量化开始时间
        quantized_start = round(note.start_time / grid_size) * grid_size
        
        # 量化持续时间
        quantized_duration = round(note.duration / grid_size) * grid_size
        
        # 确保最小持续时间
        if quantized_duration < grid_size / 2:
            quantized_duration = grid_size
        
        quantized_note = Note(
            pitch=note.pitch,
            frequency=note.frequency,
            start_time=quantized_start,
            duration=quantized_duration,
            velocity=note.velocity
        )
        quantized.append(quantized_note)
    
    return quantized
