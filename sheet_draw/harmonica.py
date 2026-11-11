"""
口琴曲谱生成器 (Harmonica Score Generator)

针对口琴演奏优化的曲谱生成，支持：
- 半音阶口琴 (Chromatic)
- 十孔蓝调口琴 (Diatonic/Blues Harp)
- 复音口琴 (Tremolo)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .score import Note
from .audio_utils import frequency_to_note, note_to_frequency


@dataclass
class HarmonicaNote:
    """口琴音符数据"""
    note: Note
    hole: int              # 孔位
    action: str            # 'blow'(吹) 或 'draw'(吸)
    bend: Optional[str]    # 压音标记 (如 'half', 'full')
    button: bool           # 半音阶口琴推键 (True=按下)
    
    @property
    def tab_notation(self) -> str:
        """生成TAB谱标记"""
        parts = []
        
        # 孔位
        parts.append(str(self.hole))
        
        # 吹吸
        if self.action == 'blow':
            parts.append('')  # 吹通常不标记
        else:
            parts.append('-')  # 吸用'-'标记
        
        # 推键 (半音阶)
        if self.button:
            parts.append('↑')
        
        # 压音 (蓝调)
        if self.bend:
            if self.bend == 'half':
                parts.append("'")
            elif self.bend == 'three_half':
                parts.append("'''")
            elif self.bend == 'full':
                parts.append("''")
        
        return ''.join(parts)


class Harmonica:
    """口琴基类"""
    
    def __init__(self, key: str = "C"):
        """
        初始化口琴
        
        Args:
            key: 调性 (如 'C', 'G', 'A')
        """
        # Standard Richter keys run from G3 through F#4 at hole one.
        offsets = {"G": -5, "Ab": -4, "A": -3, "Bb": -2, "B": -1,
                   "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6}
        aliases = {"G#": "Ab", "A#": "Bb", "C#": "Db", "D#": "Eb", "Gb": "F#"}
        key = aliases.get(key, key)
        if key not in offsets:
            raise ValueError(f"未知口琴调性: {key}")
        self.key = key
        self.offset = offsets[key]

    def transpose_layout_note(self, name):
        return frequency_to_note(note_to_frequency(name) * 2 ** (self.offset / 12))
    
    def get_range(self) -> Tuple[float, float]:
        """返回音域范围 (Hz)"""
        raise NotImplementedError
    
    def note_to_hole(self, note: str) -> Optional[HarmonicaNote]:
        """将音符转换为口琴孔位"""
        raise NotImplementedError
    
    def is_playable(self, note: str) -> bool:
        """检查音符是否可以在该口琴上演奏"""
        return self.note_to_hole(note) is not None


class ChromaticHarmonica(Harmonica):
    """
    半音阶口琴 (如 Hohner Chrometta, Suzuki SCX)
    
    12孔半音阶口琴音域: C4 - D7
    标准调音 (C调)
    """
    
    # 12孔半音阶口琴标准布局
    # hole: (吹音, 吸音, 吹音+推键, 吸音+推键)
    LAYOUT_12 = {}
    for octave in range(4, 7):
        first = 1 + 4 * (octave - 4)
        LAYOUT_12[first] = (f"C{octave}", f"D{octave}", f"C#{octave}", f"D#{octave}")
        LAYOUT_12[first+1] = (f"E{octave}", f"F{octave}", f"F{octave}", f"F#{octave}")
        LAYOUT_12[first+2] = (f"G{octave}", f"A{octave}", f"G#{octave}", f"A#{octave}")
        LAYOUT_12[first+3] = (f"C{octave+1}", f"B{octave}", f"C#{octave+1}", f"C{octave+1}")
    LAYOUT_12[12] = ("C7", "B6", "C#7", "D7")
    LAYOUT_16 = {
        1: ("C3", "D3", "C#3", "D#3"), 2: ("E3", "F3", "F3", "F#3"),
        3: ("G3", "A3", "G#3", "A#3"), 4: ("C4", "B3", "C#4", "C4"),
        **{hole + 4: notes for hole, notes in LAYOUT_12.items()},
    }

    def __init__(self, key: str = "C", holes: int = 12):
        """
        初始化半音阶口琴
        
        Args:
            key: 调性
            holes: 孔数 (12或16)
        """
        super().__init__(key)
        self.holes = holes
        layout = self.LAYOUT_16 if holes >= 16 else self.LAYOUT_12
        self.layout = {h: tuple(self.transpose_layout_note(n) for n in row)
                       for h, row in layout.items()}
        
        # 计算音域
        self._min_freq = self._note_to_freq("C3" if holes >= 16 else "C4")
        self._max_freq = self._note_to_freq("D7")
    
    def get_range(self) -> Tuple[float, float]:
        return (self._min_freq * 2 ** (self.offset / 12), self._max_freq * 2 ** (self.offset / 12))
    
    def note_to_hole(self, note: str) -> Optional[HarmonicaNote]:
        """
        将音符转换为口琴孔位信息
        
        Args:
            note: 音符名称 (如 "C5", "F#6")
            
        Returns:
            HarmonicaNote对象 或 None
        """
        # 标准化音符名称
        note = self._normalize_note(note)
        
        for hole, (blow, draw, blow_btn, draw_btn) in self.layout.items():
            if note == blow:
                return HarmonicaNote(
                    note=None, hole=hole, action='blow',
                    bend=None, button=False
                )
            elif note == draw:
                return HarmonicaNote(
                    note=None, hole=hole, action='draw',
                    bend=None, button=False
                )
            elif note == blow_btn:
                return HarmonicaNote(
                    note=None, hole=hole, action='blow',
                    bend=None, button=True
                )
            elif note == draw_btn:
                return HarmonicaNote(
                    note=None, hole=hole, action='draw',
                    bend=None, button=True
                )
        
        return None
    
    def _normalize_note(self, note: str) -> str:
        """标准化音符名称"""
        # 移除空格，统一格式
        note = note.strip().replace(' ', '')
        
        # 将降号转换为升号 (如 Bb = A#)
        enharmonic_map = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#',
            'Ab': 'G#', 'Bb': 'A#'
        }
        
        for flat, sharp in enharmonic_map.items():
            note = note.replace(flat, sharp)
        
        return note
    
    def _note_to_freq(self, note: str) -> float:
        """音符转频率"""
        if note is None:
            return 0
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 解析
        if '#' in note:
            name = note[:2]
            octave = int(note[2:])
        else:
            name = note[0]
            octave = int(note[1:])
        
        note_idx = note_names.index(name)
        midi_num = (octave + 1) * 12 + note_idx
        
        return 440.0 * (2 ** ((midi_num - 69) / 12))


class DiatonicHarmonica(Harmonica):
    """
    十孔蓝调口琴 (Richter调音)
    
    标准 C 调十孔蓝调口琴音域: C4 - C7
    """
    
    # Richter调音布局
    # hole: (吹音, 吸音, 压音音高，从半音起)
    RICHTER_LAYOUT = {
        1:  ("C4",  "D4",  ["C#4"]),     # 1孔: 吸气压低半音
        2:  ("E4",  "G4",  ["F#4", "F4"]),     # 2孔: 吹E, 吸G, 压音F#, F
        3:  ("G4",  "B4",  ["A#4", "A4", "G#4"]), # 3孔: 吹G, 吸B, 压音A#, A, G#
        4:  ("C5",  "D5",  ["C#5"]),           # 4孔: 吹C, 吸D, 压音C#
        5:  ("E5",  "F5",  []),                # 5孔: 吹E, 吸F, 无压音
        6:  ("G5",  "A5",  ["G#5"]),           # 6孔: 吹G, 吸A, 压音G#
        7:  ("C6",  "B5",  []),                # 7孔: 吹C, 吸B, 无压音
        8:  ("E6",  "D6",  ["D#6"]),     # 8孔: 吹E, 吸D, 吹气压音
        9:  ("G6",  "F6",  ["F#6"]),           # 9孔: 吹G, 吸F, 吹气压音
        10: ("C7",  "A6",  ["B6", "A#6"]),     # 10孔: 吹C, 吸A, 吹气压音
    }
    
    def __init__(self, key: str = "C"):
        super().__init__(key)
        self.layout = {h: (self.transpose_layout_note(b), self.transpose_layout_note(d),
                           [self.transpose_layout_note(n) for n in bends])
                       for h, (b, d, bends) in self.RICHTER_LAYOUT.items()}
    
    def get_range(self) -> Tuple[float, float]:
        return (self._note_to_freq(self.layout[1][0]), self._note_to_freq(self.layout[10][0]))
    
    def note_to_hole(self, note: str) -> Optional[HarmonicaNote]:
        """将音符转换为孔位"""
        note = self._normalize_note(note)
        
        for hole, (blow, draw, bends) in self.layout.items():
            if note == blow:
                return HarmonicaNote(
                    note=None, hole=hole, action='blow',
                    bend=None, button=False
                )
            elif note == draw:
                return HarmonicaNote(
                    note=None, hole=hole, action='draw',
                    bend=None, button=False
                )
            elif note in bends:
                bend_idx = bends.index(note)
                bend_type = ("half", "full", "three_half")[bend_idx]
                return HarmonicaNote(
                    note=None, hole=hole, action='draw' if hole <= 6 else 'blow',
                    bend=bend_type, button=False
                )
        
        return None
    
    def _normalize_note(self, note: str) -> str:
        """标准化音符"""
        note = note.strip().replace(' ', '')
        
        enharmonic_map = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#',
            'Ab': 'G#', 'Bb': 'A#'
        }
        
        for flat, sharp in enharmonic_map.items():
            note = note.replace(flat, sharp)
        
        return note
    
    def _note_to_freq(self, note: str) -> float:
        """音符转频率"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        if '#' in note:
            name = note[:2]
            octave = int(note[2:])
        else:
            name = note[0]
            octave = int(note[1:])
        
        note_idx = note_names.index(name)
        midi_num = (octave + 1) * 12 + note_idx
        
        return 440.0 * (2 ** ((midi_num - 69) / 12))


class PaddyHarmonica(DiatonicHarmonica):
    """Paddy Richter raises the third blow reed by a whole tone."""
    RICHTER_LAYOUT = {**DiatonicHarmonica.RICHTER_LAYOUT, 3: ("A4", "B4", ["A#4"])}


def make_harp(instrument, key):
    classes = {"paddy": PaddyHarmonica, "chromatic": ChromaticHarmonica, "diatonic": DiatonicHarmonica}
    if instrument not in classes:
        raise ValueError("未知口琴类型")
    return classes[instrument](key)


class HarmonicaScoreGenerator:
    """口琴曲谱生成器"""
    
    def __init__(self, harmonica: Harmonica):
        """
        初始化生成器
        
        Args:
            harmonica: 口琴对象
        """
        self.harmonica = harmonica
    
    def generate_tab(self, notes: List[Note]) -> str:
        """
        生成口琴TAB谱
        
        Args:
            notes: 音符列表
            
        Returns:
            TAB谱文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"口琴TAB谱 | {self._get_harmonica_name()}")
        lines.append("=" * 60)
        lines.append("")
        
        # 图例
        lines.append("图例:")
        if isinstance(self.harmonica, ChromaticHarmonica):
            lines.append("  数字 = 吹 (如 '4')")
            lines.append("  数字- = 吸 (如 '4-')")
            lines.append("  数字↑ = 推键吹 (如 '4↑')")
            lines.append("  数字-↑ = 推键吸 (如 '4-↑')")
        else:  # Diatonic
            lines.append("  数字 = 吹 (如 '4')")
            lines.append("  数字- = 吸 (如 '4-')")
            lines.append("  数字' = 半音压音 (如 '3-'')")
            lines.append("  数字'' = 全音压音 (如 '3-''')")
        lines.append("")
        
        # TAB谱
        tab_line = "TAB: "
        note_line = "音符: "
        
        playable_count = 0
        for note in notes[:50]:  # 限制显示前50个音符
            harp_note = self.harmonica.note_to_hole(note.pitch)
            
            if harp_note:
                tab_line += harp_note.tab_notation + " "
                note_line += note.pitch + " "
                playable_count += 1
            else:
                # 不可演奏的音符
                tab_line += "[X] "
                note_line += note.pitch + " "
        
        lines.append(tab_line)
        lines.append(note_line)
        lines.append("")
        lines.append(f"可演奏音符: {playable_count}/{len(notes)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def generate_jianpu_with_tab(self, notes: List[Note], key: str = "C major") -> str:
        """
        生成带TAB的简谱
        
        Args:
            notes: 音符列表
            key: 调性
            
        Returns:
            曲谱文本
        """
        from .key_detection import KeyDetector
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"口琴简谱 + TAB | {self._get_harmonica_name()}")
        lines.append(f"原曲调性: {key}")
        lines.append("=" * 60)
        lines.append("")
        
        kd = KeyDetector()
        
        # 生成两行: 简谱 + TAB
        jianpu_line = "简谱: "
        tab_line = "TAB:  "
        
        for note in notes[:50]:
            jianpu_num = kd.get_jianpu_number(note.pitch, key)
            harp_note = self.harmonica.note_to_hole(note.pitch)
            
            if jianpu_num == 0:
                jianpu_str = "0"
            else:
                # 八度标记
                if note.octave > 4:
                    jianpu_str = str(jianpu_num) + "'" * (note.octave - 4)
                elif note.octave < 4:
                    jianpu_str = str(jianpu_num) + "," * (4 - note.octave)
                else:
                    jianpu_str = str(jianpu_num)
            
            jianpu_line += jianpu_str.ljust(4)
            
            if harp_note:
                tab_line += harp_note.tab_notation.ljust(4)
            else:
                tab_line += "[X] ".ljust(4)
        
        lines.append(jianpu_line)
        lines.append(tab_line)
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_unplayable_notes(self, notes: List[Note]) -> List[str]:
        """获取不可演奏的音符列表"""
        unplayable = []
        for note in notes:
            if not self.harmonica.is_playable(note.pitch):
                unplayable.append(note.pitch)
        return list(set(unplayable))
    
    def transpose_to_key(self, notes: List[Note], target_key: str) -> List[Note]:
        """
        将音符转调到适合口琴演奏的调性
        
        Args:
            notes: 原音符列表
            target_key: 目标调性
            
        Returns:
            转调后的音符列表
        """
        # 简化的转调逻辑
        # 实际应用中需要考虑更多因素
        return notes
    
    def _get_harmonica_name(self) -> str:
        """获取口琴名称"""
        if isinstance(self.harmonica, ChromaticHarmonica):
            return f"{self.harmonica.key}调半音阶口琴 ({self.harmonica.holes}孔)"
        elif isinstance(self.harmonica, DiatonicHarmonica):
            return f"{self.harmonica.key}调十孔口琴"
        else:
            return "口琴"


# 便捷函数
def create_chromatic_harmonica(key: str = "C", holes: int = 12) -> ChromaticHarmonica:
    """创建半音阶口琴"""
    return ChromaticHarmonica(key, holes)


def create_diatonic_harmonica(key: str = "C") -> DiatonicHarmonica:
    """创建十孔蓝调口琴"""
    return DiatonicHarmonica(key)


def generate_harmonica_tab(
    notes: List[Note],
    harmonica_type: str = "chromatic",
    key: str = "C"
) -> str:
    """
    快速生成口琴TAB谱
    
    Args:
        notes: 音符列表
        harmonica_type: 口琴类型 ('chromatic', 'diatonic')
        key: 口琴调性
        
    Returns:
        TAB谱文本
    """
    if harmonica_type == "chromatic":
        harp = ChromaticHarmonica(key)
    else:
        harp = DiatonicHarmonica(key)
    
    generator = HarmonicaScoreGenerator(harp)
    return generator.generate_tab(notes)
