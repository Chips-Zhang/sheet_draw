"""
调性检测 (Key Detection) - 确定音乐的调性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


class KeyDetector:
    """调性检测器"""
    
    # Krumhansl-Schmuckler 关键轮廓 (Major)
    MAJOR_PROFILE = np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ])
    
    # Krumhansl-Schmuckler 关键轮廓 (Minor)
    MINOR_PROFILE = np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self):
        pass
    
    def detect(self, notes: List[str]) -> Dict:
        """
        从音符列表检测调性
        
        Args:
            notes: 音符名称列表 (如 ['C4', 'E4', 'G4', ...])
            
        Returns:
            调性信息字典
        """
        # 提取音级 (忽略八度)
        pitch_classes = []
        for note in notes:
            pitch_class = self._note_to_pitch_class(note)
            if pitch_class is not None:
                pitch_classes.append(pitch_class)
        
        if not pitch_classes:
            return {"error": "No valid notes"}
        
        # 计算音级分布
        distribution = self._compute_distribution(pitch_classes)
        
        # 使用Krumhansl-Schmuckler算法检测调性
        key_result = self._krumhansl_schmuckler(distribution)
        
        return {
            "key": key_result["key"],
            "mode": key_result["mode"],  # major/minor
            "confidence": key_result["confidence"],
            "distribution": distribution.tolist(),
            "tonic": key_result["tonic"]
        }
    
    def detect_from_frequencies(self, frequencies: List[float]) -> Dict:
        """
        从频率列表检测调性
        
        Args:
            frequencies: 频率列表 (Hz)
            
        Returns:
            调性信息字典
        """
        # 转换为音符
        notes = []
        for freq in frequencies:
            note = self._freq_to_note_name(freq)
            if note:
                notes.append(note)
        
        return self.detect(notes)
    
    def _note_to_pitch_class(self, note: str) -> Optional[int]:
        """将音符名称转换为音级 (0-11)"""
        if not note or len(note) < 2:
            return None
        
        # 提取音符名 (去掉八度)
        if '#' in note:
            note_name = note[:2]
        else:
            note_name = note[0]
        
        try:
            return self.NOTE_NAMES.index(note_name)
        except ValueError:
            return None
    
    def _freq_to_note_name(self, frequency: float) -> Optional[str]:
        """频率转音符名称"""
        if frequency <= 0:
            return None
        
        A4 = 440.0
        semitones = 12 * np.log2(frequency / A4)
        midi_note = int(round(69 + semitones))
        
        note_idx = (midi_note - 12) % 12
        octave = (midi_note - 12) // 12
        
        return f"{self.NOTE_NAMES[note_idx]}{octave}"
    
    def _compute_distribution(self, pitch_classes: List[int]) -> np.ndarray:
        """计算音级分布"""
        distribution = np.zeros(12)
        
        for pc in pitch_classes:
            distribution[pc] += 1
        
        # 归一化
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        
        return distribution
    
    def _krumhansl_schmuckler(self, distribution: np.ndarray) -> Dict:
        """
        Krumhansl-Schmuckler 调性检测算法
        
        通过计算输入分布与标准关键轮廓的相关性来确定调性
        """
        if np.ptp(distribution) == 0:
            return {"key": "C major", "tonic": "C", "mode": "major",
                    "correlation": 0., "confidence": 0.}
        best_correlation = -1
        best_key = None
        best_mode = None
        
        # 测试所有大调
        for tonic in range(12):
            # 旋转轮廓以匹配当前tonic
            rotated_profile = np.roll(self.MAJOR_PROFILE, tonic)
            
            # 计算相关性
            correlation = np.corrcoef(distribution, rotated_profile)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = tonic
                best_mode = "major"
        
        # 测试所有小调
        for tonic in range(12):
            rotated_profile = np.roll(self.MINOR_PROFILE, tonic)
            
            correlation = np.corrcoef(distribution, rotated_profile)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = tonic
                best_mode = "minor"
        
        # 转换音级为音符名
        tonic_name = self.NOTE_NAMES[best_key]
        key_name = f"{tonic_name} {best_mode}"
        
        # 计算置信度 (映射到0-1)
        confidence = (best_correlation + 1) / 2
        
        return {
            "key": key_name,
            "tonic": tonic_name,
            "mode": best_mode,
            "correlation": best_correlation,
            "confidence": round(confidence, 3)
        }
    
    def get_scale_notes(self, key: str) -> List[str]:
        """
        获取调性的音阶音符
        
        Args:
            key: 调性名称 (如 "C major", "A minor")
            
        Returns:
            音阶音符列表
        """
        parts = key.split()
        if len(parts) != 2:
            return []
        
        tonic = parts[0]
        mode = parts[1]
        
        # 获取tonic的音级
        try:
            tonic_pc = self.NOTE_NAMES.index(tonic)
        except ValueError:
            return []
        
        # 大调音程: 0, 2, 4, 5, 7, 9, 11
        # 小调音程: 0, 2, 3, 5, 7, 8, 10
        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]
        else:  # minor
            intervals = [0, 2, 3, 5, 7, 8, 10]
        
        scale = []
        for interval in intervals:
            note_idx = (tonic_pc + interval) % 12
            scale.append(self.NOTE_NAMES[note_idx])
        
        return scale
    
    def get_solfege(self, note: str, key: str) -> str:
        """
        获取音符在调性中的唱名
        
        Args:
            note: 音符名称
            key: 调性
            
        Returns:
            唱名 (Do, Re, Mi, ...)
        """
        solfege_names = ['Do', 'Di', 'Re', 'Ri', 'Mi', 'Fa', 'Fi', 'Sol', 'Si', 'La', 'Li', 'Ti']
        
        note_pc = self._note_to_pitch_class(note)
        if note_pc is None:
            return "?"
        
        # 获取调性的tonic
        parts = key.split()
        if len(parts) != 2:
            return "?"
        
        tonic = parts[0]
        try:
            tonic_pc = self.NOTE_NAMES.index(tonic)
        except ValueError:
            return "?"
        
        # 计算相对位置
        relative = (note_pc - tonic_pc) % 12
        
        return solfege_names[relative]
    
    def get_jianpu_number(self, note: str, key: str) -> int:
        """
        获取简谱数字 (1-7)
        
        Args:
            note: 音符名称
            key: 调性
            
        Returns:
            简谱数字 (1-7, 0表示不在音阶内)
        """
        note_pc = self._note_to_pitch_class(note)
        if note_pc is None:
            return 0
        
        parts = key.split()
        if len(parts) != 2:
            return 0
        
        tonic = parts[0]
        mode = parts[1]
        
        try:
            tonic_pc = self.NOTE_NAMES.index(tonic)
        except ValueError:
            return 0
        
        # 计算相对位置
        relative = (note_pc - tonic_pc) % 12
        
        # 大调音程映射到简谱
        if mode == "major":
            jianpu_map = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
        else:  # minor
            # 小调以La(6)为1
            jianpu_map = {0: 6, 2: 7, 3: 1, 5: 2, 7: 3, 8: 4, 10: 5}
        
        return jianpu_map.get(relative, 0)
