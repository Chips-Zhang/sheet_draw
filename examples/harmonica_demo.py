"""
口琴扒谱示例 - 针对口琴优化的扒谱演示
"""

import numpy as np
from sheet_draw import (
    MusicTranscriber,
    TranscriptionConfig,
    create_chromatic_harmonica,
    create_diatonic_harmonica,
    HarmonicaScoreGenerator,
    VocalSeparator
)
from sheet_draw.audio_utils import generate_note, save_wav


def example_1_chromatic_harmonica():
    """示例1: 半音阶口琴曲谱生成"""
    print("=" * 60)
    print("示例1: 半音阶口琴曲谱")
    print("=" * 60)
    
    # 创建一个适合口琴的旋律
    sample_rate = 44100
    melody = []
    
    # C大调音阶 (适合C调半音阶口琴)
    notes = [
        ("C4", 0.5), ("D4", 0.5), ("E4", 0.5), ("F4", 0.5),
        ("G4", 0.5), ("A4", 0.5), ("B4", 0.5), ("C5", 1.0)
    ]
    
    for note_name, duration in notes:
        note_audio = generate_note(note_name, duration, sample_rate)
        melody.extend(note_audio)
    
    melody = np.array(melody)
    save_wav(melody, "chromatic_demo.wav")
    
    print("已生成测试音频: chromatic_demo.wav")
    print(f"旋律: {' '.join([n[0] for n in notes])}\n")
    
    # 扒谱
    config = TranscriptionConfig(sample_rate=sample_rate)
    transcriber = MusicTranscriber(config)
    result = transcriber.transcribe(melody)
    
    print(f"检测调性: {result.key}")
    print(f"检测BPM: {result.bpm}")
    
    # 生成半音阶口琴TAB谱
    print("\n" + "=" * 60)
    print("C调半音阶口琴 TAB谱")
    print("=" * 60)
    harp = create_chromatic_harmonica("C", holes=12)
    generator = HarmonicaScoreGenerator(harp)
    print(generator.generate_tab(result.notes))
    
    # 生成带简谱的TAB
    print("\n" + "=" * 60)
    print("C调半音阶口琴 简谱+TAB")
    print("=" * 60)
    print(generator.generate_jianpu_with_tab(result.notes, result.key))
    print()


def example_2_diatonic_harmonica():
    """示例2: 十孔蓝调口琴曲谱"""
    print("=" * 60)
    print("示例2: 十孔蓝调口琴曲谱")
    print("=" * 60)
    
    # 创建一个蓝调音阶
    sample_rate = 44100
    
    # C调蓝调音阶 (可以演示压音)
    # 1孔吹C, 2孔吸G, 3孔吸B (可以压音到Bb, A)
    blues_scale = [
        ("C4", 0.5),   # 1孔吹
        ("E4", 0.5),   # 2孔吹
        ("F4", 0.5),   # 2孔吸压音
        ("G4", 0.5),   # 2孔吸
        ("A4", 0.5),   # 3孔吸压音
        ("B4", 0.5),   # 3孔吸
        ("C5", 0.5),   # 4孔吹
    ]
    
    melody = []
    for note_name, duration in blues_scale:
        note_audio = generate_note(note_name, duration, sample_rate)
        melody.extend(note_audio)
    
    melody = np.array(melody)
    save_wav(melody, "diatonic_demo.wav")
    
    print("已生成测试音频: diatonic_demo.wav")
    print(f"蓝调音阶: {' '.join([n[0] for n in blues_scale])}\n")
    
    # 扒谱
    config = TranscriptionConfig(sample_rate=sample_rate)
    transcriber = MusicTranscriber(config)
    result = transcriber.transcribe(melody)
    
    # 生成十孔口琴TAB谱
    print("=" * 60)
    print("C调十孔蓝调口琴 TAB谱")
    print("=" * 60)
    harp = create_diatonic_harmonica("C")
    generator = HarmonicaScoreGenerator(harp)
    print(generator.generate_tab(result.notes))
    
    # 检查不可演奏的音符
    unplayable = generator.get_unplayable_notes(result.notes)
    if unplayable:
        print(f"\n注意: 以下音符在C调口琴上较难演奏:")
        print(f"  {', '.join(set(unplayable))}")
        print("  建议转调到其他调性或使用半音阶口琴")
    print()


def example_3_vocal_separation():
    """示例3: 人声分离演示"""
    print("=" * 60)
    print("示例3: 人声分离")
    print("=" * 60)
    
    # 创建一个模拟的人声+伴奏音频
    sample_rate = 44100
    duration = 2.0
    
    # 人声部分 (旋律)
    vocal = []
    for note in ["C4", "E4", "G4", "C5"]:
        note_audio = generate_note(note, 0.5, sample_rate, amplitude=0.7)
        vocal.extend(note_audio)
    vocal = np.array(vocal)
    
    # 伴奏部分 (低音节奏)
    accompaniment = []
    for note in ["C3", "G3", "C3", "G3"]:
        note_audio = generate_note(note, 0.5, sample_rate, amplitude=0.4)
        accompaniment.extend(note_audio)
    accompaniment = np.array(accompaniment)
    
    # 混合
    mixed = vocal + accompaniment
    mixed = mixed / np.max(np.abs(mixed))
    
    save_wav(mixed, "mixed_demo.wav")
    save_wav(vocal, "vocal_only.wav")
    save_wav(accompaniment, "accompaniment_only.wav")
    
    print("已生成测试音频:")
    print("  - mixed_demo.wav (人声+伴奏混合)")
    print("  - vocal_only.wav (仅人声)")
    print("  - accompaniment_only.wav (仅伴奏)\n")
    
    # 使用人声分离
    print("进行人声分离...")
    separator = VocalSeparator(sample_rate=sample_rate, method="hpss")
    separated_vocal, separated_accompaniment = separator.separate(mixed)
    
    save_wav(separated_vocal, "separated_vocal.wav")
    print("分离完成: separated_vocal.wav\n")
    
    # 对分离后的人声扒谱
    print("对分离后的人声进行扒谱...")
    config = TranscriptionConfig(
        sample_rate=sample_rate,
        enable_vocal_separation=False  # 已经分离过了
    )
    transcriber = MusicTranscriber(config)
    result = transcriber.transcribe(separated_vocal)
    
    print(f"检测结果: {result.key}, {result.bpm} BPM")
    print("音符:", [n.pitch for n in result.notes])
    print()


def example_4_full_harmonica_pipeline():
    """示例4: 完整的口琴扒谱流程"""
    print("=" * 60)
    print("示例4: 完整口琴扒谱流程")
    print("=" * 60)
    
    # 创建一段《小星星》旋律
    sample_rate = 44100
    
    melody_notes = [
        ("C4", 0.5), ("C4", 0.5), ("G4", 0.5), ("G4", 0.5),
        ("A4", 0.5), ("A4", 0.5), ("G4", 1.0),
        ("F4", 0.5), ("F4", 0.5), ("E4", 0.5), ("E4", 0.5),
        ("D4", 0.5), ("D4", 0.5), ("C4", 1.0)
    ]
    
    melody = []
    for note_name, duration in melody_notes:
        note_audio = generate_note(note_name, duration, sample_rate)
        melody.extend(note_audio)
    
    melody = np.array(melody)
    save_wav(melody, "twinkle.wav")
    
    print("《小星星》旋律")
    print(f"音符: {' '.join([n[0] for n in melody_notes])}\n")
    
    # 使用口琴优化的配置
    config = TranscriptionConfig(
        sample_rate=sample_rate,
        harmonica_optimization=True,
        harmonica_type="chromatic",
        harmonica_key="C"
    )
    
    transcriber = MusicTranscriber(config)
    result = transcriber.transcribe(melody)
    
    # 输出各种格式的曲谱
    print("=" * 60)
    print("1. 标准简谱")
    print("=" * 60)
    print(result.to_jianpu())
    
    print("\n" + "=" * 60)
    print("2. 半音阶口琴TAB谱 (C调)")
    print("=" * 60)
    print(result.to_harmonica_tab(harmonica_type="chromatic", harmonica_key="C"))
    
    print("\n" + "=" * 60)
    print("3. 十孔口琴TAB谱 (C调)")
    print("=" * 60)
    print(result.to_harmonica_tab(harmonica_type="diatonic", harmonica_key="C"))
    
    # 保存MIDI
    result.to_midi("twinkle.mid")
    print("\nMIDI文件已保存: twinkle.mid")
    print()


def example_5_command_line_examples():
    """示例5: 命令行使用示例"""
    print("=" * 60)
    print("示例5: 命令行使用")
    print("=" * 60)
    
    print("""
口琴扒谱命令行用法:

1. 基本扒谱 (生成简谱):
   note-get transcribe song.wav

2. 人声分离后扒谱:
   note-get transcribe song.wav --separate-vocal

3. 使用特定分离方法:
   note-get transcribe song.wav --separate-vocal --separation-method hpss
   note-get transcribe song.wav --separate-vocal --separation-method spleeter

4. 口琴优化 (半音阶口琴 C调):
   note-get transcribe song.wav --harmonica

5. 十孔蓝调口琴 (G调):
   note-get transcribe song.wav --harmonica --harmonica-type diatonic --harmonica-key G

6. 生成口琴TAB谱:
   note-get transcribe song.wav -o tab
   note-get transcribe song.wav -o tab --harmonica-type diatonic --harmonica-key C

7. 完整配置 (人声分离 + 口琴优化):
   note-get transcribe song.wav \\
       --separate-vocal \\
       --harmonica \\
       --harmonica-type chromatic \\
       --harmonica-key C \\
       -o tab
""")


if __name__ == "__main__":
    print("\n🎵 Note Get - 口琴扒谱演示\n")
    
    try:
        example_1_chromatic_harmonica()
        example_2_diatonic_harmonica()
        example_3_vocal_separation()
        example_4_full_harmonica_pipeline()
        example_5_command_line_examples()
        
        print("=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
