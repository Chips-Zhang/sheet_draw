"""
扒谱示例 - 演示 sheet_draw 的完整扒谱功能
"""

import numpy as np
from sheet_draw import (
    MusicTranscriber,
    TranscriptionConfig,
    OnsetDetector,
    BeatTracker,
    KeyDetector,
    Note
)
from sheet_draw.audio_utils import generate_note, save_wav


def example_1_simple_transcription():
    """示例1: 简单旋律扒谱"""
    print("=" * 60)
    print("示例1: 简单旋律扒谱")
    print("=" * 60)
    
    # 生成一个简单的C大调旋律: C D E F G
    sample_rate = 44100
    melody = []
    notes = ["C4", "D4", "E4", "F4", "G4"]
    duration = 0.5  # 每个音符0.5秒
    
    for note_name in notes:
        note_audio = generate_note(note_name, duration, sample_rate)
        melody.extend(note_audio)
    
    melody = np.array(melody)
    
    # 保存测试音频
    save_wav(melody, "test_melody.wav")
    print("已生成测试音频: test_melody.wav")
    print(f"旋律: {' '.join(notes)}\n")
    
    # 扒谱
    config = TranscriptionConfig(sample_rate=sample_rate)
    transcriber = MusicTranscriber(config)
    result = transcriber.transcribe(melody)
    
    # 输出结果
    print(f"检测到的调性: {result.key}")
    print(f"检测到的BPM: {result.bpm}")
    print(f"检测到的音符数: {len(result.notes)}")
    print("\n检测到的音符:")
    for i, note in enumerate(result.notes[:10]):  # 只显示前10个
        print(f"  {i+1}. {note.pitch} ({note.frequency:.2f}Hz) "
              f"@ {note.start_time:.3f}s, 持续{note.duration:.3f}s")
    
    print("\n生成的简谱:")
    print(result.to_jianpu())
    print()


def example_2_onset_detection():
    """示例2: Onset检测"""
    print("=" * 60)
    print("示例2: Onset检测 (音符分割)")
    print("=" * 60)
    
    # 生成带节奏的音频
    sample_rate = 44100
    
    # 创建节奏型: 四分音符 + 八分音符
    audio_parts = []
    
    # 强拍音符
    audio_parts.extend(generate_note("C4", 0.5, sample_rate, amplitude=0.8))
    # 弱拍音符
    audio_parts.extend(generate_note("E4", 0.25, sample_rate, amplitude=0.5))
    audio_parts.extend(generate_note("G4", 0.25, sample_rate, amplitude=0.5))
    
    audio = np.array(audio_parts)
    
    # Onset检测
    detector = OnsetDetector(sample_rate=sample_rate)
    onsets = detector.detect(audio)
    
    print(f"检测到的音符开始点: {len(onsets)} 个")
    for i, onset in enumerate(onsets):
        print(f"  音符 {i+1}: {onset:.3f} 秒")
    print()


def example_3_beat_tracking():
    """示例3: 节拍追踪"""
    print("=" * 60)
    print("示例3: 节拍追踪 (BPM检测)")
    print("=" * 60)
    
    # 生成4/4拍节奏
    sample_rate = 44100
    bpm_target = 120
    beat_duration = 60 / bpm_target
    
    # 生成4小节
    audio_parts = []
    for bar in range(4):
        for beat in range(4):
            # 强拍
            if beat == 0:
                note = generate_note("C3", 0.1, sample_rate, amplitude=0.8)
            else:
                note = generate_note("C3", 0.05, sample_rate, amplitude=0.4)
            
            # 填充到节拍长度
            beat_samples = int(beat_duration * sample_rate)
            note_padded = np.zeros(beat_samples)
            note_padded[:len(note)] = note
            audio_parts.extend(note_padded)
    
    audio = np.array(audio_parts)
    
    # 节拍追踪
    tracker = BeatTracker(sample_rate=sample_rate)
    result = tracker.analyze(audio)
    
    print(f"目标BPM: {bpm_target}")
    print(f"检测BPM: {result['bpm']}")
    print(f"检测到的节拍数: {len(result['beats'])}")
    print()


def example_4_key_detection():
    """示例4: 调性检测"""
    print("=" * 60)
    print("示例4: 调性检测")
    print("=" * 60)
    
    # C大调音阶
    c_major_scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    
    # A小调音阶
    a_minor_scale = ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"]
    
    detector = KeyDetector()
    
    # 检测C大调
    print("C大调音阶:", c_major_scale)
    result_major = detector.detect(c_major_scale)
    print(f"  检测结果: {result_major['key']} (置信度: {result_major['confidence']})")
    
    # 检测A小调
    print("\nA小调音阶:", a_minor_scale)
    result_minor = detector.detect(a_minor_scale)
    print(f"  检测结果: {result_minor['key']} (置信度: {result_minor['confidence']})")
    
    # 获取唱名
    print("\nC大调中的音符唱名:")
    for note in ["C4", "D4", "E4", "F4", "G4"]:
        solfege = detector.get_solfege(note, "C major")
        jianpu = detector.get_jianpu_number(note, "C major")
        print(f"  {note} -> {solfege} (简谱: {jianpu})")
    print()


def example_5_full_transcription():
    """示例5: 完整扒谱流程演示"""
    print("=" * 60)
    print("示例5: 完整扒谱流程")
    print("=" * 60)
    
    # 生成一个更复杂的旋律 (小星星)
    sample_rate = 44100
    # 小星星简谱: 1 1 5 5 6 6 5 - 4 4 3 3 2 2 1
    melody_notes = [
        ("C4", 0.5), ("C4", 0.5), ("G4", 0.5), ("G4", 0.5),
        ("A4", 0.5), ("A4", 0.5), ("G4", 1.0),
        ("F4", 0.5), ("F4", 0.5), ("E4", 0.5), ("E4", 0.5),
        ("D4", 0.5), ("D4", 0.5), ("C4", 1.0)
    ]
    
    audio_parts = []
    for note_name, duration in melody_notes:
        note_audio = generate_note(note_name, duration, sample_rate)
        audio_parts.extend(note_audio)
    
    audio = np.array(audio_parts)
    save_wav(audio, "twinkle.wav")
    
    print("已生成《小星星》测试音频: twinkle.wav")
    print(f"包含 {len(melody_notes)} 个音符\n")
    
    # 扒谱
    config = TranscriptionConfig(sample_rate=sample_rate)
    transcriber = MusicTranscriber(config)
    result = transcriber.transcribe(audio)
    
    print("扒谱结果:")
    print(f"  调性: {result.key}")
    print(f"  速度: {result.bpm} BPM")
    print(f"  音符数量: {len(result.notes)}")
    
    print("\n音符序列:")
    for i, note in enumerate(result.notes):
        # 计算简谱数字
        kd = KeyDetector()
        jianpu_num = kd.get_jianpu_number(note.pitch, result.key)
        print(f"  {i+1:2d}. {note.pitch:5s} ({jianpu_num}) @ {note.start_time:.2f}s")
    
    print("\n生成的简谱:")
    print(result.to_jianpu())
    
    # 保存为MIDI
    result.to_midi("twinkle.mid")
    print("MIDI文件已保存: twinkle.mid\n")


def example_6_command_line():
    """示例6: 命令行使用"""
    print("=" * 60)
    print("示例6: 命令行使用")
    print("=" * 60)
    
    print("""
安装后，可以使用命令行工具进行扒谱:

  # 扒谱并显示简谱
  note-get transcribe song.wav

  # 输出为MIDI文件
  note-get transcribe song.wav -o midi

  # 输出为JSON格式
  note-get transcribe song.wav -o json --output-file song.json

  # 只检测BPM
  note-get transcribe song.wav --bpm

  # 指定BPM范围
  note-get transcribe song.wav --tempo-min 80 --tempo-max 140
""")


if __name__ == "__main__":
    print("\n🎼 Note Get - 扒谱功能演示\n")
    
    try:
        example_1_simple_transcription()
        example_2_onset_detection()
        example_3_beat_tracking()
        example_4_key_detection()
        example_5_full_transcription()
        example_6_command_line()
        
        print("=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
