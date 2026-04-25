"""
示例脚本：演示如何使用 sheet_draw 库
"""

import numpy as np
from sheet_draw import PitchDetector, ToneAnalyzer
from sheet_draw.audio_utils import (
    generate_sine_wave,
    generate_note,
    save_wav,
    frequency_to_note,
    note_to_frequency
)


def example_1_basic_detection():
    """示例1: 基本音高检测"""
    print("=" * 50)
    print("示例1: 基本音高检测")
    print("=" * 50)
    
    # 生成一个440Hz的正弦波 (A4)
    audio = generate_sine_wave(frequency=440.0, duration=1.0)
    
    # 创建检测器
    detector = PitchDetector()
    
    # 检测音高
    result = detector.detect(audio)
    
    print(f"生成频率: 440.0 Hz (A4)")
    print(f"检测结果: {result['frequency']:.2f} Hz")
    print(f"音符: {result['note']}")
    print(f"唱名: {result['tone']} (简谱)")
    print()


def example_2_note_generation():
    """示例2: 生成音符并检测"""
    print("=" * 50)
    print("示例2: 生成音符并检测")
    print("=" * 50)
    
    detector = PitchDetector()
    
    # 测试不同音符
    test_notes = ["C4", "E4", "G4", "A4", "C5"]
    
    for note in test_notes:
        # 生成音符
        audio = generate_note(note, duration=0.5)
        
        # 检测
        result = detector.detect(audio)
        
        expected_freq = note_to_frequency(note)
        detected_freq = result['frequency']
        error = abs(detected_freq - expected_freq)
        
        print(f"{note}: 期望={expected_freq:.2f}Hz, "
              f"检测={detected_freq:.2f}Hz, "
              f"误差={error:.2f}Hz")
    print()


def example_3_different_algorithms():
    """示例3: 对比不同算法"""
    print("=" * 50)
    print("示例3: 对比不同算法")
    print("=" * 50)
    
    # 生成测试音频
    audio = generate_note("G3", duration=1.0)
    expected = note_to_frequency("G3")
    
    print(f"期望频率: {expected:.2f} Hz (G3)\n")
    
    for method in ["yin", "autocorr", "fft"]:
        detector = PitchDetector(method=method)
        result = detector.detect(audio)
        
        error = abs(result['frequency'] - expected)
        print(f"{method.upper():10s}: {result['frequency']:.2f} Hz "
              f"(误差: {error:.2f} Hz)")
    print()


def example_4_continuous_analysis():
    """示例4: 连续音调分析"""
    print("=" * 50)
    print("示例4: 连续音调分析")
    print("=" * 50)
    
    # 生成一个音高变化的音频
    t = np.linspace(0, 2, 88200, False)
    
    # 前1秒是C4，后1秒是G4
    audio = np.zeros_like(t)
    audio[:44100] = 0.5 * np.sin(2 * np.pi * 261.63 * t[:44100])
    audio[44100:] = 0.5 * np.sin(2 * np.pi * 392.00 * t[44100:])
    
    # 连续检测
    detector = PitchDetector()
    pitches = detector.detect_continuous(audio, hop_size=2048)
    
    print(f"检测到 {len(pitches)} 个音高帧")
    print("\n前5帧结果:")
    for i, p in enumerate(pitches[:5]):
        print(f"  t={p['time']:.3f}s: {p['frequency']:.2f} Hz ({p['note']})")
    print("  ...")
    print("\n后5帧结果:")
    for p in pitches[-5:]:
        print(f"  t={p['time']:.3f}s: {p['frequency']:.2f} Hz ({p['note']})")
    print()


def example_5_tone_analysis():
    """示例5: 音调特征分析"""
    print("=" * 50)
    print("示例5: 音调特征分析")
    print("=" * 50)
    
    # 生成一个稳定的E4音符
    audio = generate_note("E4", duration=2.0)
    
    # 添加少量噪声模拟真实录音
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # 分析
    analyzer = ToneAnalyzer()
    result = analyzer.analyze(audio)
    
    print(f"主要音高: {result['mean_frequency']} Hz")
    print(f"音符: {result['note']}")
    print(f"简谱唱名: {result['tone']}")
    print(f"稳定性: {result['stability']*100:.1f}%")
    print(f"频率标准差: {result['std_frequency']} Hz")
    print()


def example_6_save_and_detect():
    """示例6: 保存和加载音频文件"""
    print("=" * 50)
    print("示例6: 保存和加载音频文件")
    print("=" * 50)
    
    # 生成音频
    audio = generate_note("D4", duration=1.0)
    
    # 保存为WAV文件
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    save_wav(audio, temp_path)
    print(f"已保存到: {temp_path}")
    
    # 从文件检测
    detector = PitchDetector()
    result = detector.detect_from_file(temp_path)
    
    print(f"检测结果: {result['frequency']:.2f} Hz ({result['note']})")
    
    # 清理
    os.remove(temp_path)
    print(f"已删除临时文件")
    print()


def example_7_command_line():
    """示例7: 命令行使用说明"""
    print("=" * 50)
    print("示例7: 命令行使用")
    print("=" * 50)
    
    print("安装后，可以使用命令行工具:")
    print()
    print("  # 分析音频文件")
    print("  note-get audio.wav")
    print()
    print("  # 显示详细信息")
    print("  note-get audio.wav -v")
    print()
    print("  # 连续分析整个音频")
    print("  note-get audio.wav --continuous")
    print()
    print("  # 使用不同的算法")
    print("  note-get audio.wav -m fft")
    print()


if __name__ == "__main__":
    print("\n🎵 Note Get - 使用示例\n")
    
    example_1_basic_detection()
    example_2_note_generation()
    example_3_different_algorithms()
    example_4_continuous_analysis()
    example_5_tone_analysis()
    example_6_save_and_detect()
    example_7_command_line()
    
    print("=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)
