# Sheet Draw

一个用于识别音频中音高和音调的工具，支持完整的音乐扒谱功能，特别针对**口琴演奏**进行了优化。

## ✨ 功能特性

### 基础功能
- 🎵 **音高检测** - 检测音频中的基频 (Fundamental Frequency)
- 🎼 **音符转换** - 将频率转换为音符 (Note) 和音调 (Tone)
- 📊 **多种算法** - YIN算法、自相关法、FFT峰值法

### 扒谱功能 (v0.2 新增)
- 🎶 **音符分割** (Onset Detection) - 自动识别每个音符的开始时间
- 🥁 **节奏分析** (Beat Tracking) - 检测BPM和节拍位置
- 🎹 **调性检测** (Key Detection) - 识别歌曲调性和音阶
- 📝 **简谱生成** - 自动生成简谱文本
- 🎹 **MIDI导出** - 导出为标准MIDI文件

### 口琴优化 (v0.3 新增)
- 🎹 **半音阶口琴** - 支持12/16孔半音阶口琴TAB谱
- 🎵 **十孔蓝调口琴** - 支持Richter调音，含压音标记
- 🎼 **孔位标记** - 自动生成吹/吸/推键/压音标记
- 🔍 **可演奏检测** - 自动识别难以演奏的音符

### 人声分离 (v0.3 新增)
- 🎤 **人声提取** - 从伴奏中分离人声
- 🎸 **伴奏去除** - 多种分离算法 (HPSS, 频谱减法)
- 🤖 **Spleeter支持** - 可选的深度学习分离

## 安装

```bash
# 基础安装
pip install -e .

# 完整安装 (包含所有扒谱功能)
pip install -e ".[full]"
```

### 依赖项
- numpy - 数值计算
- scipy - 信号处理
- soundfile - 音频文件读写
- (可选) mido - MIDI文件生成
- (可选) matplotlib - 可视化

## 快速开始

### 音高检测

```python
from sheet_draw import PitchDetector

# 创建检测器
detector = PitchDetector()

# 分析音频文件
result = detector.detect_from_file("audio.wav")
print(f"音高: {result['frequency']:.2f} Hz")
print(f"音符: {result['note']}")
print(f"唱名: {result['tone']}")
```

### 音乐扒谱

```python
from sheet_draw import MusicTranscriber

# 创建扒谱器
transcriber = MusicTranscriber()

# 扒谱
result = transcriber.transcribe_file("song.wav")

# 输出简谱
print(result.to_jianpu())

# 导出MIDI
result.to_midi("output.mid")
```

### 人声分离

```python
from sheet_draw import VocalSeparator, separate_vocal

# 方法1: 使用分离器
separator = VocalSeparator(method="hpss")
vocal, accompaniment = separator.separate(audio)

# 方法2: 快速分离
vocal = separate_vocal(audio, sample_rate=44100)
```

### 口琴曲谱

```python
from sheet_draw import (
    MusicTranscriber,
    TranscriptionConfig,
    create_chromatic_harmonica,
    HarmonicaScoreGenerator
)

# 配置口琴优化
config = TranscriptionConfig(
    harmonica_optimization=True,
    harmonica_type="chromatic",  # 或 "diatonic"
    harmonica_key="C"
)

transcriber = MusicTranscriber(config)
result = transcriber.transcribe_file("song.wav")

# 生成口琴TAB谱
tab = result.to_harmonica_tab(
    harmonica_type="chromatic",
    harmonica_key="C"
)
print(tab)

# 或使用生成器直接生成
harp = create_chromatic_harmonica("C", holes=12)
generator = HarmonicaScoreGenerator(harp)
print(generator.generate_tab(result.notes))
```

## 命令行使用

### 音高检测

```bash
# 分析音频文件
sheet-draw pitch audio.wav

# 显示详细信息
sheet-draw pitch audio.wav -v

# 连续分析整个音频
sheet-draw pitch audio.wav --continuous
```

### 扒谱

```bash
# 扒谱并显示简谱
sheet-draw transcribe song.wav

# 导出为MIDI
sheet-draw transcribe song.wav -o midi

# 导出为JSON
sheet-draw transcribe song.wav -o json --output-file song.json

# 只检测BPM
sheet-draw transcribe song.wav --bpm
```

### 人声分离 + 扒谱

```bash
# 自动分离人声后扒谱 (使用 HPSS 算法)
sheet-draw transcribe song.wav --separate-vocal

# 指定分离方法
sheet-draw transcribe song.wav --separate-vocal --separation-method hpss
sheet-draw transcribe song.wav --separate-vocal --separation-method spectral

# 使用 Spleeter (深度学习, 需要安装)
sheet-draw install-spleeter             # 安装 Spleeter
sheet-draw transcribe song.wav --separate-vocal --separation-method spleeter

# Spleeter GPU 版本 (需要 NVIDIA GPU)
sheet-draw install-spleeter --gpu
```

### 口琴曲谱生成

```bash
# 生成半音阶口琴TAB谱 (C调)
sheet-draw transcribe song.wav -o tab --harmonica

# 生成十孔蓝调口琴TAB谱 (G调)
sheet-draw transcribe song.wav -o tab \\
    --harmonica-type diatonic \\
    --harmonica-key G

# 完整配置: 人声分离 + 口琴优化
sheet-draw transcribe song.wav \\
    --separate-vocal \\
    --harmonica \\
    --harmonica-type chromatic \\
    --harmonica-key C \\
    -o tab
```

## 扒谱实现原理

### 处理流程

```
输入音频
   ↓
[Onset检测] - 找到音符开始点
   ↓
[Beat检测] - 确定BPM和节拍
   ↓
[音高检测] - 检测每个音符的音高
   ↓
[调性分析] - 识别调性，计算唱名
   ↓
[量化] - 音高和时值标准化
   ↓
[曲谱生成] - 输出简谱/MIDI
```

### 算法说明

| 功能 | 算法 | 说明 |
|------|------|------|
| Onset检测 | Spectral Flux | 频谱通量法检测音符边界 |
| BPM检测 | 自相关分析 | 基于onset包络的自相关 |
| 音高检测 | YIN算法 | 经典音高检测算法 |
| 调性检测 | Krumhansl-Schmuckler | 标准调性检测算法 |
| 时值量化 | 网格对齐 | 按节拍网格量化音符 |

## 示例

### 生成测试音频并扒谱

```python
import numpy as np
from sheet_draw import MusicTranscriber
from sheet_draw.audio_utils import generate_note, save_wav

# 生成C大调音阶
sample_rate = 44100
melody = []
for note in ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]:
    audio = generate_note(note, 0.5, sample_rate)
    melody.extend(audio)

melody = np.array(melody)
save_wav(melody, "scale.wav")

# 扒谱
transcriber = MusicTranscriber()
result = transcriber.transcribe(melody)

print(result.to_jianpu())
```

## 项目结构

```
sheet_draw/
├── detector.py           # 音高检测核心
├── onset.py              # Onset检测
├── rhythm.py             # 节奏分析
├── key_detection.py      # 调性检测
├── score.py              # 曲谱生成
├── transcriber.py        # 完整扒谱器
├── vocal_separation.py   # 人声分离
├── harmonica.py          # 口琴曲谱生成
└── cli.py                # 命令行接口
```

## 相关开源项目

在构建本项目时，参考了以下优秀的开源实现：

| 项目 | 链接 | 说明 |
|------|------|------|
| **Basic Pitch** | spotify/basic-pitch | Spotify的深度学习扒谱工具 |
| **Librosa** | librosa/librosa | Python音频分析库 |
| **Aubio** | aubio/aubio | C语言音高检测库 |
| **Madmom** | CPJKU/madmom | 音乐信息检索库 |
| **Music21** | cuthbertLab/music21 | 音乐理论分析 |

## 局限性与改进方向

### 当前局限
1. **单音旋律** - 目前主要针对单音，和弦检测能力有限
2. **简单节奏** - 复杂节奏型(如切分音)检测可能不准确
3. **人声分离** - 需要预先分离人声，否则可能受伴奏影响

### 改进计划
- [ ] 多音高检测 (Polyphonic)
- [ ] 深度学习模型集成
- [ ] 人声分离预处理
- [ ] 更复杂的节奏模式识别
- [ ] 五线谱图像生成

## 许可证

MIT License

## 版本历史

### v0.3.1 (2024-04)
- ✨ 新增 Spleeter 一键安装功能
- ✨ 新增 `sheet-draw install-spleeter` 命令
- ✨ 支持 CPU 和 GPU 版本安装
- ✨ 添加安装检查和测试功能

### v0.3.0 (2024-04)
- ✨ 新增口琴曲谱生成功能
- ✨ 支持半音阶口琴 (12/16孔)
- ✨ 支持十孔蓝调口琴
- ✨ 新增吹/吸/推键/压音标记
- ✨ 新增人声分离功能
- ✨ 支持HPSS/频谱减法/Spleeter

### v0.2.0 (2024-04)
- ✨ 新增完整扒谱功能
- ✨ 新增Onset检测
- ✨ 新增Beat追踪
- ✨ 新增调性检测
- ✨ 新增简谱生成
- ✨ 新增MIDI导出

### v0.1.0 (2024-04)
- 🎉 初始版本
- 基础音高检测功能
- YIN/自相关/FFT算法
