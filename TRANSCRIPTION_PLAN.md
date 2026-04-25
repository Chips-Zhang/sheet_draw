# Sheet Draw - 扒谱功能扩展规划

## 目标功能
从流行歌曲音频中自动分析音调唱名，划分节奏，输出曲谱。

## 需要扩展的能力

### 1. 🎵 音符分割 (Onset Detection)
- **作用**: 检测每个音符开始的时间点
- **算法**: 
  - 频域差异法 (Spectral Flux)
  - 相位偏差法 (Phase Deviation)
  - 复合 onset 检测
- **难点**: 处理连音、滑音、音符边界模糊

### 2. 🎼 音高追踪 (Pitch Tracking)
- **作用**: 连续跟踪音高变化
- **当前**: 已有基础，需增强
- **改进**:
  - 处理滑音 (glissando)
  - 处理颤音 (vibrato)
  - 去除伪音/噪音
  - 音高平滑处理

### 3. 🥁 节奏分析 (Rhythm Analysis)
- **作用**: 检测节拍和速度
- **能力**:
  - BPM 检测 (每分钟节拍数)
  - 节拍位置检测 (Beat Tracking)
  - 强拍/弱拍识别
  - 拍号识别 (4/4, 3/4等)
- **算法**: 
  -  onset 自相关
  - 动态规划节拍追踪

### 4. 🎹 调性检测 (Key Detection)
- **作用**: 确定歌曲的调性
- **输出**: 
  - 调性 (C大调、A小调等)
  - 音阶
  - 临时变音标记
- **算法**: 
  - Krumhansl-Schmuckler 调性检测
  - 音级分布分析

### 5. ⏱️ 音符量化 (Note Quantization)
- **作用**: 将检测到的音高对齐到标准音符
- **处理**:
  - 音准偏差修正
  - 时值量化 (四分音符、八分音符等)
  - 休止符识别

### 6. 🎸 多声部分离 (可选)
- **作用**: 分离人声和伴奏
- **算法**: 
  - Spleeter (深度学习)
  - 频谱减法
- **用途**: 扒谱通常只需要主旋律

### 7. 📄 曲谱生成
- **输出格式**:
  - MIDI 文件 (可导入DAW)
  - MusicXML (专业制谱软件)
  - 简谱 (文本格式)
  - 五线谱 (图像/PDF)

## 技术实现建议

### 推荐库
```python
# 核心音频处理
librosa          # 音频分析、onset检测、节拍追踪
madmom           # 音乐信息检索 (beat tracking)
aubio            # 音高和onset检测

# 调性检测
music21          # 音乐理论、调性分析

# 曲谱生成
mido             # MIDI生成
musicxml         # MusicXML生成

# 人声分离 (可选)
spleeter         # 音轨分离
```

### 处理流程

```
输入音频
   ↓
[预处理] - 降噪、归一化、人声分离(可选)
   ↓
[Onset检测] - 找到所有音符开始点
   ↓
[Beat检测] - 确定节拍位置和BPM
   ↓
[音高检测] - 对每个音符检测音高
   ↓
[调性分析] - 确定调性，计算唱名
   ↓
[量化] - 音高和时值标准化
   ↓
[曲谱生成] - 输出MIDI/简谱/MusicXML
```

## 实现优先级

### Phase 1: 核心功能 (MVP)
1. ✅ 音高检测 (已有)
2. 🔄 Onset检测
3. 🔄 Beat检测 (BPM)
4. 📝 简谱文本输出

### Phase 2: 增强功能
5. 调性检测
6. 时值量化
7. MIDI输出

### Phase 3: 高级功能
8. 人声分离
9. MusicXML输出
10. 五线谱图像生成

## 相似开源项目

| 项目 | 链接 | 特点 |
|------|------|------|
| **Basic Pitch** | spotify/basic-pitch | Spotify的轻量级扒谱工具 |
| **piano-transcription** | bytedance/piano-transcription | 字节跳动的钢琴转录 |
| ** omnizart** | Music-and-Culture-Technology-Lab/omnizart | 多乐器音乐转录 |
| **AutoPiano** | AutoPiano | 自动钢琴转录 |
| ** anthemscore** | 商业软件 | 流行的自动扒谱软件 |

## 快速开始建议

对于流行歌曲扒谱，推荐先集成 **Basic Pitch** (Spotify开源)：
- 轻量级，基于深度学习
- 直接输出音高和onset
- 效果比传统算法好

```bash
pip install basic-pitch
```

然后在其基础上添加节奏分析和曲谱生成功能。
