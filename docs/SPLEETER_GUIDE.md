# Spleeter 人声分离使用指南

[Spleeter](https://github.com/deezer/spleeter) 是 Deezer 开源的深度学习人声分离工具，
分离效果比传统算法好很多。

## 安装方法

### 方法1: 使用 sheet-draw 命令行安装 (推荐)

```bash
# 检查是否已安装
sheet-draw install-spleeter --check

# 安装 CPU 版本
sheet-draw install-spleeter

# 安装 GPU 版本 (需要 NVIDIA GPU)
sheet-draw install-spleeter --gpu

# 测试是否安装成功
sheet-draw install-spleeter --test
```

### 方法2: 使用安装脚本

```bash
# 在项目目录下
python scripts/install_spleeter.py install

# 安装 GPU 版本
python scripts/install_spleeter.py install-gpu

# 检查安装状态
python scripts/install_spleeter.py check

# 测试安装
python scripts/install_spleeter.py test
```

### 方法3: 手动安装

```bash
pip install spleeter

# GPU 版本 (需要 NVIDIA GPU)
pip install tensorflow-gpu
pip install spleeter
```

### 方法4: 安装时一起安装 (使用 pip extras)

```bash
# 安装时包含 spleeter
pip install "sheet-draw[spleeter]"

# 或者安装所有可选依赖
pip install "sheet-draw[full]"
```

## 使用方法

### 命令行使用

```bash
# 使用 Spleeter 分离人声后扒谱
sheet-draw transcribe song.wav --separate-vocal --separation-method spleeter

# 生成口琴 TAB 谱
sheet-draw transcribe song.wav \
    --separate-vocal \
    --separation-method spleeter \
    --harmonica \
    -o tab
```

### Python 代码使用

```python
from sheet_draw import SpleeterWrapper

# 创建包装器
wrapper = SpleeterWrapper()

# 检查是否可用
if wrapper.available:
    # 分离人声
    vocal = wrapper.extract_vocal(audio, sample_rate=44100)
    
    # 或者同时获取所有音轨
    stems = wrapper.separate(audio, sample_rate=44100)
    vocals = stems['vocals']
    accompaniment = stems['accompaniment']
else:
    print("Spleeter 未安装，使用备用方法")
    from sheet_draw import separate_vocal
    vocal = separate_vocal(audio)
```

## 分离效果对比

| 方法 | 质量 | 速度 | 适用场景 |
|------|------|------|----------|
| HPSS | 中等 | 快 | 快速预览 |
| 频谱减法 | 中等 | 快 | 简单分离 |
| **Spleeter** | **优秀** | 中等 | **正式使用** |

## 注意事项

1. **首次使用会下载模型** (约 1.5GB)
   - 首次运行时需要下载深度学习模型
   - 下载时间取决于网络速度
   - 模型会缓存在本地，后续使用不需要重新下载

2. **GPU 加速**
   - GPU 版本比 CPU 版本快 5-10 倍
   - 需要 NVIDIA GPU 和 CUDA 支持
   - 如果没有 GPU，CPU 版本也可以使用

3. **内存需求**
   - Spleeter 需要约 4GB 内存
   - 处理长音频时可能需要更多内存

4. **分离模式**
   - `2stems`: 人声 + 伴奏 (默认)
   - `4stems`: 人声 + 鼓 + 贝斯 + 其他
   - `5stems`: 人声 + 鼓 + 贝斯 + 钢琴 + 其他

## 故障排除

### 安装失败

```bash
# 确保 pip 是最新版本
pip install --upgrade pip

# 使用管理员权限 (Windows)
# 或 sudo (Linux/Mac)
sudo pip install spleeter
```

### 模型下载失败

```bash
# 手动设置模型路径
export SPLEETER_MODEL_PATH=/path/to/models

# 或者重新安装
pip uninstall spleeter
pip install spleeter
```

### 内存不足

```bash
# 分段处理音频
# 或使用更小的 batch size
```

## 相关链接

- [Spleeter GitHub](https://github.com/deezer/spleeter)
- [Spleeter 文档](https://github.com/deezer/spleeter/wiki)
- [Deezer Research](https://research.deezer.com/)
