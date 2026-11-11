# 转谱实现与验证

项目提供一条从音频到五线谱和吹吸谱的本地路径。安装基础依赖可用单音追踪和信号处理；安装可选依赖后，同一页面可使用神经网络识别及分离。

| 能力 | 已接入的路径 | 如何核对 |
|---|---|---|
| 多音识别 | 单音追踪、Basic Pitch 主旋律、可演奏双音、原始多音。双音在同向吹吸、按键状态一致的相邻孔中选伴音；原始模式展示额外声部，未知组合标问号。C、G、降 B 调独立映射。 | 对照各模式的逐曲起音、收音 F1、重叠音召回和多报。测试覆盖同一音高的替代孔位及错开收音。 |
| 模型选择 | 本地 ONNX 推理，保留原始音符和弯音序列。用固定开发曲目比较阈值与最短时长，再在分曲留出集上评估。 | 运行 benchmarks/neural_profiles.py 对应的模块命令；检查被选中的参数与模型官方默认配置。 |
| 分离预处理 | 保留声道直到分离完成；中央声像、REPET、HPSS、频谱减法，以及可选的 Demucs 与 Spleeter。可选原音、前景或神经网络目标声部。 | 在同一配对混音上测转谱 F1、分离 SNR，并核对长度和时间轴。实际表现依赖乐器声像和编配。[分离实验](benchmarks/SEPARATION.md) |
| 复杂节奏 | 动态规划追踪节拍，按局部拍距映射音符时间；手填 BPM 时使用固定速度。支持指定首个强拍、逐拍选择二进或三连音网格、休止、切分和跨小节延音。 | 用预先知道音符和时值的乐句核对 ABC；对实录另外计算谱面量化后的事件 F1。 |
| 五线谱图像 | 同一套 ABC 数据供网页和 CLI 使用，网页可试听原录音、打印、下载含全部谱行的 SVG 或 PNG。 | 浏览器上传多音样本，核对五线谱、吹吸标签、谱行数及导出尺寸。 |

## 使用

在已验证的 Python 3.9 环境，从项目根目录执行。分离模型首次使用需要能够取得权重。

```sh
python -m pip install -e '.[neural,separation]'
python -m sheet_draw.web
```

默认地址为 http://127.0.0.1:8765 。可用 `--port` 指定其他端口。命令行可直接生成同一份组合谱的 ABC 文件：

```sh
sheet-draw transcribe recording.wav --engine basic-pitch-duet --harmonica-type diatonic --harmonica-key Bb --meter 2 --subdivision auto -o abc --output-file result.abc
```

## 复核与后续判断

1. 运行 `python -m pytest -q` 检查现有与新增回归用例。运行 `python -m benchmarks.evaluate --basic-pitch` 复测合成音阶、带休止或特殊节奏的旋律和实录单音。
2. 运行 `python scripts/compare_performances.py /tmp/harp-eval --json results.json` 复测有作者标注的完整演奏；首次下载，此后核验缓存。[分组和逐曲评测](benchmarks/README.md)
3. 运行 `python -m benchmarks.neural_profiles /tmp/harp-eval` 比较开发集解码配置。识别输出与量化后的谱面分别计分，公开样例与留出曲目分开报告。
4. 实录中尚有漏音。若原始多音已经漏音，可针对弱音、压音或和弦密集段补充训练与验证数据；若原始事件正确、谱面时值仍错，应优先改进强拍和网格估计。口琴独奏、人声改编与混音中的口琴需要分别评测。[初版逐曲对照](docs/TRANSCRIPTION_EVALUATION.md)
