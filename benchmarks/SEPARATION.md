# 混音分离对照

从仓库根目录运行 `python -m benchmarks.separation`。安装 Demucs 后加 `--demucs` 可复现后两行。此次神经网络实测使用 Demucs 4.0.1、Torch / torchaudio 2.5.1、NumPy 1.26.4 和官方 htdemucs 权重（SHA-256 前缀 8726e21a）。首次加载模型需要联网。

混音由仓库内四个 CC0 真口琴单音的持续段构成。每音 360 ms，间隔 140 ms，口琴在左右声道等响，左侧加入有音量起伏的合成低音，右侧加入持续音。源录音见 [录音说明](README.md)。使用默认单音引擎、关闭谱面量化。音符匹配要求 MIDI 音高相同且起音误差不超过 50 ms。信噪比以干净口琴为参照，保留输出增益，惩罚漏音与伴奏残留。

| 前处理 | 匹配 / 输出音数（参考 4 音） | 音符 F1 | 信噪比 dB |
|---|---:|---:|---:|
| 原混音 | 0 / 1 | 0.00 | −5.74 |
| 中央声像 | 4 / 4 | 1.00 | 18.71 |
| 重复伴奏抑制（REPET） | 3 / 6 | 0.60 | 12.89 |
| 谐波保留（HPSS） | 0 / 1 | 0.00 | −5.41 |
| 稳态频谱减法 | 0 / 9 | 0.00 | 7.69 |
| Demucs 人声轨 | 0 / 6 | 0.00 | 0.01 |
| Demucs 其他乐器轨 | 0 / 1 | 0.00 | −6.56 |

中央声像在这个人为指定声像的例子里有直接优势。伴奏也居中、口琴偏向一侧、单声道、混响或真实乐队演奏都会改变结果。Demucs 的“其他乐器”轨包含多类乐器，其标签没有区分口琴。以上数字只描述这一段受控混音；未对 Spleeter 运行权重推理。

[Demucs 官方模型列表](https://github.com/facebookresearch/demucs/blob/main/demucs/remote/files.txt) · [Spleeter 原始波形 API](https://github.com/deezer/spleeter/wiki/4.-API-Reference)

以目标声部相对双声道伴奏的 RMS 电平重配同一混音，按起音和音高评分。用 `python -m benchmarks.separation --levels -6 0 6 --demucs` 可复算。

| 前处理 | −6 dB F1 | 0 dB F1 | +6 dB F1 |
|---|---:|---:|---:|
| dry | 1.000 | 1.000 | 1.000 |
| none | 0.000 | 0.000 | 0.133 |
| center | 1.000 | 1.000 | 1.000 |
| hpss | 0.000 | 0.000 | 0.154 |
| spectral | 0.167 | 0.667 | 0.667 |
| repet | 0.600 | 1.000 | 0.889 |
| demucs-vocals | 0.000 | 0.000 | 0.333 |
| demucs-other | 0.000 | 0.000 | 0.133 |
