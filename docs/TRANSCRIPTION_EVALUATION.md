# 连续口琴演奏的转谱效果

选取 C、G、降 B 调各一个完整片段，共 86 个标注音符。参考答案来自论文作者的演奏标注，保留同时发声的音。每个引擎收到同一整段音频，不裁剪前导静音，不按录音调整参数。表中左侧数字是音高和起音同时匹配的音符 F1，右侧还要求收音匹配。每个参考音只能命中一次，多报和漏报都会扣分。

| 录音 | 参考音数 | 旧算法 | 新单音追踪 | Basic Pitch 原始多音 | Basic Pitch 网页单行谱 |
|---|---:|---:|---:|---:|---:|
| Oh! Susanna（C） | 29 | 0 / 0 | .667 / .667 | .841 / .754 | .931 / .828 |
| Ode to Joy（G） | 25 | 0 / 0 | .170 / .085 | .267 / .089 | .143 / .000 |
| Help Me（降 B） | 32 | 0 / 0 | .167 / .083 | .407 / .259 | .304 / .130 |

按 mir_eval 的音符匹配规则，音高容差为 50 音分，起音容差为 50 ms，收音容差取 50 ms 和参考时长 20% 中的较大者。评测关闭节奏量化，表中数字衡量识别输出；孔位、拍号和页面排版需要另行验证。这三段是小样本对照，不能据此推算任意歌曲的准确率。[原始音频与标注](https://github.com/filipemlins/Automatic-Music-Transcription-Harmonica/tree/25bf104035b3aac560324576103dc0c664e87256/AnnotatedSetupFinalMidi) · [逐段运行结果](../benchmarks/performances.jsonl)

## 深度学习路线

已接入 Spotify Basic Pitch 0.4.0 的官方 ONNX 权重。模型先预测可重叠的音符事件。为了生成一行吹吸谱，程序从中选择一组起止兼容的音，按持续时间和模型振幅加权，容许相邻音有少量收尾重叠。这一步把 Oh! Susanna 的预测从 40 音减至 29 音，也使 Ode to Joy 丢掉了部分同时发声的标注音。复杂双音演奏仍需人工核谱。网页和 CLI 都可切换引擎。[Basic Pitch 官方实现](https://github.com/spotify/basic-pitch)

下一轮开发可优先针对双音和压音分别评估。对双音保留模型的重叠事件，再按口琴调性、相邻孔位和吹吸方向筛选可演奏组合，用和弦记谱；作者的口琴专用研究提供了相应约束和更多演奏标注。对压音可用 torchcrepe 估计连续音高与周期性，再比较滑音是否被误拆成多个半音；它仍需要单独的起止音分割。这两条路线尚未接入网页。[口琴转录研究与数据](https://github.com/filipemlins/Automatic-Music-Transcription-Harmonica) · [torchcrepe](https://github.com/maxrmorrison/torchcrepe)

## 复算

在项目根目录使用 Python 3.9 虚拟环境执行。首次运行按固定上游版本下载三段录音和标注，此后读取本地缓存。另有离线的合成音阶、带休止旋律和真实单音评测。[离线基线](../benchmarks/README.md)

```sh
python -m pip install -e '.[neural]'
python scripts/compare_performances.py /tmp/sheet-harp-phrases
```
