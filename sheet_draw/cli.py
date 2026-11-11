"""
命令行接口
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .detector import PitchDetector, ToneAnalyzer
from .transcriber import MusicTranscriber, TranscriptionConfig


def main():
    parser = argparse.ArgumentParser(
        description="Sheet Draw - 乐谱绘制工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 音高检测
  sheet-draw pitch audio.wav                    # 分析音频文件
  sheet-draw pitch audio.wav -v                 # 显示详细信息
  sheet-draw pitch audio.wav --continuous       # 连续分析整个音频

  # 扒谱功能
  sheet-draw transcribe song.wav          # 扒谱并输出简谱
  sheet-draw transcribe song.wav -o midi  # 输出为MIDI文件
  sheet-draw transcribe song.wav --bpm    # 只检测BPM

  # 人声分离
  sheet-draw transcribe song.wav --separate-vocal --separation-method spleeter

  # Spleeter 管理
  sheet-draw install-spleeter             # 安装 Spleeter
  sheet-draw install-spleeter --check     # 检查是否已安装
  sheet-draw install-spleeter --test      # 测试 Spleeter
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 音高检测命令
    pitch_parser = subparsers.add_parser("pitch", help="音高检测")
    pitch_parser.add_argument("input", help="输入音频文件路径")
    pitch_parser.add_argument(
        "-m", "--method",
        choices=["yin", "autocorr", "fft"],
        default="yin",
        help="检测算法 (默认: yin)"
    )
    pitch_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.1,
        help="YIN算法阈值 0.0-1.0 (默认: 0.1)"
    )
    pitch_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细信息"
    )
    pitch_parser.add_argument(
        "--continuous",
        action="store_true",
        help="连续分析整个音频"
    )
    
    # 扒谱命令
    transcribe_parser = subparsers.add_parser("transcribe", help="扒谱 (音频转曲谱)")
    transcribe_parser.add_argument("input", help="输入音频文件路径")
    transcribe_parser.add_argument(
        "-o", "--output",
        choices=["jianpu", "midi", "json", "tab", "abc"],
        default=None,
        help="输出格式 (默认: jianpu, tab=口琴TAB谱)"
    )
    transcribe_parser.add_argument(
        "--output-file",
        type=str,
        help="输出文件路径"
    )
    transcribe_parser.add_argument(
        "--bpm",
        action="store_true",
        help="只检测BPM"
    )
    transcribe_parser.add_argument(
        "--tempo-min",
        type=float,
        default=60,
        help="最小BPM (默认: 60)"
    )
    transcribe_parser.add_argument(
        "--tempo-max",
        type=float,
        default=180,
        help="最大BPM (默认: 180)"
    )
    
    transcribe_parser.add_argument("--engine", choices=["monophonic", "basic-pitch", "basic-pitch-poly", "basic-pitch-duet", "legacy"],
                                   default="monophonic", help="音符识别引擎")

    transcribe_parser.add_argument("--meter", type=int, choices=(2, 3, 4), default=4)
    transcribe_parser.add_argument("--subdivision", choices=("auto", "straight", "triplet"), default="auto")

    # 人声分离选项
    transcribe_parser.add_argument(
        "--separate-vocal",
        action="store_true",
        help="启用人声分离"
    )
    transcribe_parser.add_argument(
        "--separation-method",
        choices=["hpss", "spectral", "repet", "center", "spleeter", "demucs"],
        default="hpss",
        help="人声分离方法 (默认: hpss)"
    )
    
    transcribe_parser.add_argument("--separation-target", choices=["vocals", "accompaniment", "other"],
                                   default="vocals", help="分离后用于转谱的音轨（other 需要 Demucs）")

    # 口琴选项
    transcribe_parser.add_argument(
        "--harmonica",
        action="store_true",
        help="启出口琴优化"
    )
    transcribe_parser.add_argument(
        "--harmonica-type",
        choices=["chromatic", "diatonic"],
        default="chromatic",
        help="口琴类型 (默认: chromatic)"
    )
    transcribe_parser.add_argument(
        "--harmonica-key",
        type=str,
        default="C",
        help="口琴调性 (默认: C)"
    )
    
    # Spleeter 安装命令
    spleeter_parser = subparsers.add_parser("install-spleeter", help="安装 Spleeter 人声分离工具")
    spleeter_parser.add_argument(
        "--gpu",
        action="store_true",
        help="安装 GPU 版本 (需要 NVIDIA GPU)"
    )
    spleeter_parser.add_argument(
        "--check",
        action="store_true",
        help="检查 Spleeter 是否已安装"
    )
    spleeter_parser.add_argument(
        "--test",
        action="store_true",
        help="测试 Spleeter 是否工作正常"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "transcribe":
        poly = args.engine in ("basic-pitch-duet", "basic-pitch-poly")
        args.output = args.output or ("abc" if poly else "jianpu")
        if poly and args.output in ("jianpu", "tab"):
            parser.error("多音和双音请用 -o abc 输出组合谱，或选择 midi / json。")

    # 处理 install-spleeter 命令 (不需要输入文件)
    if args.command == "install-spleeter":
        handle_install_spleeter(args)
        return
    
    # 检查文件是否存在
    if not Path(args.input).exists():
        print(f"错误: 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if args.command == "pitch":
        handle_pitch(args)
    elif args.command == "transcribe":
        handle_transcribe(args)


def handle_pitch(args):
    """处理音高检测命令"""
    detector = PitchDetector(
        method=args.method,
        threshold=args.threshold
    )
    
    try:
        if args.continuous:
            # 连续分析
            import soundfile as sf
            audio, sr = sf.read(args.input)
            detector.sample_rate = sr
            
            analyzer = ToneAnalyzer(detector)
            result = analyzer.analyze(audio)
            
            if "error" in result:
                print(f"无法检测到音高: {result['error']}")
                sys.exit(1)
            
            print("\n🎵 音调分析结果\n")
            print(f"主音高:    {result['mean_frequency']} Hz")
            print(f"音符:      {result['note']}")
            print(f"简谱:      {result['tone']} (唱名)")
            print(f"八度:      {result['octave']}")
            print(f"音分偏差:  {result['cents']} cents")
            print(f"稳定性:    {result['stability']*100:.1f}%")
            
            if args.verbose:
                print(f"\n详细统计:")
                print(f"  频率标准差: {result['std_frequency']} Hz")
                print(f"  检测帧数:   {result['pitch_count']}")
                print(f"  频率范围:   {result['frequency_range']['min']} - {result['frequency_range']['max']} Hz")
                print(f"  主导音符:   {result['dominant_note']}")
        
        else:
            # 单帧分析
            result = detector.detect_from_file(args.input)
            
            if result is None:
                print("无法检测到音高")
                sys.exit(1)
            
            print("\n🎵 音高检测结果\n")
            print(f"频率:      {result['frequency']:.2f} Hz")
            print(f"音符:      {result['note']}")
            print(f"简谱:      {result['tone']} (唱名)")
            print(f"八度:      {result['octave']}")
            print(f"音分偏差:  {result['cents']} cents")
            print(f"MIDI编号:  {result['midi']}")
            
            if args.verbose:
                print(f"\n算法信息:")
                print(f"  使用算法: {args.method.upper()}")
                print(f"  采样率:   {detector.sample_rate} Hz")
    
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def handle_transcribe(args):
    """处理扒谱命令"""
    from .rhythm import estimate_bpm_simple
    import soundfile as sf
    
    try:
        # 只检测BPM
        if args.bpm:
            audio, sr = sf.read(args.input)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            bpm = estimate_bpm_simple(audio, sr)
            print(f"\n🥁 BPM: {bpm}\n")
            return
        
        # 完整扒谱
        print("\n🎼 开始扒谱...")
        print(f"   输入文件: {args.input}")
        
        # 显示配置
        if args.separate_vocal:
            print(f"   人声分离: 启用 ({args.separation_method})")
        if args.harmonica:
            print(f"   口琴优化: 启用 ({args.harmonica_key}调{args.harmonica_type})")
        print()
        
        config = TranscriptionConfig(
            engine=args.engine,
            enable_quantization=args.output != "abc" and not (args.output == "json" and args.engine in ("basic-pitch-duet", "basic-pitch-poly")),
            tempo_min=args.tempo_min,
            tempo_max=args.tempo_max,
            enable_vocal_separation=args.separate_vocal,
            vocal_separation_method=args.separation_method,
            separation_target=args.separation_target,
            harmonica_optimization=args.harmonica,
            harmonica_type=args.harmonica_type,
            harmonica_key=args.harmonica_key
        )
        transcriber = MusicTranscriber(config)
        result = transcriber.transcribe_file(args.input)
        
        # 输出结果
        if args.output == "jianpu":
            jianpu_text = result.to_jianpu()
            print(jianpu_text)
            
            # 保存到文件
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(jianpu_text)
                print(f"\n简谱已保存到: {args.output_file}")
        
        elif args.output == "abc":
            from .notation import score_abc
            abc = score_abc(result, args.harmonica_type, args.meter, result.bpm,
                            args.harmonica_key, args.subdivision, result.beats)
            if args.output_file:
                Path(args.output_file).write_text(abc, encoding="utf-8")
            else:
                print(abc)
        elif args.output == "midi":
            output_path = args.output_file or args.input.rsplit('.', 1)[0] + ".mid"
            result.to_midi(output_path)
            print(f"\nMIDI文件已保存到: {output_path}")
        
        elif args.output == "json":
            import json
            output_path = args.output_file or args.input.rsplit('.', 1)[0] + ".json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\nJSON文件已保存到: {output_path}")
        
        elif args.output == "tab":
            # 口琴TAB谱
            tab_text = result.to_harmonica_tab(
                harmonica_type=args.harmonica_type,
                harmonica_key=args.harmonica_key
            )
            print(tab_text)
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(tab_text)
                print(f"\nTAB谱已保存到: {args.output_file}")
    
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_install_spleeter(args):
    """处理 Spleeter 安装命令"""
    from .vocal_separation import SpleeterWrapper
    
    # 检查是否已安装
    wrapper = SpleeterWrapper()
    
    if args.check:
        if wrapper.available:
            print("✅ Spleeter 已安装")
            try:
                import spleeter
                print(f"   版本: {spleeter.__version__}")
            except:
                pass
        else:
            print("❌ Spleeter 未安装")
            print("   运行: note-get install-spleeter")
        return
    
    if args.test:
        if not wrapper.available:
            print("❌ Spleeter 未安装")
            print("   请先运行: note-get install-spleeter")
            return
        print("🧪 测试 Spleeter...")
        try:
            from spleeter.separator import Separator
            print("   加载模型...")
            separator = Separator('spleeter:2stems')
            print("✅ Spleeter 工作正常!")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        return
    
    # 安装 Spleeter
    if wrapper.available:
        print("✅ Spleeter 已经安装")
        response = input("是否重新安装? (y/N): ")
        if response.lower() != 'y':
            return
    
    print("🎵 正在安装 Spleeter...")
    print("   这是一个深度学习人声分离工具")
    print("   需要下载约 1.5GB 的模型文件")
    print("   安装可能需要几分钟时间...")
    print()
    
    if args.gpu:
        print("   安装 GPU 版本 (需要 NVIDIA GPU 和 CUDA)")
        packages = ["tensorflow-gpu", "spleeter"]
    else:
        print("   安装 CPU 版本")
        packages = ["spleeter"]
    
    try:
        import subprocess
        import sys
        
        for package in packages:
            print(f"   安装 {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        print()
        print("✅ Spleeter 安装成功!")
        print()
        print("📝 使用方法:")
        print("   1. 命令行: note-get transcribe song.wav --separate-vocal --separation-method spleeter")
        print("   2. Python:")
        print("      from note_get import SpleeterWrapper")
        print("      wrapper = SpleeterWrapper()")
        print("      vocal = wrapper.extract_vocal(audio, sample_rate)")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        print()
        print("💡 可能的解决方案:")
        print("   1. 确保网络连接正常")
        print("   2. 使用管理员权限运行")
        print("   3. 手动安装: pip install spleeter")
        print()


if __name__ == "__main__":
    main()
