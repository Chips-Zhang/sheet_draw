from .detector import PitchDetector, ToneAnalyzer
from .audio_utils import (
    generate_sine_wave,
    generate_note,
    save_wav,
    load_wav,
    frequency_to_note,
    note_to_frequency
)
from .onset import OnsetDetector
from .rhythm import BeatTracker, estimate_bpm_simple
from .key_detection import KeyDetector
from .score import (
    Note,
    JianpuGenerator,
    MidiGenerator,
    TranscriptionResult,
    quantize_notes
)
from .transcriber import (
    MusicTranscriber,
    TranscriptionConfig,
    transcribe_audio,
    transcribe_file
)
from .vocal_separation import (
    VocalSeparator,
    separate_vocal,
    preprocess_for_transcription
)
from .harmonica import (
    Harmonica,
    ChromaticHarmonica,
    DiatonicHarmonica,
    HarmonicaNote,
    HarmonicaScoreGenerator,
    create_chromatic_harmonica,
    create_diatonic_harmonica,
    generate_harmonica_tab
)

__version__ = "0.3.0"
__all__ = [
    # 基础检测
    "PitchDetector",
    "ToneAnalyzer",
    
    # 高级功能
    "OnsetDetector",
    "BeatTracker",
    "KeyDetector",
    "MusicTranscriber",
    "TranscriptionConfig",
    
    # 人声分离
    "VocalSeparator",
    "separate_vocal",
    "preprocess_for_transcription",
    
    # 口琴功能
    "Harmonica",
    "ChromaticHarmonica",
    "DiatonicHarmonica",
    "HarmonicaNote",
    "HarmonicaScoreGenerator",
    "create_chromatic_harmonica",
    "create_diatonic_harmonica",
    "generate_harmonica_tab",
    
    # 曲谱生成
    "Note",
    "JianpuGenerator",
    "MidiGenerator",
    "TranscriptionResult",
    
    # 工具函数
    "generate_sine_wave",
    "generate_note",
    "save_wav",
    "load_wav",
    "frequency_to_note",
    "note_to_frequency",
    "estimate_bpm_simple",
    "quantize_notes",
    "transcribe_audio",
    "transcribe_file",
]
