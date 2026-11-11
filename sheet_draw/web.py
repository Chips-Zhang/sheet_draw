"""Loopback web interface for the monophonic transcription pipeline."""
import argparse
from importlib.util import find_spec
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import numpy as np
import soundfile as sf

from .notation import score_abc, relative_major, tonic_pc, PITCH_NAMES
from .vocal_arrangement import arrange, estimate_reference_tonic, fit_missing_octaves, vocal_pulse, TONICS
from .transcriber import MusicTranscriber, TranscriptionConfig

ASSETS = Path(__file__).with_name("web_assets")
STATIC = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/pdf-lib.min.js": ("pdf-lib.min.js", "text/javascript"),
          "/combined-score.js": ("combined-score.js", "text/javascript"),
          "/style.css": ("style.css", "text/css"), "/abc.js": ("abc.js", "text/javascript")}


def convert(payload, instrument, meter=4, bpm=None, engine="monophonic", key="C",
            subdivision="straight", separation="none", downbeat=None, source="instrument", shift=None, target_tonic=None, fit_octaves=True, video_link=False):
    if target_tonic is not None and target_tonic not in TONICS:
        raise ValueError("未知目标曲调。")
    if source not in ("instrument", "song"):
        raise ValueError("未知音源类型。")
    if source == "song":
        if find_spec("demucs") is None:
            raise ValueError("原唱改编需要安装 Demucs。")
        engine, separation = "basic-pitch", "demucs-vocals"
    if engine not in ("monophonic", "basic-pitch", "basic-pitch-poly", "basic-pitch-duet"):
        raise ValueError("未知识别引擎。")
    if engine.startswith("basic-pitch") and not has_basic_pitch():
        raise ValueError("此环境尚未安装 Basic Pitch。")
    if meter not in (2, 3, 4) or (bpm is not None and not 30 <= bpm <= 300):
        raise ValueError("拍号或速度超出范围。")
    if instrument not in ("diatonic", "chromatic", "paddy"):
        raise ValueError("请选择支持的口琴。")
    if key not in ("C", "G", "Bb", "A") or subdivision not in ("auto", "straight", "triplet"):
        raise ValueError("口琴调性或节奏细分无效。")
    if separation not in ("none", "hpss", "repet", "spectral", "center", "spleeter", "demucs-vocals", "demucs-other"):
        raise ValueError("未知音轨处理方式。")
    try:
        stream = sf.SoundFile(io.BytesIO(payload))
    except (RuntimeError, ValueError) as exc:
        raise ValueError("无法读取音频，请使用 WAV、MP3、FLAC 或 OGG 文件。") from exc
    with stream:
        limit = 300 if video_link else (600 if source == "song" else 120)
        max_samples = 60000000 if source == "song" or video_link else 24000000
        if (not 4000 <= stream.samplerate <= 192000 or not 1 <= stream.channels <= 8
                or stream.frames / stream.samplerate > limit or stream.frames * stream.channels > max_samples):
            raise ValueError(f"请使用 {limit//60} 分钟以内的录音，或降低采样率后重试。")
        sr = stream.samplerate
        data = stream.read(dtype="float32", always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("音频为空或数据异常。")
    if downbeat is not None and not 0 <= downbeat <= len(data) / sr:
        raise ValueError("首个强拍必须位于录音内。")
    method, _, target = separation.partition("-")
    config = TranscriptionConfig(sample_rate=sr, enable_quantization=False, engine=engine,
                                 harmonica_type=instrument, harmonica_key=key,
                                 enable_vocal_separation=method != "none",
                                 vocal_separation_method=method, separation_target=target or "vocals")
    result = MusicTranscriber(config).transcribe(data)
    if not result.notes:
        raise ValueError("没有识别到稳定的音符，请试试单音旋律录音。")
    actual_shift = unplayable = octave_changes = 0
    score_tonic = relative_major(result.key)
    if source == "song":
        original_tonic = estimate_reference_tonic(data,sr,result.notes) if target_tonic and shift is None else None
        result.notes, actual_shift, unplayable = arrange(result.notes, instrument, key, shift,
                                                         target_tonic, original_tonic)
        score_tonic = target_tonic if target_tonic and shift is None else PITCH_NAMES[(tonic_pc(score_tonic) + actual_shift) % 12]
        result.bpm, result.beats = vocal_pulse(result.bpm, result.beats)
        if fit_octaves:
            result.notes, octave_changes, unplayable = fit_missing_octaves(result.notes, instrument, key)
    beats = result.beats if bpm is None else []
    bpm = bpm or result.bpm
    tie_ends = []
    abc = score_abc(result, instrument, meter, bpm, key, subdivision, beats, downbeat, tie_ends, tonic=score_tonic)
    return dict(abc=abc, tieEnds=tie_ends, bpm=round(bpm), count=len(result.notes), shift=actual_shift, unplayable=unplayable, octaveChanges=octave_changes)


def has_basic_pitch():
    return all(find_spec(package) is not None for package in ("basic_pitch", "onnxruntime"))


class Handler(BaseHTTPRequestHandler):
    def send_data(self, code, mime, body):
        self.send_response(code)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/capabilities":
            return self.send_data(200, "application/json", json.dumps({"basicPitch": has_basic_pitch(), "spleeter": find_spec("spleeter") is not None, "demucs": find_spec("demucs") is not None}).encode())
        asset = STATIC.get(urlsplit(self.path).path)
        if not asset:
            return self.send_error(404)
        name, mime = asset
        self.send_data(200, mime, (ASSETS / name).read_bytes())

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path not in ("/transcribe", "/transcribe-link"):
            return self.send_error(404)
        # Restrict uploads to requests made by our own local UI.
        if self.headers.get("X-Sheet-Draw") != "1":
            return self.send_error(403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            is_link = url.path == "/transcribe-link"
            if is_link and not 0 < size <= 8192:
                raise ValueError("请输入有效的媒体链接。")
            if not 0 < size <= 128 * 1024 ** 2:
                raise ValueError("请选择小于等于 128 MB 的音频。")
            instrument = parse_qs(url.query).get("instrument", ["chromatic"])[0]
            query = parse_qs(url.query)
            meter = int(query.get("meter", ["4"])[0])
            bpm = int(query["bpm"][0]) if "bpm" in query else None
            payload = self.rfile.read(size)
            title, video_link = None, False
            if is_link:
                from .media_link import link_audio
                limit = 600 if query.get("source") == ["song"] else 120
                payload, title, video_link = link_audio(payload.decode("utf-8"), limit)
            output = convert(payload, instrument, meter, bpm, query.get("engine", ["monophonic"])[0],
                             query.get("key", ["C"])[0], query.get("subdivision", ["straight"])[0],
                             query.get("separation", ["none"])[0],
                             float(query["downbeat"][0]) if "downbeat" in query else None,
                             query.get("source", ["instrument"])[0],
                             int(query["shift"][0]) if query.get("shift", [""])[0] else None,
                             query.get("tonic", [None])[0], query.get("fit", ["1"])[0] == "1", video_link=video_link)
            if is_link:
                output["title"] = title
            status = 200
        except ValueError as exc:
            status, output = 400, {"error": str(exc)}
        except Exception:
            import traceback
            traceback.print_exc()
            status, output = 500, {"error": "转谱失败，请检查录音或换一段较短的旋律。"}
        self.send_data(status, "application/json", json.dumps(output).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
