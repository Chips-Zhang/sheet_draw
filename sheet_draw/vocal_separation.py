"""Source separation for transcription, preserving sample rate and time origin."""

from functools import lru_cache
from math import gcd

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree


def _mono(audio):
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _stereo(audio):
    if audio.ndim == 2 and audio.shape[1] == 2:
        return audio
    return np.repeat(_mono(audio)[:, None], 2, axis=1)


def _resample(audio, original, target):
    if original == target:
        return audio
    divisor = gcd(original, target)
    return signal.resample_poly(audio, target // divisor, original // divisor, axis=0)


def _restore(audio, model_rate, rate, length):
    data = _resample(audio, model_rate, rate)[:length]
    if len(data) < length:
        data = np.pad(data, [(0, length - len(data))] + [(0, 0)] * (data.ndim - 1))
    return data


def _repeating_mask(magnitude, sr, hop):
    """Estimate recurring accompaniment from distant, spectrally similar frames."""
    count = magnitude.shape[1]
    # Pool the spectrum to search by spectral shape, independent of loudness.
    features = np.stack([band.mean(axis=0) for band in np.array_split(
        magnitude, min(48, magnitude.shape[0]))], axis=1)
    features = np.sqrt(features)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    tree = cKDTree(features)
    exclusion = max(1, round(.6 * sr / hop))
    neighbours = min(count, 2 * exclusion + 9)
    mask = np.ones_like(magnitude)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        _, indices = tree.query(features[start:stop], k=neighbours)
        for i, row in enumerate(np.asarray(indices).reshape(stop - start, neighbours), start):
            distant = row[np.abs(row - i) > exclusion][:8]
            if len(distant) < 3:
                continue
            background = np.median(magnitude[:, distant], axis=1)
            foreground = np.maximum(magnitude[:, i] - background, 0)
            mask[:, i] = foreground**2 / (foreground**2 + background**2 + 1e-16)
    return mask


class VocalSeparator:
    """Foreground/background estimates; classical methods do not identify singers."""

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512,
                 method="spectral_subtraction"):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.method = method

    def separate(self, audio):
        data = np.asarray(audio, dtype=np.float32)
        if self.method in ("demucs", "spleeter"):
            stems = _neural_stems(data, self.sample_rate, self.method)
            return stems["vocals"], stems["accompaniment"]
        if self.method == "center" and (data.ndim != 2 or data.shape[1] != 2):
            raise ValueError("居中声像提取需要双声道音频")
        mixture = _mono(data)
        if not len(mixture):
            return mixture, mixture.copy()
        rate = min(self.sample_rate, 22050)
        audio = _resample(mixture, self.sample_rate, rate)
        # The same padded frame grid is used for both spectra and inversion.
        def spectrum(x):
            x = np.pad(x, (0, max(0, self.n_fft - len(x))))
            return signal.stft(x, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)[2]
        spec = spectrum(audio)
        mag = np.abs(spec)
        if self.method in ("spectral", "spectral_subtraction"):
            # Remove the stationary spectral floor while retaining short-lived
            # harmonics. The gain floor limits artefacts on sustained lead notes.
            stationary = np.percentile(mag, 20, axis=1, keepdims=True).astype(mag.dtype)
            mask = np.maximum(.04, 1. - stationary / (mag + 1e-10))
        elif self.method == "hpss":
            harmonic = median_filter(mag, size=(1, 15))
            transient = median_filter(mag, size=(15, 1))
            mask = harmonic**2 / (harmonic**2 + transient**2 + 1e-16)
        elif self.method == "center":
            side = _resample((data[:, 0] - data[:, 1]) * .5, self.sample_rate, rate)
            # Identical L/R has zero side. A hard-panned instrument has equal mid/side.
            side_power = np.abs(spectrum(side))**2
            mask = np.maximum(0., 1. - side_power / (mag**2 + 1e-16))
        elif self.method == "repet":
            mask = _repeating_mask(mag, rate, self.hop_length)
        elif self.method == "median_filtering":
            background = median_filter(mag, size=(15, 5))
            mask = np.maximum(0, mag - .8 * background) / (mag + 1e-10)
        else:
            raise ValueError(f"未知分离方法：{self.method}")
        _, lead = signal.istft(spec * mask, nperseg=self.n_fft,
                               noverlap=self.n_fft - self.hop_length)
        lead = _restore(lead[:len(audio)], rate, self.sample_rate, len(mixture))
        return lead, mixture - lead

    def extract_vocal(self, audio):
        return self.separate(audio)[0]

    def extract_accompaniment(self, audio):
        return self.separate(audio)[1]


class SpleeterWrapper:
    def __init__(self, model="2stems"):
        self.model = model
        try:
            from spleeter.separator import Separator
        except ImportError:
            self.separator = None
        else:
            self.separator = Separator(f"spleeter:{model}")
        self.available = self.separator is not None

    def separate(self, audio, sample_rate=44100):
        if not self.available:
            raise RuntimeError("当前环境未安装 Spleeter，请安装 spleeter 后重试。")
        audio = np.asarray(audio, dtype=np.float32)
        wave = _resample(_stereo(audio), sample_rate, 44100)
        stems = self.separator.separate(wave)
        if "vocals" not in stems:
            raise RuntimeError("Spleeter 没有返回人声轨")
        return {name: _restore(_mono(track) if audio.ndim == 1 else track,
                               44100, sample_rate, len(audio)) for name, track in stems.items()}

    def extract_vocal(self, audio, sample_rate=44100):
        return self.separate(audio, sample_rate)["vocals"]


@lru_cache(maxsize=1)
def _demucs_model():
    try:
        from demucs.pretrained import get_model
    except (ImportError, OSError) as exc:
        raise RuntimeError("当前环境需要安装可用的 demucs、torch 和 torchaudio。") from exc
    try:
        return get_model("htdemucs").cpu().eval()
    except Exception as exc:
        raise RuntimeError("无法加载 htdemucs 权重，请检查模型缓存和首次下载连接。") from exc


def _neural_stems(audio, rate, method):
    if method == "spleeter":
        return SpleeterWrapper().separate(audio, rate)
    model = _demucs_model()
    import torch
    from demucs.apply import apply_model
    samples = _resample(_stereo(audio), rate, model.samplerate)
    wave = torch.from_numpy(np.ascontiguousarray(samples.T))
    reference = wave.mean(0)
    scale = reference.std(unbiased=False).clamp_min(1e-8)
    with torch.inference_mode():
        prediction = apply_model(model, ((wave - reference.mean()) / scale)[None],
                                 device="cpu", shifts=0, split=True, progress=False)[0]
        prediction = prediction * scale + reference.mean()
    stems = {}
    for name, track in zip(model.sources, prediction):
        data = track.numpy().T
        stems[name] = _restore(_mono(data) if audio.ndim == 1 else data,
                               model.samplerate, rate, len(audio))
    stems["accompaniment"] = sum(track for name, track in stems.items() if name != "vocals")
    return stems


def preprocess_for_transcription(audio, sample_rate=44100, method="auto", target="vocals"):
    """Select a stem first, then return a time-aligned mono transcription signal."""
    data = np.asarray(audio, dtype=np.float32)
    if method == "auto":
        from importlib.util import find_spec
        method = "spleeter" if find_spec("spleeter") else "hpss"
    if method in ("demucs", "spleeter"):
        # Long songs would otherwise retain all separated stems at full length.
        block = 45 * sample_rate
        if len(data) > block:
            margin = 2 * sample_rate
            output = np.empty(len(data), dtype=np.float32)
            for start in range(0, len(data), block):
                end = min(start + block, len(data))
                left, right = max(0, start-margin), min(len(data), end+margin)
                stems = _neural_stems(data[left:right], sample_rate, method)
                if target not in stems:
                    raise ValueError(f"{method} 不提供音轨 {target}")
                output[start:end] = _mono(stems[target])[start-left:end-left]
            return output
        stems = _neural_stems(data, sample_rate, method)
        if target not in stems:
            raise ValueError(f"{method} 不提供音轨 {target}")
        return _mono(stems[target])
    if target not in ("vocals", "accompaniment"):
        raise ValueError("该方法支持前景或背景音轨")
    lead, backing = VocalSeparator(sample_rate=sample_rate, method=method).separate(data)
    return lead if target == "vocals" else backing


def separate_vocal(audio, sample_rate=44100, use_spleeter=False):
    method = "spleeter" if use_spleeter else "spectral"
    return preprocess_for_transcription(audio, sample_rate, method)
