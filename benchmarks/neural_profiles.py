"""Select a Basic Pitch decoding profile on development songs, then score holdout once."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import soundfile as sf

from scripts.compare_performances import ROOT, cached_file, load_reference, metrics, score
from sheet_draw.basic_pitch_backend import HARMONICA_PROFILES, predict_harmonica_notes, select_duet


def evaluate(rows, names, cache):
    counts = {name: Counter() for name in names}
    for row in rows:
        audio, sr = sf.read(cached_file(cache, row["audio"], row["audio_blob"]), always_2d=True)
        labels = load_reference(row, cache)
        key = row["audio"].split("/")[0].rsplit("_", 1)[1]
        for name in names:
            with redirect_stdout(io.StringIO()):
                raw = predict_harmonica_notes(audio.mean(axis=1), sr, "diatonic", key, name)
            counts[name].update(score(labels, select_duet(raw, "diatonic", key)))
    return {name: metrics(c) for name, c in counts.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "benchmarks/performances_manifest.json").read_text())
    dev = evaluate([r for r in manifest if r["split"] == "dev"], HARMONICA_PROFILES, args.cache)
    best = max(dev, key=lambda name: dev[name]["offset_f1"])
    holdout = evaluate([r for r in manifest if r["split"] == "holdout"],
                       list(dict.fromkeys(["official", best])), args.cache)
    report = json.dumps(dict(development=dev, selected=best, holdout=holdout), indent=2)
    if args.json:
        args.json.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
