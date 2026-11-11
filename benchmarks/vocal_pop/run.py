"""Evaluate original commercial album previews against user-supplied harmonica arrangements."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

import numpy as np
import soundfile as sf

from .compare import compare, distance, reference
from sheet_draw import Note, TranscriptionResult
from sheet_draw.audio_utils import frequency_to_note
from sheet_draw.basic_pitch_backend import predict_melody
from sheet_draw.monophonic import track_notes
from sheet_draw.notation import score_abc
from sheet_draw.rhythm import BeatTracker
from sheet_draw.vocal_arrangement import arrange, estimate_reference_tonic, fit_missing_octaves, vocal_pulse, TONICS
from sheet_draw.vocal_separation import preprocess_for_transcription

ROOT = Path(__file__).parent


def acquire(cache, name, info):
    meta = cache / (name + ".source.json")
    audio = cache / (name + ".m4a")
    wav = cache / (name + ".wav")
    if not meta.exists() or not audio.exists():
        url = f"https://itunes.apple.com/lookup?id={info['track']}&country=tw"
        rows = json.loads(urlopen(url, timeout=30).read())["results"]
        row = next(r for r in rows if r.get("trackId") == info["track"] and r["artistName"] == info["artist"])
        data = urlopen(row["previewUrl"], timeout=60).read()
        audio.write_bytes(data)
        meta.write_text(json.dumps(row, ensure_ascii=False))
        if wav.exists():
            wav.unlink()
    if not wav.exists():
        if shutil.which("afconvert"):
            cmd = ["afconvert","-f","WAVE","-d","LEI16",str(audio),str(wav)]
        else:
            cmd = ["ffmpeg","-v","error","-i",str(audio),str(wav)]
        subprocess.run(cmd, check=True)
    row = json.loads(meta.read_text())
    return wav, dict(track_id=row["trackId"], artist=row["artistName"], album=row["collectionName"],
                     track_url=row["trackViewUrl"], preview_sha256=hashlib.sha256(audio.read_bytes()).hexdigest())


def events(notes):
    return [[round(n.start_time,4),round(n.duration,4),round(69+12*np.log2(n.frequency/440)),
             round(n.velocity,3)] for n in notes]


def decode(audio, rate):
    vocal = preprocess_for_transcription(audio, rate, "demucs", "vocals")
    return {"mix":events(predict_melody(audio.mean(axis=1),rate)),
            "vocals":events(predict_melody(vocal,rate)), "tracker":events(track_notes(vocal,rate))}


def notes_from_events(rows):
    notes=[]
    for start,duration,midi,volume in rows:
        hz=440*2**((midi-69)/12)
        notes.append(Note(frequency_to_note(hz),hz,start,duration,volume))
    return notes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores",type=Path,help="提供的目标谱根目录")
    parser.add_argument("cache",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--reuse-events",type=Path,help="重排已保存且试听指纹相同的事件")
    args=parser.parse_args()
    saved=json.loads(args.reuse_events.read_text()) if args.reuse_events else {}
    args.cache.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    phrases=json.loads((ROOT/"phrases.json").read_text())
    pages=[]
    report={}
    for name,info in phrases.items():
        target=(args.scores/name/"score_1.jpg").resolve(strict=True)
        wav,source=acquire(args.cache,name,info)
        audio,rate=sf.read(wav,always_2d=True,dtype="float32")
        previous=saved.get(name, {})
        if previous.get("source", {}).get("preview_sha256")==source["preview_sha256"]:
            pred=previous["events"]
        else:
            pred=decode(audio,rate)
        comparison=compare(info,pred)
        vocal_notes=notes_from_events(pred["vocals"])
        source_tonic=estimate_reference_tonic(audio,rate,vocal_notes)
        notes,shift,unplayable=arrange(vocal_notes,info["instrument"],info["harp"],
                                       target_tonic=info["tonic"],source_tonic=source_tonic)
        notes,octave_changes,unplayable=fit_missing_octaves(notes,info["instrument"],info["harp"])
        i,j=comparison["degree_events"]
        phrase=reference(info["phrases"][comparison["phrase"]], info["tonic"])[i-1:j]
        start,end=comparison["window_seconds"]
        actual=[(n[2]+shift)%12 for n in pred["vocals"] if start <= n[0] <= end]
        d,extra,missing,wrong=distance(phrase,actual)
        comparison["arranged"]=dict(similarity=round(1-d/max(len(phrase),len(actual)),3),
                                    extra=extra,missing=missing,wrong_pitch_class=wrong)
        names="C Db D Eb E F Gb G Ab A Bb B".split()
        comparison["target_pitch_classes"]=[names[n] for n in phrase]
        comparison["arranged_pitch_classes"]=[names[n] for n in actual]
        rhythm=BeatTracker(sample_rate=rate,tempo_min=60,tempo_max=180).analyze(audio.mean(axis=1))
        bpm,beats=vocal_pulse(rhythm["bpm"],rhythm["beats"])
        result=TranscriptionResult(notes,"",bpm,beats)
        from scripts.compare_performances import quantized_score_notes
        written=events(quantized_score_notes(result))
        abc=score_abc(result,info["instrument"],4,result.bpm,info["harp"],"auto",result.beats,tonic=info["tonic"])
        (args.output/(name+".abc")).write_text(abc)
        report[name]=dict(source=source, duration=len(audio)/rate, target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          comparison=comparison, auto_shift=shift, unplayable=unplayable,
                          octave_changes=octave_changes,note_count=len(notes), bpm=result.bpm, estimated_source_do=TONICS[source_tonic],
                          target_do=info["tonic"],events=pred,written=written)
        url=html.escape(source["track_url"],quote=True)
        phrase=html.escape(comparison["phrase"])
        start,end=comparison["window_seconds"]
        pages.append(f'<section><h2>{html.escape(name)}</h2><p>{info["artist"]} · {info["album"]} · '
                     f'<a href="{url}">原唱专辑试听</a> · <a href="{target.as_uri()}">查看提供的目标谱</a></p>'
                     f'<p><a href="{(target.parent/"demo.mp3").as_uri()}">目标口琴示范</a> · '
                     f'<a href="{(target.parent/"accompaniment.mp3").as_uri()}">目标伴奏</a></p>'
                     f'<p>{info["harp"]} 调 {info["instrument"]}，原唱 1≈{TONICS[source_tonic]}，目标 1={info["tonic"]}，整体移调 {shift:+} 半音，'
                     f'{len(notes)} 个音，{octave_changes} 个音单独换八度，{unplayable} 个无对应孔位。以下为整段试听的生成谱。</p>'
                     f'<p>对照片段：试听 {start:.2f}–{end:.2f} 秒，目标谱{phrase}，'
                     f'{comparison["reference_count"]} 个参考音。音级序列相似度：'
                     f'混音直取 {comparison["mix"]["similarity"]:.1%}，分离后 {comparison["vocals"]["similarity"]:.1%}。</p>'
                     f'<p>按实际出谱调性对照：{comparison["arranged"]["similarity"]:.1%}。<br>'
                     f'目标音级　{" ".join(comparison["target_pitch_classes"])}<br>'
                     f'生成音级　{" ".join(comparison["arranged_pitch_classes"])}</p>'
                     f'<details open><summary>五线谱与孔位</summary><div class="score"></div>'
                     f'<script type="text/plain" class="abc">{abc}</script></details></section>')
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    js=Path(__file__).resolve().parents[2]/"sheet_draw/web_assets/abc.js"
    import os
    asset=os.path.relpath(js,args.output)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原唱改编对照</title>
<style>body{max-width:960px;margin:auto;padding:24px;font:16px/1.7 system-ui;color:#22342c;background:#fafbf8}section{background:white;padding:18px;margin-bottom:28px;border:1px solid #ddd}a{color:#266549}svg{max-width:100%}@media print{a,summary{display:none}section{break-before:page;border:0}}</style>
<h1>流行原唱 → 口琴谱</h1><p>六首原唱专辑试听，每首约 30 秒。下方谱面使用谱头给出的琴型、琴调和目标曲调，程序独立估计原调并选择移调。对照分数仅衡量定位乐句的音级顺序，允许整体移调、忽略八度和时值；不是整谱准确率。× 表示无对应孔位。</p>'''
    page+=''.join(pages)+f'<script src="{asset}"></script><script src="{os.path.dirname(asset)}/combined-score.js"></script>'+'''<script>document.querySelectorAll("section").forEach(s=>renderCombinedScore(s.querySelector(".score"),s.querySelector(".abc").textContent,{responsive:"resize",staffwidth:740}));</script></html>'''
    (args.output/"index.html").write_text(page)


if __name__=="__main__":
    main()
