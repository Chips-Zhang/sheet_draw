"""Locate an independently read numbered-score phrase, then compare two singing pipelines."""
import json
import re
from pathlib import Path

import numpy as np
from numba import njit

TONICS = {"C": 0, "Eb": 3, "G": 7, "A": 9}


def reference(text, tonic):
    degrees = [0, 2, 4, 5, 7, 9, 11]
    return np.array([(degrees[int(d)-1] + TONICS[tonic] - bool(flat)) % 12
                     for flat, d in re.findall(r"(b?)([1-7])", text)], dtype=np.int64)


@njit
def locate(ref, pred):
    # Smith-Waterman; return both coordinate ranges, including indels.
    scores = np.zeros((len(ref)+1, len(pred)+1), np.int32)
    best = end_i = end_j = 0
    for i in range(1, len(ref)+1):
        for j in range(1, len(pred)+1):
            s = max(0, scores[i-1,j-1] + (3 if ref[i-1] == pred[j-1] else -3),
                    scores[i-1,j]-2, scores[i,j-1]-2)
            scores[i,j] = s
            if s > best:
                best, end_i, end_j = s,i,j
    i,j = end_i,end_j
    while scores[i,j] > 0:
        if scores[i,j] == scores[i-1,j-1] + (3 if ref[i-1] == pred[j-1] else -3):
            i,j = i-1,j-1
        elif scores[i,j] == scores[i-1,j]-2:
            i -= 1
        else:
            j -= 1
    return best,i,end_i,j,end_j


def best_match(refs, notes):
    best = None
    for name, ref in refs.items():
        for shift in range(-6, 6):
            score,i,k,j,l = locate(ref, (notes + shift) % 12)
            if best is None or score > best[0]:
                best = score,name,shift,i,k,j,l
    return best


def distance(ref, notes):
    # Minimal edit script: (cost, extra, missing, wrong).
    prev = [(i,i,0,0) for i in range(len(notes)+1)]
    for i, desired in enumerate(ref,1):
        row = [(i,0,i,0)]
        for j,note in enumerate(notes,1):
            d,x,m,w = prev[j-1]; diff = int(desired != note)
            a,b,c,e = prev[j]; f,g,h,k = row[-1]
            row.append(min([(d+diff,x,m,w+diff),(a+1,b,c+1,e),(f+1,g+1,h,k)],key=lambda s:s[0]))
        prev = row
    return prev[-1]


def compare(info, prediction):
    refs = {k: reference(v,info["tonic"]) for k,v in info["phrases"].items()}
    anchors = prediction["tracker"]
    pitches = np.array([x[2] for x in anchors],dtype=np.int64)
    score,name,shift,i,k,j,l = best_match(refs,pitches)
    rng = np.random.default_rng(419)
    null = [best_match(refs,rng.permutation(pitches))[0] for _ in range(99)]
    p = (1+sum(x >= score for x in null))/100
    t0 = anchors[j][0] - .04
    t1 = anchors[l-1][0] + min(.08, anchors[l-1][1]/2)
    result = dict(phrase=name, degree_events=[int(i)+1,int(k)], reference_count=int(k-i),
                  window_seconds=[round(max(0,t0),3),round(t1,3)], pitch_class_shift=shift,
                  shuffle_p=p, supported=bool(p<=.05 and k-i>=12))
    for mode in ("mix","vocals"):
        window=[(n[2]+shift)%12 for n in prediction[mode] if t0<=n[0]<=t1]
        d,extra,miss,wrong=distance(refs[name][i:k],window)
        result[mode]=dict(notes=len(window),extra=extra,missing=miss,wrong_pitch_class=wrong,
                          similarity=round(1-d/max(k-i,len(window)),3))
    return result
