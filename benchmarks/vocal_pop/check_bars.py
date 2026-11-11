"""Check Blue Lotus measures 17–20 against the independently specified staff pitches and beats."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from sheet_draw.harmonica import make_harp
from sheet_draw.audio_utils import frequency_to_note

ROOT=Path(__file__).parent
# From the score: an eighth rest, six repeated E5 attacks, F#5,
# a re-articulated whole F#5; in m.19 the last E5 is tied through m.20.
BARS=[(.5+i/2, .5, 76) for i in range(6)] + [(3.5,.5,78),(4.,4.,78)]
BARS += [(8.5+i/2,.5 if i<6 else 4.5,p) for i,p in enumerate([78,78,78,81,78,78,76])]


def check(record):
    # Set one origin from the isolated A5 attack at m.19 beat 3 in the vocal track.
    # This landmark is excluded from both predictions and the reference for scoring.
    origin=7.1672 - 10 * 60/90
    shift=record["auto_shift"]
    harp=make_harp("paddy","A")
    ref=[(origin+b*60/90,d*60/90,p) for b,d,p in BARS if b!=10]
    result={}
    for mode in ["mix","vocals","written"]:
        rows=record["written"] if mode=="written" else record["events"][mode]
        pitch_shift=0 if mode=="written" else shift
        est=[(t,d,p+pitch_shift) for t,d,p,_ in rows
             if origin<=t<origin+16*60/90 and abs(t-7.1672)>.1]
        match=np.zeros((len(ref),len(est)),dtype=int)
        strict=match.copy()
        for i,(a,d,p) in enumerate(ref):
            for j,(t,length,m) in enumerate(est):
                match[i,j]=abs(a-t)<=.1 and p==m
                strict[i,j]=match[i,j] and abs(a+d-t-length)<=max(.1,.2*d)
        hits=[]
        for matrix in (match,strict):
            i,j=linear_sum_assignment(-matrix)
            hits.append(int(matrix[i,j].sum()))
        result[mode]=dict(reference=len(ref),predicted=len(est),pitch_onset_hits=hits[0],
                          pitch_onset_f1=round(2*hits[0]/(len(ref)+len(est)),3),
                          with_offset_hits=hits[1],with_offset_f1=round(2*hits[1]/(len(ref)+len(est)),3))
    fingers={}
    for p in [76,78,81]:
        name=frequency_to_note(440*2**((p-69)/12))
        pos=harp.note_to_hole(name)
        fingers[name]=[pos.hole,pos.action]
    return dict(tempo=90,origin_seconds=origin,excluded_landmark_seconds=7.1672,
                arranged_shift=shift,fingerings=fingers,results=result)


if __name__=="__main__":
    record=json.loads((ROOT/"results/report.json").read_text())["蓝莲花"]
    out=json.dumps(check(record),ensure_ascii=False,indent=2)
    (ROOT/"results/blue_bars.json").write_text(out+"\n")
    print(out)
