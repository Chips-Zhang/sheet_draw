import io
import socket
import struct
from contextlib import contextmanager
from unittest.mock import patch

import av
import numpy as np
import pytest
import soundfile as sf

from sheet_draw.media_link import PublicHTTP, resolve
from sheet_draw.mp4_audio import boxes, child, extract_mp4


def video_file():
    out=io.BytesIO()
    with av.open(out,"w",format="mp4") as movie:
        video=movie.add_stream("mpeg4",rate=10)
        video.width=video.height=64
        audio=movie.add_stream("aac",rate=44100)
        audio.layout="mono"
        for i in range(10):
            frame=av.VideoFrame.from_ndarray(np.full((64,64,3),i*15,dtype=np.uint8),format="rgb24")
            for packet in video.encode(frame):
                movie.mux(packet)
            x=np.arange(4410)+i*4410
            samples=np.sin(2*np.pi*440*x/44100).astype(np.float32)[None]
            frame=av.AudioFrame.from_ndarray(samples,format="fltp",layout="mono")
            frame.sample_rate=44100
            for packet in audio.encode(frame):
                movie.mux(packet)
        for stream in (video,audio):
            for packet in stream.encode(None):
                movie.mux(packet)
    return out.getvalue()


def test_mp4_reads_audio_only_even_when_metadata_follows_video():
    data=video_file()
    ranges=[]
    def read(start,size):
        ranges.append((start,start+size))
        return data[start:start+size]
    payload,is_video=extract_mp4(read,len(data))
    assert is_video
    with av.open(io.BytesIO(data)) as src:
        video_packets=[(p.pos,p.pos+p.size) for p in src.demux(video=0) if p.size]
    assert video_packets
    assert all(end<=a or start>=b for start,end in ranges for a,b in video_packets)
    with av.open(io.BytesIO(payload)) as audio:
        assert not audio.streams.video
        frames=list(audio.decode(audio=0))
        signal=np.concatenate([f.to_ndarray().ravel() for f in frames])
        assert 430 < np.argmax(abs(np.fft.rfft(signal)))*44100/len(signal) < 450


def test_video_duration_rejected_before_media_bytes():
    data=bytearray(video_file())
    header=data.index(b"mvhd")+4
    struct.pack_into(">I",data,header+16,301*struct.unpack_from(">I",data,header+12)[0])
    mdat=data.index(b"mdat")+4
    end=mdat-8+struct.unpack_from(">I",data,mdat-8)[0]
    def read(start,size):
        assert start+size<=mdat or start>=end
        return bytes(data[start:start+size])
    with pytest.raises(ValueError,match="5 分钟"):
        extract_mp4(read,len(data))


def test_audio_element_with_relative_source():
    out=io.BytesIO()
    sf.write(out,np.sin(np.arange(22050)*.1),22050,format="WAV")
    items={"https://example.org/page":(b'<title>Solo</title><audio><source src="/solo.wav"></audio>',"text/html"),
           "https://example.org/solo.wav":(out.getvalue(),"audio/wav")}
    class MemoryHTTP:
        @contextmanager
        def open(self,url,headers=None):
            body,kind=items[url]
            r=io.BytesIO(body)
            r.status=200
            r.headers={"Content-Type":kind}
            yield url,r
        def read(self,r,size):
            return r.read(size)
        def body(self,r,limit):
            return r.read(limit)
    wav,title,is_video=resolve(MemoryHTTP(),"https://example.org/page",120)
    assert title=="Solo" and not is_video
    assert sf.info(io.BytesIO(wav)).duration==pytest.approx(1.,abs=.002)


def test_private_destinations_rejected_before_connection():
    http=PublicHTTP()
    with patch("socket.getaddrinfo",return_value=[(socket.AF_INET,1,6,"",("127.0.0.1",80))]):
        with pytest.raises(ValueError,match="公开"), http.open("http://example.com/private"):
            pass
