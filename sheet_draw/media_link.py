"""Resolve public media pages and fetch audio for the local transcription UI."""
import io
import ipaddress
import json
import struct
import re
import socket
import time
import wave
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urljoin, urlsplit, urlunsplit

from .mp4_audio import extract_mp4

MIB = 1024**2
MAX_AUDIO = 96 * MIB
AGENT = "Mozilla/5.0 (compatible; SheetDraw/1.0)"


class PublicHTTP:
    """Pin the connection to a validated public IP on every origin, including redirects."""
    def __init__(self):
        self.pools = {}
        self.deadline = time.monotonic() + 180
        self.remaining = 128 * MIB

    def close(self):
        for pool in self.pools.values():
            pool.close()

    @contextmanager
    def open(self, url, headers=None, method="GET", data=None):
        import urllib3
        headers = {k:v for k,v in (headers or {}).items() if k.lower() in
                   {"user-agent", "referer", "range", "content-type", "accept", "origin"}}
        for _ in range(6):
            parts = urlsplit(url)
            if (parts.scheme not in ("http","https") or not parts.hostname or parts.username is not None
                    or parts.password is not None or len(url)>8192 or "\\" in url
                    or any(ord(c)<32 for c in url)):
                raise ValueError("请输入公开的 HTTP 或 HTTPS 链接。")
            host = parts.hostname.encode("idna").decode()
            port = parts.port or (443 if parts.scheme=="https" else 80)
            key = parts.scheme,host,port
            if key not in self.pools:
                ips = [ipaddress.ip_address(r[4][0]) for r in socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)]
                if not ips or any(not ip.is_global or ip.is_multicast or
                                  (getattr(ip,"ipv4_mapped",None) and not ip.ipv4_mapped.is_global) for ip in ips):
                    raise ValueError("仅支持公开互联网媒体地址。")
                ip = str(next((i for i in ips if i.version==4),ips[0]))
                self.pools[key] = (urllib3.HTTPSConnectionPool(ip,port,server_hostname=host,assert_hostname=host)
                                   if parts.scheme=="https" else urllib3.HTTPConnectionPool(ip,port))
            if time.monotonic()>self.deadline:
                raise ValueError("读取媒体超时，请使用独立音频链接。")
            path = urlunsplit(("","",quote(parts.path or "/",safe="/%:@!$&'()*+,;=-._~"),
                               quote(parts.query,safe="=&%/?+:,;@!$'()*-._~"),""))
            res = self.pools[key].urlopen(method,path,body=data,headers={
                "User-Agent":AGENT, **headers,"Host":parts.netloc,"Accept-Encoding":"identity"},
                redirect=False,retries=False,preload_content=False,timeout=urllib3.Timeout(connect=8,read=18))
            try:
                if res.status in (301,302,303,307,308) and res.headers.get("Location"):
                    url=urljoin(url,res.headers["Location"])
                    if res.status in (301,302,303):
                        method,data="GET",None
                    continue
                yield url,res
                return
            finally:
                res.close()
                res.release_conn()
        raise ValueError("媒体链接重定向过多。")

    def read(self,res,size):
        if time.monotonic()>self.deadline or self.remaining<=0:
            raise ValueError("媒体读取超过限额，请换用音频直链。")
        chunk=res.read(min(size,self.remaining),decode_content=False)
        self.remaining-=len(chunk)
        return chunk

    def body(self,res,limit):
        if res.status >= 400:
            raise ValueError(f"媒体网站拒绝或无法提供此内容（HTTP {res.status}）。")
        if int(res.headers.get("Content-Length") or 0)>limit:
            raise ValueError("媒体响应超过大小限制。")
        out=bytearray()
        while len(out)<=limit:
            chunk=self.read(res,min(65536,limit+1-len(out)))
            if not chunk:
                return bytes(out)
            out.extend(chunk)
        raise ValueError("媒体响应超过大小限制。")


class MediaPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.audio,self.video=[],[]
        self.embeds=[]
        self.parent=self.title=self.base=""
        self.in_title=False

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in ("audio","video"):
            self.parent=tag
        if tag in ("audio","video","source") and a.get("src"):
            if tag=="audio" or (tag=="source" and (self.parent=="audio" or (a.get("type") or "").startswith("audio/"))):
                self.audio.append(a["src"])
            elif self.parent=="video":
                self.video.append(a["src"])
        if tag=="meta" and a.get("content"):
            prop=a.get("property") or a.get("name")
            if prop in ("og:audio","og:audio:url","og:audio:secure_url"):
                self.audio.append(a["content"])
            if prop in ("og:video","og:video:url","og:video:secure_url"):
                self.video.append(a["content"])
        if tag=="iframe" and a.get("src"):
            embedded=urlsplit(a["src"])
            if embedded.hostname in ("www.youtube.com","www.youtube-nocookie.com","player.vimeo.com","player.bilibili.com"):
                self.embeds.append(a["src"])
        if tag=="base" and not self.base:
            self.base=a.get("href") or ""
        if tag=="title":
            self.in_title=True

    def handle_endtag(self,tag):
        if tag==self.parent:
            self.parent=""
        if tag=="title":
            self.in_title=False

    def handle_data(self,text):
        if self.in_title:
            self.title+=text


def platform_audio(http,url):
    from yt_dlp import YoutubeDL
    from yt_dlp.networking.common import Response
    from yt_dlp.networking.exceptions import HTTPError
    from yt_dlp.utils import DownloadError

    class PageReader(YoutubeDL):
        def urlopen(self,req):
            with http.open(req.url,dict(req.headers),req.method,req.data) as (final,res):
                if res.status>=400:
                    raise HTTPError(Response(io.BytesIO(),final,res.headers,res.status))
                if res.headers.get("Content-Type","").startswith(("video/","audio/")):
                    raise ValueError("页面未提供可单独获取的音轨。")
                return Response(io.BytesIO(http.body(res,5*MIB)),final,res.headers,res.status)

    try:
        with PageReader({"quiet":True,"no_warnings":True,"noplaylist":True,"cachedir":False,
                         "ignore_no_formats_error":True}) as ydl:
            info=ydl.extract_info(url,download=False,process=False)
    except DownloadError as exc:
        raise ValueError("无法解析该平台的视频，可能需要登录或网站限制了访问。请尝试音频直链。") from exc
    if not info or info.get("_type") in ("playlist","multi_video","url","url_transparent"):
        raise ValueError("请粘贴单个视频的播放页链接。")
    if info.get("is_live") or info.get("live_status") in ("is_live","is_upcoming"):
        raise ValueError("直播流不支持转谱。")
    if not info.get("duration") or not 0<float(info["duration"])<=300:
        raise ValueError("视频时长须可确认且不超过 5 分钟。")
    available=[f for f in info.get("formats",[]) if not f.get("has_drm") and f.get("url")]
    formats=[f for f in available if f.get("vcodec")=="none" and f.get("acodec")!="none"
             and f.get("protocol") in ("http","https")]
    separate=bool(formats)
    if not formats:
        # Progressive MP4 fallback is handled strictly by byte-range audio extraction.
        formats=[f for f in available if f.get("ext")=="mp4" and f.get("protocol") in ("http","https")
                 and f.get("acodec") not in (None,"none")]
    if not formats:
        raise ValueError("该视频没有公开的独立音轨，请提供音频链接或支持分段读取的 MP4 直链。")
    best=max(formats,key=lambda f: f.get("abr") or f.get("tbr") or 0)
    return best["url"],info.get("title") or "视频音轨",best.get("http_headers") or {"Referer":url},separate


def audio_wav(payload,limit):
    import av
    try:
        with av.open(io.BytesIO(payload),options={"protocol_whitelist":"pipe",
                     "format_whitelist":"mov,mp4,m4a,3gp,3g2,mj2,mp3,aac,wav,flac,ogg,matroska,webm,aiff"}) as media:
            if media.streams.video and any(not (v.disposition & av.stream.Disposition.attached_pic) for v in media.streams.video):
                raise ValueError("该资源包含视频画面，请提供独立音轨或 MP4 直链。")
            if not media.streams.audio:
                raise ValueError("没有找到音轨。")
            stream=media.streams.audio[0]
            if media.duration and media.duration/av.time_base>limit+.05:
                raise ValueError(f"媒体时长超过 {limit//60} 分钟。")
            rate=44100
            channels=1 if len(stream.codec_context.layout.channels)==1 else 2
            resampler=av.AudioResampler(format="s16",layout="mono" if channels==1 else "stereo",rate=rate)
            out=io.BytesIO()
            count=0
            with wave.open(out,"wb") as writer:
                writer.setparams((channels,2,rate,0,"NONE","not compressed"))
                def write(frame):
                    nonlocal count
                    for f in resampler.resample(frame):
                        count+=f.samples
                        if count>rate*limit:
                            raise ValueError(f"媒体时长超过 {limit//60} 分钟。")
                        writer.writeframesraw(f.to_ndarray().tobytes())
                for frame in media.decode(stream):
                    write(frame)
                write(None)
            if not count:
                raise ValueError("音轨为空。")
            return out.getvalue()
    except av.error.FFmpegError as exc:
        raise ValueError("音频解码失败，链接可能是受保护内容或不支持的媒体格式。") from exc


def resolve(http,url,audio_limit):
    title=""
    headers={}
    video=False
    independent=False
    for depth in range(4):
        parts=urlsplit(url)
        if parts.scheme not in ("http","https") or parts.username is not None or parts.password is not None:
            raise ValueError("请输入公开的 HTTP 或 HTTPS 链接。")
        host=(parts.hostname or "").lower()
        if host=="music.apple.com":
            ids=parse_qs(parts.query).get("i") or [parts.path.rsplit("/",1)[-1]]
            country=parts.path.strip("/").split("/")[0]
            if not ids[0].isdigit() or not re.fullmatch("[a-z]{2}",country):
                raise ValueError("请使用 Apple Music 的单曲链接。")
            with http.open(f"https://itunes.apple.com/lookup?id={ids[0]}&country={country}") as (_,res):
                tracks=json.loads(http.body(res,MIB)).get("results",[])
            track=next((t for t in tracks if str(t.get("trackId"))==ids[0] and t.get("previewUrl")),None)
            if not track:
                raise ValueError("该单曲没有公开试听音轨。")
            independent=True
            url,title=track["previewUrl"],f'{track["trackName"]} · {track["artistName"]}（试听）'
            continue
        if host in ("www.youtube-nocookie.com","player.vimeo.com","www.youtube.com","youtube.com","m.youtube.com","youtu.be","www.bilibili.com","bilibili.com","vimeo.com","www.vimeo.com"):
            url,title,headers,independent=platform_audio(http,url)
            video=True
        with http.open(url,{**headers,"Range":"bytes=0-7"}) as (final,res):
            if res.status>=400:
                raise ValueError(f"无法访问媒体（HTTP {res.status}）。")
            content=res.headers.get("Content-Type","").lower()
            head=http.read(res,8)
            match=re.fullmatch(r"bytes 0-7/(\d+)",res.headers.get("Content-Range",""))
            total=int(match[1]) if res.status==206 and match else None
        url=final
        host2=(urlsplit(url).hostname or "").lower()
        if host2!=host and host2 in ("www.bilibili.com","www.youtube.com","music.apple.com"):
            continue
        if "html" in content or head.lstrip().startswith(b"<"):
            with http.open(url,headers) as (_,res):
                page=MediaPage()
                page.feed(http.body(res,2*MIB).decode("utf-8","replace"))
            # Prefer a progressive audio source over alternate Ogg/video MIME variants.
            choices=sorted(page.audio,key=lambda src: not urlsplit(src).path.lower().endswith(
                (".mp3",".m4a",".aac",".wav",".flac"))) or page.video or page.embeds
            if not choices:
                raise ValueError("页面没有公开的音频或视频资源，请改用播放页或媒体直链。")
            title=title or page.title.strip()
            video=video or not page.audio
            headers={"Referer":url}
            url=urljoin(urljoin(url,page.base),choices[0])
            continue
        name=title or unquote(PurePosixPath(urlsplit(url).path).name) or "链接音轨"
        mp4=head[4:8] in (b"ftyp",b"moov",b"wide",b"free",b"mdat")
        if mp4 and total and not independent:
            def read(start,size):
                end=start+size-1
                with http.open(url,{**headers,"Range":f"bytes={start}-{end}"}) as (_,r):
                    if r.status!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{total}":
                        raise ValueError("视频服务器不支持精确分段读取。")
                    data=http.body(r,size)
                    if len(data)!=size:
                        raise ValueError("媒体分段不完整。")
                    return data
            try:
                payload,has_video=extract_mp4(read,total,300 if video else audio_limit)
            except (KeyError,IndexError,struct.error) as exc:
                raise ValueError("MP4 音轨索引无效。") from exc
            video=video or has_video
        else:
            if not independent and (content.startswith("video/") and not (head.startswith(b"OggS") and content.startswith("video/ogg")) or mp4 and not content.startswith("audio/")):
                raise ValueError("该视频不支持独立音轨或分段读取，不会下载完整视频。")
            sound_magic = (head.startswith((b"ID3",b"fLaC",b"OggS",b"RIFF",b"FORM"))
                           or len(head)>=2 and head[0]==255 and head[1]&224==224)
            if not independent and not (content.startswith("audio/") or sound_magic):
                raise ValueError("未找到可读取的独立音轨或可分段解析的 MP4。")
            with http.open(url,headers) as (_,r):
                payload=http.body(r,MAX_AUDIO)
        return audio_wav(payload,300 if video else audio_limit),name[:180],video
    raise ValueError("页面嵌套过多，请使用音频直链。")


def link_audio(url,audio_limit):
    from importlib.util import find_spec
    if not all(find_spec(name) for name in ("urllib3", "av", "yt_dlp")):
        raise ValueError("链接解析需要 Python 3.9 或以上版本及项目依赖，请重新安装项目。")
    import urllib3
    http=PublicHTTP()
    try:
        return resolve(http,url.strip(),audio_limit)
    except (OSError,urllib3.exceptions.HTTPError) as exc:
        raise ValueError("连接媒体网站失败，请检查链接或稍后重试。") from exc
    finally:
        http.close()
