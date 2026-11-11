"""Read an ordinary (non-fragmented) MP4's audio track using exact HTTP ranges."""
import struct


def u32(data, offset):
    return struct.unpack_from(">I", data, offset)[0]


def boxes(data):
    pos = 0
    while pos < len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif not size:
            size = len(data) - pos
        if size < header or pos + size > len(data):
            raise ValueError("MP4 元数据不完整。")
        yield kind, data[pos + header:pos + size]
        pos += size


def child(data, *path):
    for name in path:
        data = next((body for kind, body in boxes(data) if kind == name), None)
        if data is None:
            raise ValueError("该 MP4 缺少可独立读取的音轨索引。")
    return data


def box(kind, data):
    return struct.pack(">I4s", len(data) + 8, kind) + data


def media_duration(header):
    pos = 20 if header[0] == 1 else 12
    scale = u32(header, pos)
    ticks = struct.unpack_from(">Q" if header[0] == 1 else ">I", header, pos + 4)[0]
    if not ticks or not scale:
        raise ValueError("无法确认视频时长。")
    return ticks / scale


def extract_mp4(read, total, limit=300):
    """read(start, size) must return exactly those bytes; never read video chunks."""
    pos = 0
    while pos + 8 <= total:
        size, kind = struct.unpack(">I4s", read(pos, 8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", read(pos+8, 8))[0]
            header = 16
        if not size:
            size = total - pos
        if size < header or size > total-pos:
            raise ValueError("MP4 容器索引无效。")
        if kind == b"moov":
            if size > 12*1024**2:
                raise ValueError("MP4 索引过大。")
            moov = read(pos+header, size-header)
            break
        pos += size
    else:
        raise ValueError("找不到 MP4 的音轨索引。")

    tracks = [b for k,b in boxes(moov) if k == b"trak"]
    is_video = any(child(t,b"mdia",b"hdlr")[8:12] == b"vide" for t in tracks)
    duration = media_duration(child(moov,b"mvhd"))
    if duration > (300 if is_video else limit):
        raise ValueError("视频不能超过 5 分钟。" if is_video else "音频超过当前模式的时长上限。")
    audio = next((t for t in tracks if child(t,b"mdia",b"hdlr")[8:12] == b"soun"),None)
    if not audio:
        raise ValueError("视频不包含音轨。")
    table = dict(boxes(child(audio,b"mdia",b"minf",b"stbl")))
    if b"stsz" not in table or not (b"stco" in table or b"co64" in table):
        raise ValueError("该视频需要独立音轨链接。")
    wide = b"co64" in table
    offsets = table[b"co64" if wide else b"stco"]
    addresses = [v for v, in struct.iter_unpack(">Q" if wide else ">I",offsets[8:])]
    if not addresses or len(addresses) != u32(offsets,4):
        raise ValueError("不支持该分片 MP4，请使用视频页面链接。")
    mapping = list(struct.iter_unpack(">III",child(audio,b"mdia",b"minf",b"stbl",b"stsc")[8:]))
    sizes = table[b"stsz"]
    uniform,count = struct.unpack_from(">II",sizes,4)
    if not mapping or mapping[0][0]!=1 or (not uniform and len(sizes)!=12+4*count):
        raise ValueError("音轨索引无效。")
    ftyp = box(b"ftyp", b"M4A \x00\x00\x00\x00M4A isom")
    runs, new_offsets = [], []
    index = group = length = 0
    for n,offset in enumerate(addresses,1):
        if group+1<len(mapping) and n==mapping[group+1][0]:
            group += 1
        samples = mapping[group][1]
        if index+samples > count:
            raise ValueError("音轨样本数量无效。")
        width = uniform*samples if uniform else sum(v for v, in struct.iter_unpack(
            ">I",sizes[12+4*index:12+4*(index+samples)]))
        if offset+width > total or length+width > 96*1024**2:
            raise ValueError("音轨过大或索引无效。")
        new_offsets.append(len(ftyp)+8+length)
        if runs and runs[-1][0]+runs[-1][1]==offset:
            runs[-1][1]+=width
        elif width:
            runs.append([offset,width])
        index += samples
        length += width
    if index != count:
        raise ValueError("音轨样本缺失。")
    def rebuild(data):
        result = []
        for kind,body in boxes(data):
            if kind in (b"stco",b"co64"):
                kind=b"stco"
                body=offsets[:8]+b"".join(struct.pack(">I",v) for v in new_offsets)
            elif kind in (b"mdia",b"minf",b"stbl"):
                body=rebuild(body)
            result.append(box(kind,body))
        return b"".join(result)
    payload = b"".join(read(start,size) for start,size in runs)
    return ftyp+box(b"mdat",payload)+box(b"moov",box(b"mvhd",child(moov,b"mvhd"))+box(b"trak",rebuild(audio))),is_video
