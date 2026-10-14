"""電腦版測試框架:用與 App 相同的 FFmpeg 解碼/編碼參數處理影片片段,方便快速驗證演算法。

App 端(lib/engine/pipeline.dart)的流程是:FFmpeg 解碼成 bgr24 → OpenCV 處理 → FFmpeg 編碼。
這裡用同樣的參數重現,處理函式可替換,用來比較「不處理」與各種增強的畫質。

用法:python roundtrip.py 原片.mp4 輸出.mp4 模式 [起始秒數 秒數]
  模式:none(只解碼再編碼)、enhance(新版提高畫質)
"""
from __future__ import annotations

import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str) -> tuple[int, int, float, bool]:
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    line = next(l for l in info.splitlines() if "Video:" in l)
    import re
    w, h = map(int, re.search(r", (\d{3,5})x(\d{3,5})", line).groups())
    fps = float(re.search(r"([\d.]+) fps", line).group(1))
    rot = re.search(r"rotation of (-?[\d.]+)", info)
    if rot and round(abs(float(rot.group(1)))) % 180 == 90:
        w, h = h, w
    return w // 2 * 2, h // 2 * 2, fps, "arib-std-b67" in line


def run(src: str, dst: str, process, start: float, dur: float, crf: int = 12) -> None:
    w, h, fps, hdr = probe(src)
    m = "bt2020" if hdr else "bt709"
    dec = subprocess.Popen(
        [FFMPEG, "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", src, "-an",
         "-vf", f"scale={w}:{h}:in_color_matrix={m}:in_range=tv:flags=accurate_rnd+full_chroma_int+full_chroma_inp",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    codec = (["-vf", f"scale=out_color_matrix={m}:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p10le",
              "-c:v", "libx265", "-preset", "fast", "-crf", f"{crf}", "-tag:v", "hvc1",
              "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:log-level=error",
              "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc"]
             if hdr else
             ["-vf", f"scale=out_color_matrix={m}:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
              "-c:v", "libx264", "-crf", f"{crf}"])
    enc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-framerate", f"{fps}", "-i", "-", *codec, "-color_range", "tv", dst], stdin=subprocess.PIPE)
    n, t0, size = 0, time.time(), w * h * 3
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        enc.stdin.write(np.ascontiguousarray(process(frame)).tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"{n} 格 {w}x{h} 用時 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    src, dst, mode = sys.argv[1:4]
    start = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    dur = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    if mode == "none":
        fn = lambda f: f
    else:
        from algorithms import DetailEnhancer
        fn = DetailEnhancer()
    run(src, dst, fn, start, dur)
