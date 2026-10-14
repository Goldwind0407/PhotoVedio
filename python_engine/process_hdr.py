"""電腦版完整處理:與 App(lib/engine/pipeline.dart)相同的流程與演算法,用原生解析度輸出。

  1. FFmpeg 解碼縮小灰階影格 → 估計晃動 → 自動判斷是否需要防手震
  2. FFmpeg 解碼原尺寸 → (防手震)→ 提高畫質(DetailEnhancer)→ FFmpeg 編碼
  3. HDR 原片輸出 10-bit HEVC 並保留 HLG/BT.2020 標註;原音軌直接複製

用法:python process_hdr.py 原片.mp4 輸出.mp4 [--strong | --lock]
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from algorithms import DetailEnhancer
from algorithms import (RS_BANDS, STABILIZE_LEVELS, apply_stabilization, estimate_motion_bands,
                        lock_trajectory, rolling_shutter_offsets, smoothing_sigma, stabilization_zoom,
                        stabilize_path)
from roundtrip import FFMPEG, probe

ANALYSIS_LONG_SIDE = 1280
SHAKE_THRESHOLD = 0.4e-3  # 與 App 相同:高頻抖動 / 畫面長邊
LOCK = False  # --lock:三腳架鎖定模式(完全無晃動,裁切較多)
STRONG = False  # --strong:強力防手震(裁切較多)
ACC = "accurate_rnd+full_chroma_int+full_chroma_inp"


def shake_score(motions: np.ndarray, long_side: int) -> float:
    """軌跡偏離其移動平均(視窗 15 格)的平均距離 / 畫面長邊。與 App 的 shakeScore 相同。"""
    if len(motions) < 16:
        return 0.0
    traj = np.cumsum(motions[:, :2], axis=0)
    pad = np.pad(traj, ((7, 7), (0, 0)), mode="reflect")
    k = np.ones(15) / 15
    sm = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], axis=1)
    return float(np.sqrt(((traj - sm) ** 2).sum(axis=1)).mean() / long_side)


def analyze(src: str, w: int, h: int, matrix: str) -> tuple[np.ndarray, np.ndarray]:
    s = min(1.0, ANALYSIS_LONG_SIDE / max(w, h))
    aw, ah = max(2, round(w * s) // 2 * 2), max(2, round(h * s) // 2 * 2)
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", src, "-an",
                            "-vf", f"scale={aw}:{ah},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                           stdout=subprocess.PIPE)
    size, prev, out, res = aw * ah, None, [(0.0, 0.0, 0.0)], [np.zeros((RS_BANDS, 2))]
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        g = np.frombuffer(buf, np.uint8).reshape(ah, aw)
        if prev is not None:
            dx, dy, da, r = estimate_motion_bands(prev, g)
            out.append((dx * w / aw, dy * h / ah, da))
            res.append(r * [w / aw, h / ah])
        prev = g
    dec.wait()
    return np.array(out), np.array(res)


def main() -> int:
    src, dst = sys.argv[1:3]
    global LOCK, STRONG
    LOCK = "--lock" in sys.argv[3:]
    STRONG = LOCK or "--strong" in sys.argv[3:]
    w, h, fps, hdr = probe(src)
    m = "bt2020" if hdr else "bt709"
    t0 = time.time()

    motions, residuals = analyze(src, w, h, m)
    rs = rolling_shutter_offsets(residuals, fps / 2)  # 果凍效應修正
    score = shake_score(motions, max(w, h))
    shaky = score >= SHAKE_THRESHOLD
    corr, zoom, mode = None, 1.0, ""
    if shaky:
        locked = lock_trajectory(motions, w, h) if LOCK else None
        if locked is not None:
            corr, zoom, deg = locked
            mode = ["鎖定:固定鏡頭", "鎖定:等速運鏡", "鎖定:二次曲線運鏡", "鎖定:三次曲線運鏡"][deg]
        else:
            budget, rot_budget, max_zoom = STABILIZE_LEVELS["strong" if STRONG else "standard"]
            corr = stabilize_path(motions, w, h, budget, rot_budget, smoothing_sigma(fps))
            mode = "平滑"
        # 最終倍率要把果凍修正的條帶偏移也算進去,才不會露邊
        zoom = stabilization_zoom(corr, w, h, max_zoom=1.25 if locked is not None else max_zoom, rs=rs)
    print(f"晃動分數 {score:.2e}({'有晃動,套用防手震 ' + mode if shaky else '穩定,略過防手震'})  放大 {zoom:.3f}  "
          f"分析 {time.time() - t0:.0f}s", flush=True)

    codec = (["-vf", f"scale=out_color_matrix={m}:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p10le",
              "-c:v", "libx265", "-preset", "fast", "-crf", "14", "-tag:v", "hvc1",
              "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:log-level=error",
              "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc"]
             if hdr else
             ["-vf", f"scale=out_color_matrix={m}:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
              "-c:v", "libx264", "-crf", "14", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"])
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", src, "-an",
                            "-vf", f"scale={w}:{h}:in_color_matrix={m}:in_range=tv:flags={ACC}",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-framerate", f"{fps}", "-i", "-", "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
                            *codec, "-color_range", "tv", "-c:a", "copy", "-shortest",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    enhancer = DetailEnhancer()
    size, i, total = w * h * 3, 0, len(motions)
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        if corr is not None and i < len(corr):
            f = apply_stabilization(f, corr[i], zoom, rs[i])
        f = enhancer(f)
        enc.stdin.write(np.ascontiguousarray(f).tobytes())
        i += 1
        if i % 60 == 0:
            print(f"  {i}/{total} 格  {time.time() - t0:.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"完成 {i} 格,共 {time.time() - t0:.0f}s → {dst}", flush=True)
    return 0 if enc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
