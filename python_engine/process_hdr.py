"""電腦版完整處理:與 App(lib/engine/pipeline.dart)相同的流程與演算法,用原生解析度輸出。

  1. FFmpeg 解碼縮小灰階影格 → 估計晃動 → 自動判斷是否需要防手震
  2. FFmpeg 解碼原尺寸 → (防手震)→ 提高畫質(DetailEnhancer)→ FFmpeg 編碼
  3. HDR 原片輸出 10-bit HEVC 並保留 HLG/BT.2020 標註;原音軌直接複製

用法:python process_hdr.py 原片.mp4 輸出.mp4 [--strong | --extreme | --lock] [--no-fill] [--sr [長邊像素]]
  補邊(預設開啟):防手震露出的邊緣用前後格的真實畫面補,只需少量放大;--no-fill 改回放大到完全蓋住。
  --sr:AI 超解析度放大(預設輸出長邊 3840,即 4K),取代 DetailEnhancer。需要 onnxruntime 或 OpenCV。
"""
from __future__ import annotations

import subprocess
import sys
import time

import cv2
import numpy as np

from algorithms import DetailEnhancer, compression_level
from algorithms import (FILL_LEVELS, RS_BANDS, STABILIZE_LEVELS, apply_stabilization, estimate_motion_bands, fill_borders, fill_zoom,
                        lock_trajectory, rolling_shutter_offsets, smoothing_sigma, stabilization_zoom,
                        stabilize_path)
from roundtrip import FFMPEG, bitrate_info, probe

ANALYSIS_LONG_SIDE = 1280
SHAKE_THRESHOLD = 0.4e-3  # 與 App 相同:高頻抖動 / 畫面長邊
LOCK = False  # --lock:三腳架鎖定模式(完全無晃動,裁切較多)
LEVEL = "standard"  # --strong / --extreme:更強的防手震(裁切較多)
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
    global LOCK, LEVEL
    flags = sys.argv[3:]
    sr_long = 0
    if "--sr" in flags:
        nxt = flags[flags.index("--sr") + 1:flags.index("--sr") + 2]
        sr_long = int(nxt[0]) if nxt and nxt[0].isdigit() else 3840
    LOCK = "--lock" in flags
    fill = "--no-fill" not in flags
    # 鎖定需要的裁切太多時改用超強平滑
    LEVEL = "extreme" if LOCK or "--extreme" in flags else ("strong" if "--strong" in flags else "standard")
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
            budget, rot_budget, max_zoom = STABILIZE_LEVELS[LEVEL]
            if fill:
                budget, rot_budget = FILL_LEVELS[LEVEL]
                max_zoom = 3.0  # 補邊時只用一部分放大,上限不卡
            corr = stabilize_path(motions, w, h, budget, rot_budget, smoothing_sigma(fps))
            mode = "平滑"
        # 最終倍率要把果凍修正的條帶偏移也算進去,才不會露邊
        zoom = stabilization_zoom(corr, w, h, max_zoom=1.25 if locked is not None else max_zoom, rs=rs)
        if fill:
            print(f"補邊:放大 {zoom:.3f} → {fill_zoom(zoom):.3f}", flush=True)
            zoom = fill_zoom(zoom)
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
    sr = None
    ow, oh = w, h
    if sr_long:
        from superres import SuperResolver, target_size
        sr = SuperResolver()
        ow, oh = target_size(w, h, sr_long)
        print(f"AI 超解析度:{w}x{h} → {ow}x{oh}({sr.backend})", flush=True)
    enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
                            "-framerate", f"{fps}", "-i", "-", "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
                            *codec, "-color_range", "tv", "-c:a", "copy", "-shortest",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    bitrate, hevc = bitrate_info(src)
    compression = compression_level(bitrate, w, h, fps, hevc)
    enhancer = None if sr else DetailEnhancer(compression=compression)
    if enhancer:
        print(f"壓縮程度 {compression:.2f}(位元率 {bitrate / 1e6:.1f} Mbps)", flush=True)
    size, i, total = w * h * 3, 0, len(motions)
    # 補邊要用到前後 win 格:先多讀 win 格進緩衝區。4K 時減少格數,控制記憶體(每格 BGR 約 25 MB)。
    use_fill = fill and corr is not None
    win = max(2, min(15, int(400e6 / (w * h * 3) / 2))) if use_fill else 0
    traj = np.cumsum(motions, axis=0)
    path = traj + corr if corr is not None else None
    buf_frames, eof, unfilled = {}, False, 0

    def read_more(upto):
        nonlocal eof
        while not eof and (not buf_frames or max(buf_frames) < upto):
            raw = dec.stdout.read(size)
            if len(raw) < size:
                eof = True
                break
            buf_frames[(max(buf_frames) + 1) if buf_frames else 0] = np.frombuffer(raw, np.uint8).reshape(h, w, 3)

    while True:
        read_more(i + win)
        if i not in buf_frames:
            break
        for j in [j for j in buf_frames if j < i - win]:
            del buf_frames[j]
        f = buf_frames[i]
        if corr is not None and i < len(corr):
            if use_fill:
                f, cov = apply_stabilization(f, corr[i], zoom, rs[i], coverage=True)
                unfilled += fill_borders(f, cov, i, buf_frames, path, traj, zoom)
            else:
                f = apply_stabilization(f, corr[i], zoom, rs[i])
        if sr is not None:
            # AI 放大 4 倍後縮到目標尺寸(INTER_AREA 縮小不會產生鋸齒)
            f = sr(f)
            if f.shape[1] != ow or f.shape[0] != oh:
                f = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
        else:
            f = enhancer(f)
        enc.stdin.write(np.ascontiguousarray(f).tobytes())
        i += 1
        if i % (5 if sr else 60) == 0:
            print(f"  {i}/{total} 格  {time.time() - t0:.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    if use_fill:
        print(f"補邊:補不到的像素平均每格 {unfilled / max(i, 1) / (w * h) * 100:.2f}%(以延伸邊緣處理)", flush=True)
    print(f"完成 {i} 格,共 {time.time() - t0:.0f}s → {dst}", flush=True)
    return 0 if enc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
