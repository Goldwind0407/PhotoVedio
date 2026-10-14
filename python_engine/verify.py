"""成品驗證:用與播放器相同的方式顯示原片與成品,再比較亮度、對比、飽和度並輸出對比圖。

HDR 影片(HLG/PQ)直接用 OpenCV 讀出的是編碼值,不是播放器實際顯示的畫面。
這裡用 FFmpeg 的 zscale + tonemap 把兩者用同一套 HDR→SDR 映射轉成畫面再比較;
SDR 影片則直接轉成 BT.709 畫面。

用法:python verify.py 原片.mp4 成品.mp4 對比圖.jpg [秒數 ...]
"""
from __future__ import annotations

import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 360, 640


def _is_hdr(path: str) -> bool:
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return "arib-std-b67" in info or "smpte2084" in info


def grab(path: str, t: float) -> np.ndarray:
    """取第 t 秒的畫面,轉成播放器在 SDR 螢幕上會顯示的樣子(BGR, W×H)。"""
    if _is_hdr(path):
        vf = ("zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,"
              "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=pc,format=bgr24")
    else:
        vf = "zscale=t=bt709:m=bt709:r=pc,format=bgr24"
    vf += f",scale={W}:{H}:flags=area"
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{t}", "-i", path, "-frames:v", "1",
         "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8)[: W * H * 3].reshape(H, W, 3)


def stats(img: np.ndarray) -> tuple[float, float, float, float, float]:
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(float).mean()
    return l.mean(), float(np.median(l)), float(np.percentile(l, 5)), float(np.percentile(l, 95)), s


def main() -> int:
    src, out, jpg = sys.argv[1:4]
    times = [float(t) for t in sys.argv[4:]] or [0.5, 2.0, 4.0]
    print(f"原片 HDR={_is_hdr(src)}  成品 HDR={_is_hdr(out)}")
    print("秒數   亮度平均      中位數      P5/P95(暗/亮)        飽和度")
    rows = []
    for t in times:
        a, b = grab(src, t), grab(out, t)
        sa, sb = stats(a), stats(b)
        print(f"{t:4.1f}  {sa[0]:5.1f}→{sb[0]:5.1f}  {sa[1]:4.0f}→{sb[1]:4.0f}  "
              f"{sa[2]:3.0f}/{sa[3]:3.0f}→{sb[2]:3.0f}/{sb[3]:3.0f}  {sa[4]:5.1f}→{sb[4]:5.1f}")
        rows.append(np.hstack([a, b]))
    cv2.imwrite(jpg, np.hstack(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
