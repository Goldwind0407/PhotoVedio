"""畫質比較:把原片與成品的同一格用播放器的方式顯示(HDR 先色調映射),比較 PSNR 與細節量。

用法:python quality.py 原片.mp4 原片秒數 成品.mp4 成品秒數 裁切圖.jpg
"""
from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np

from verify import FFMPEG, _is_hdr


def frame(path: str, t: float) -> np.ndarray:
    vf = ("zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
          "zscale=t=bt709:m=bt709:r=pc,format=bgr24") if _is_hdr(path) else "format=bgr24"
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    import re
    line = next(l for l in info.splitlines() if "Video:" in l)
    w, h = map(int, re.search(r", (\d{3,5})x(\d{3,5})", line).groups())
    rot = re.search(r"rotation of (-?[\d.]+)", info)
    if rot and round(abs(float(rot.group(1)))) % 180 == 90:
        w, h = h, w
    raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t}", "-i", path, "-frames:v", "1",
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8)[: w * h * 3].reshape(h, w, 3)


def detail(img: np.ndarray) -> float:
    """細節量:亮度 Laplacian 的變異數(越高代表越多細紋與銳利邊緣)。"""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main() -> int:
    a = frame(sys.argv[1], float(sys.argv[2]))
    b = frame(sys.argv[3], float(sys.argv[4]))
    if a.shape != b.shape:
        print(f"尺寸不同:原片 {a.shape[1]}x{a.shape[0]},成品 {b.shape[1]}x{b.shape[0]}(成品放大比較)")
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    psnr = cv2.PSNR(a, b)
    h, w = a.shape[:2]
    crops = [(int(h * 0.35), int(w * 0.55)), (int(h * 0.15), int(w * 0.3)), (int(h * 0.6), int(w * 0.2))]
    print(f"PSNR {psnr:.2f} dB   全畫面細節 原片 {detail(a):.0f} → 成品 {detail(b):.0f}")
    tiles = []
    for y, x in crops:
        ca, cb = a[y:y + 400, x:x + 400], b[y:y + 400, x:x + 400]
        print(f"  區塊({x},{y}) 細節 {detail(ca):.0f} → {detail(cb):.0f}")
        tiles.append(np.hstack([ca, cb]))
    cv2.imwrite(sys.argv[5], np.vstack(tiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
