"""影片串流處理:讀影格 → 防手震 → 除霧 → 提高畫質 → 寫檔(有 ffmpeg 時併回原音軌)。"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from algorithms import (Dehazer, DetailEnhancer, ToneAnchor, apply_stabilization, estimate_motion,
                        smooth_trajectory)


@dataclass
class Options:
    enhance: bool = False       # 提高畫質:雜訊感知、邊緣感知的亮度細節增強
    dehaze: bool = False
    dehaze_strength: float = 0.6
    tone_anchor: bool = True    # 最後把亮度/色彩拉回原片
    stabilize: bool = False
    smooth_radius: int = 15     # 防手震平滑視窗(影格數)
    zoom: float = 1.05          # 防手震裁切放大,蓋住邊緣


Progress = Callable[[str, float], None]


ANALYSIS_LONG_SIDE = 1280  # 晃動分析用的長邊像素數(約 720p),輸出仍用原解析度


def _analysis_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """縮小成灰階供光流使用,回傳 (灰階圖, 縮放比例)。4K 影片可因此快數倍。"""
    scale = min(1.0, ANALYSIS_LONG_SIDE / max(frame.shape[:2]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _analyze_motion(path: str, report: Progress) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("無法讀取影片")
    prev_gray, scale = _analysis_gray(prev)
    motions = [(0.0, 0.0, 0.0)]  # 第一格沒有前一格
    i = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray, _ = _analysis_gray(frame)
        dx, dy, da = estimate_motion(prev_gray, gray)
        # 位移是縮小圖上的像素,放大回原解析度;旋轉角與尺度無關。
        motions.append((dx / scale, dy / scale, da))
        prev_gray = gray
        i += 1
        if i % 10 == 0:
            report("分析晃動", min(i / total, 1.0))
    cap.release()
    return np.array(motions, dtype=np.float64)


def process_video(src: str, dst: str, opt: Options,
                  report: Progress = lambda stage, p: None) -> None:
    corrections = None
    if opt.stabilize:
        motions = _analyze_motion(src, report)
        corrections = smooth_trajectory(motions, opt.smooth_radius)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (w // 2 * 2, h // 2 * 2)  # 編碼器要求偶數

    have_ffmpeg = shutil.which("ffmpeg") is not None
    tmp_video = Path(tempfile.mkdtemp()) / "video_only.mp4" if have_ffmpeg else Path(dst)
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError("無法建立輸出檔")

    dehazer = Dehazer(opt.dehaze_strength) if opt.dehaze else None
    # 新版畫質增強不改變色調;只有除霧會改變亮度分佈,才需要錨定回原片。
    anchor = ToneAnchor() if opt.tone_anchor and opt.dehaze else None
    enhancer = DetailEnhancer() if opt.enhance else None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if corrections is not None and i < len(corrections):
            frame = apply_stabilization(frame, corrections[i], opt.zoom)
        ref = frame if anchor else None
        if dehazer:
            frame = dehazer(frame)
        if enhancer:
            frame = enhancer(frame)
        if anchor and ref is not None:
            if ref.shape != frame.shape:
                ref = cv2.resize(ref, frame.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            frame = anchor(ref, frame)
        if frame.shape[1::-1] != out_size:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_LANCZOS4)
        writer.write(frame)
        i += 1
        if i % 5 == 0:
            report("輸出影片", min(i / total, 1.0))
    cap.release()
    writer.release()

    if have_ffmpeg:
        # 影像轉 H.264(手機相簿相容),音軌從原檔複製(沒有音軌也不會失敗)。
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", src,
             "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", dst],
            check=True)
        shutil.rmtree(tmp_video.parent, ignore_errors=True)
    report("完成", 1.0)
