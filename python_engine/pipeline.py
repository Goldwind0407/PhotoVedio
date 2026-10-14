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

from algorithms import (RS_BANDS, STABILIZE_LEVELS, Dehazer, DetailEnhancer, ToneAnchor,
                        apply_stabilization, estimate_motion_bands, rolling_shutter_offsets,
                        smoothing_sigma, stabilization_zoom, stabilize_path)


@dataclass
class Options:
    enhance: bool = False       # 提高畫質:雜訊感知、邊緣感知的亮度細節增強
    enhance_level: str = "medium"  # low / medium / high
    dehaze: bool = False
    dehaze_strength: float = 0.6
    tone_anchor: bool = True    # 最後把亮度/色彩拉回原片
    stabilize: bool = False
    stabilize_level: str = "standard"  # standard / strong / extreme(裁切越多越穩)
    zoom: float = 0             # 防手震裁切放大;0 = 依修正量自動計算


Progress = Callable[[str, float], None]


ANALYSIS_LONG_SIDE = 1280  # 晃動分析用的長邊像素數(約 720p),輸出仍用原解析度


def _analysis_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """縮小成灰階供光流使用,回傳 (灰階圖, 縮放比例)。4K 影片可因此快數倍。"""
    scale = min(1.0, ANALYSIS_LONG_SIDE / max(frame.shape[:2]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _analyze_motion(path: str, report: Progress) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (每格運動 (n, 3), 每格各條帶的果凍殘差 (n, bands, 2)),都已換算回原解析度。"""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("無法讀取影片")
    prev_gray, scale = _analysis_gray(prev)
    motions = [(0.0, 0.0, 0.0)]  # 第一格沒有前一格
    residuals = [np.zeros((RS_BANDS, 2))]
    i = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray, _ = _analysis_gray(frame)
        dx, dy, da, res = estimate_motion_bands(prev_gray, gray)
        # 位移是縮小圖上的像素,放大回原解析度;旋轉角與尺度無關。
        motions.append((dx / scale, dy / scale, da))
        residuals.append(res / scale)
        prev_gray = gray
        i += 1
        if i % 10 == 0:
            report("分析晃動", min(i / total, 1.0))
    cap.release()
    return np.array(motions, dtype=np.float64), np.array(residuals)


def process_video(src: str, dst: str, opt: Options,
                  report: Progress = lambda stage, p: None) -> None:
    corrections = None
    if opt.stabilize:
        motions, residuals = _analyze_motion(src, report)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (w // 2 * 2, h // 2 * 2)  # 編碼器要求偶數
    zoom = opt.zoom
    if opt.stabilize:
        budget, rot_budget, max_zoom = STABILIZE_LEVELS[opt.stabilize_level]
        corrections = stabilize_path(motions, w, h, budget, rot_budget, smoothing_sigma(fps))
        rs = rolling_shutter_offsets(residuals, fps / 2)
        zoom = zoom or stabilization_zoom(corrections, w, h, max_zoom=max_zoom, rs=rs)

    have_ffmpeg = shutil.which("ffmpeg") is not None
    tmp_video = Path(tempfile.mkdtemp()) / "video_only.mp4" if have_ffmpeg else Path(dst)
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError("無法建立輸出檔")

    dehazer = Dehazer(opt.dehaze_strength) if opt.dehaze else None
    # 新版畫質增強不改變色調;只有除霧會改變亮度分佈,才需要錨定回原片。
    anchor = ToneAnchor() if opt.tone_anchor and opt.dehaze else None
    enhancer = DetailEnhancer.level(opt.enhance_level) if opt.enhance else None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if corrections is not None and i < len(corrections):
            frame = apply_stabilization(frame, corrections[i], zoom, rs[i])
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
