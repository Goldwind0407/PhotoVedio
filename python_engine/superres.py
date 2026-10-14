"""AI 超解析度(Real-ESRGAN realesr-general-x4v3,4 倍放大)。

模型很小(120 萬參數),但每個輸入像素約要 240 萬次浮點運算:592×1280 的影格約 1.8 TFLOP。
有 onnxruntime 時優先用它(會自動選 CUDA / DirectML / CoreML 等 GPU),否則用 OpenCV dnn(只有 CPU)。
實測 4 核心 CPU:onnxruntime 每格 10.8 秒、OpenCV 17 秒;有獨立顯示卡時快數十倍。

大畫面分塊處理:每塊多取 [OVERLAP] 像素(大於網路 34 層 3×3 卷積的作用範圍),只寫回中間,
結果與整張處理相同,記憶體只跟塊大小有關。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "assets" / "models" / "realesr-general-x4v3.onnx"
SCALE = 4
TILE = 320
OVERLAP = 40


class SuperResolver:
    def __init__(self, model: str | Path = MODEL, tile: int = TILE):
        self.tile = tile
        self._ort = None
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prefer = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider",
                      "CPUExecutionProvider"]
            providers = [p for p in prefer if p in avail]
            self._ort = ort.InferenceSession(str(model), providers=providers)
            self.backend = f"onnxruntime({self._ort.get_providers()[0]})"
        except ImportError:
            self._net = cv2.dnn.readNetFromONNX(str(model))
            self.backend = "opencv-dnn(CPU)"

    def _run(self, bgr: np.ndarray) -> np.ndarray:
        x = cv2.dnn.blobFromImage(bgr, 1 / 255.0, swapRB=True)  # 1×3×H×W,RGB 0~1
        if self._ort is not None:
            y = self._ort.run(None, {"input": x})[0]
        else:
            self._net.setInput(x)
            y = self._net.forward()
        return y[0].transpose(1, 2, 0)[..., ::-1]  # H×W×3 BGR

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """8UC3 BGR → 4 倍大的 8UC3 BGR。"""
        h, w = bgr.shape[:2]
        out = np.empty((h * SCALE, w * SCALE, 3), np.float32)
        t = self.tile
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(h, y0 + t), min(w, x0 + t)
                ya, xa = max(0, y0 - OVERLAP), max(0, x0 - OVERLAP)
                yb, xb = min(h, y1 + OVERLAP), min(w, x1 + OVERLAP)
                r = self._run(bgr[ya:yb, xa:xb])
                out[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = \
                    r[(y0 - ya) * SCALE:(y1 - ya) * SCALE, (x0 - xa) * SCALE:(x1 - xa) * SCALE]
        return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def target_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """超解析度後的輸出尺寸:長邊為 long_side(最多放大 4 倍),取偶數。"""
    s = min(SCALE, long_side / max(w, h))
    return max(2, round(w * s) // 2 * 2), max(2, round(h * s) // 2 * 2)
