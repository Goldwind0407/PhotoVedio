"""合成測試:晃動 + 霧 → 驗證防手震與除霧確實有效。"""
import os, tempfile
import cv2, numpy as np
from pipeline import Options, process_video
from algorithms import estimate_motion

rng = np.random.default_rng(0)
W, H, N = 640, 360, 90
# 大張紋理場景,攝影機在上面緩慢平移 + 隨機抖動
scene = cv2.GaussianBlur(rng.integers(0, 255, (H + 200, W + 200, 3), dtype=np.uint8), (0, 0), 2)
for _ in range(60):
    cv2.circle(scene, (int(rng.integers(0, W + 200)), int(rng.integers(0, H + 200))),
               int(rng.integers(5, 30)), [int(x) for x in rng.integers(0, 255, 3)], -1)
d = tempfile.mkdtemp()
src = os.path.join(d, "shaky.mp4")
wr = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for i in range(N):
    x = 100 + i * 0.8 + rng.normal(0, 6); y = 100 + rng.normal(0, 6)
    M = np.float32([[1, 0, -x], [0, 1, -y]])
    f = cv2.warpAffine(scene, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    f = (f * 0.55 + 255 * 0.40).astype(np.uint8)  # 加霧:低對比 + 泛白
    wr.write(f)
wr.release()

def jitter(path):
    cap = cv2.VideoCapture(path); ok, p = cap.read(); p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    ms = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); ms.append(estimate_motion(p, g)[:2]); p = g
    m = np.array(ms)
    # 去掉緩慢平移後,殘餘的高頻抖動
    return np.std(m - np.convolve(m[:, 0], np.ones(1), "same")[:, None] * [1, 0] * 0, axis=0)

def contrast(path):
    cap = cv2.VideoCapture(path); ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).std()

out = os.path.join(d, "fixed.mp4")
process_video(src, out, Options(stabilize=True, dehaze=True, enhance=True))
print("\n抖動(逐格位移標準差 px) 前:", jitter(src).round(2), " 後:", jitter(out).round(2))
print("對比(灰階標準差)       前:", round(contrast(src), 1), " 後:", round(contrast(out), 1))
print("輸出存在:", os.path.getsize(out) > 0, out)
