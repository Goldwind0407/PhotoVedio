"""同一格 A/B:同一個起點各編碼 1 格(不處理 vs 處理),比較細節量與色彩,並輸出局部放大對比。

用法:python ab_test.py 影片.mp4 秒數 輸出.jpg
"""
import sys

import cv2
import numpy as np

import roundtrip
from algorithms import DetailEnhancer
from quality import detail, frame
from verify import stats

src, t, jpg = sys.argv[1], float(sys.argv[2]), sys.argv[3]
tmp = jpg.rsplit(".", 1)[0]
roundtrip.run(src, tmp + "_a.mp4", lambda f: f, t, 1 / 60)
roundtrip.run(src, tmp + "_b.mp4", DetailEnhancer(), t, 1 / 60)
a, b = frame(tmp + "_a.mp4", 0), frame(tmp + "_b.mp4", 0)
sa = stats(cv2.resize(a, (360, 640), interpolation=cv2.INTER_AREA))
sb = stats(cv2.resize(b, (360, 640), interpolation=cv2.INTER_AREA))
print(f"色彩  亮度 {sa[0]:.1f}→{sb[0]:.1f}  P5/P95 {sa[2]:.0f}/{sa[3]:.0f}→{sb[2]:.0f}/{sb[3]:.0f}  飽和 {sa[4]:.1f}→{sb[4]:.1f}")
print(f"細節  全畫面 {detail(a):.0f}→{detail(b):.0f}")
h, w = a.shape[:2]
# 找細節最多的區塊與最平坦的區塊(檢查雜訊是否被放大)
best, flat = None, None
for y in range(0, h - 400, 200):
    for x in range(0, w - 400, 200):
        d = detail(a[y:y + 400, x:x + 400])
        if best is None or d > best[0]:
            best = (d, y, x)
        if flat is None or d < flat[0]:
            flat = (d, y, x)
tiles = []
for name, (_, y, x) in (("紋理最多", best), ("最平坦", flat)):
    ca, cb = a[y:y + 400, x:x + 400], b[y:y + 400, x:x + 400]
    print(f"  {name}區塊 細節 {detail(ca):.1f}→{detail(cb):.1f}")
    tiles.append(np.hstack([ca, cb]))
cv2.imwrite(jpg, np.vstack(tiles))
