"""影片修復演算法(OpenCV + NumPy):除霧、提高畫質、防手震。

每個函式都只處理單張影格(或軌跡),影片串流由 pipeline.py 負責。
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------- 除霧
def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """灰階導向濾波(He et al.),用來把粗糙的透射率貼合影像邊緣。"""
    k = (2 * r + 1, 2 * r + 1)
    mean_i = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    cov_ip = cv2.boxFilter(guide * src, -1, k) - mean_i * mean_p
    var_i = cv2.boxFilter(guide * guide, -1, k) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def _dark_channel(img: np.ndarray, patch: int) -> np.ndarray:
    return cv2.erode(img.min(axis=2), np.ones((patch, patch), np.uint8))


LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR 權重


class Dehazer:
    """暗通道先驗去霧(He et al. 2009)。

    影片要避免閃爍,所以大氣光 A 在時間上做指數平滑;
    為了速度,透射率在縮小的影像上估計,再放大回原尺寸。
    """

    def __init__(self, strength: float = 0.6, patch: int = 15, t0: float = 0.3,
                 work_width: int = 480, smooth: float = 0.9,
                 keep_brightness: bool = True, chroma: float = 0.35):
        self.omega = float(np.clip(strength, 0.0, 1.0)) * 0.95
        self.patch = patch
        self.t0 = t0
        self.work_width = work_width
        self.smooth = smooth
        self.keep_brightness = keep_brightness
        self.chroma = chroma  # 色度取自去霧結果的比例;其餘取自原片,避免色彩失真
        self._A: np.ndarray | None = None
        self._gain = 1.0
        self._n = 0

    def _estimate_A(self, small: np.ndarray, dark: np.ndarray) -> np.ndarray:
        n = max(1, int(dark.size * 0.001))
        idx = np.argpartition(dark.ravel(), -n)[-n:]
        # 取暗通道最亮 0.1% 像素中,亮度最高者的顏色(避免被白色物體誤導用平均)。
        cand = small.reshape(-1, 3)[idx]
        A = cand[np.argmax(cand.sum(axis=1))]
        if self._A is None:
            self._A = A
        else:
            self._A = self.smooth * self._A + (1 - self.smooth) * A
        return self._A

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame
        s = small.astype(np.float32) / 255.0

        dark = _dark_channel(s, self.patch)
        A = np.maximum(self._estimate_A(s, dark), 1e-3)

        t = 1.0 - self.omega * _dark_channel(s / A, self.patch)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t = _guided_filter(gray, t.astype(np.float32), r=max(8, self.patch * 2), eps=1e-3)
        t = np.clip(t, self.t0, 1.0)
        if scale < 1:
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)

        gain = 1.0
        if self.keep_brightness:
            # 暗通道去霧一定會把畫面壓暗。在縮小圖上量出壓暗幅度,把平均亮度補回原片,
            # 並做時間平滑避免閃爍。乘上同一個增益不會改變色彩比例。
            ts = cv2.resize(t, (s.shape[1], s.shape[0])) if scale < 1 else t
            js = (s - A) / ts[..., None] + A
            y_in = float((s * LUMA).sum(axis=2).mean())
            y_out = float((np.clip(js, 0, 1) * LUMA).sum(axis=2).mean())
            g = float(np.clip(y_in / max(y_out, 1e-3), 1.0, 2.5))
            self._gain = self.smooth * self._gain + (1 - self.smooth) * g if self._n else g
            self._n += 1
            gain = self._gain

        f = frame.astype(np.float32) / 255.0
        J = (f - A) / t[..., None] + A
        out = np.clip(J * 255.0 * gain, 0, 255).astype(np.uint8)
        if self.chroma < 1.0:
            lo = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
            li = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            lo[..., 1:] = li[..., 1:] * (1 - self.chroma) + lo[..., 1:] * self.chroma
            out = cv2.cvtColor(np.clip(lo, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out


# --------------------------------------------------------------- 防手震
def estimate_motion(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float, float]:
    """兩張相鄰影格間的 (dx, dy, 旋轉角),用特徵點光流 + 穩健仿射估計。"""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01,
                                  minDistance=20, blockSize=3)
    if pts is None:
        return 0.0, 0.0, 0.0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 6:
        return 0.0, 0.0, 0.0
    # 部分相似變換(平移+旋轉+等比縮放),RANSAC 排除移動物體上的點。
    m, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                       ransacReprojThreshold=3.0)
    if m is None:
        return 0.0, 0.0, 0.0
    return float(m[0, 2]), float(m[1, 2]), float(np.arctan2(m[1, 0], m[0, 0]))


def smooth_trajectory(motions: np.ndarray, radius: int) -> np.ndarray:
    """累積成軌跡後做移動平均,回傳每格需要補償的 (dx, dy, da)。

    用 reflect 邊界避免開頭結尾被拉向 0;radius 越大越穩,但裁切越多。
    """
    traj = np.cumsum(motions, axis=0)
    if radius <= 0 or len(traj) < 3:
        return np.zeros_like(motions)
    k = np.ones(2 * radius + 1) / (2 * radius + 1)
    pad = np.pad(traj, ((radius, radius), (0, 0)), mode="reflect")
    smooth = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(3)], axis=1)
    return smooth - traj


def apply_stabilization(frame: np.ndarray, correction: np.ndarray, zoom: float) -> np.ndarray:
    """套用補償量並放大 zoom 倍以裁掉黑邊(以影格中心為軸)。"""
    h, w = frame.shape[:2]
    dx, dy, da = correction
    c, s = np.cos(da) * zoom, np.sin(da) * zoom
    cx, cy = w / 2, h / 2
    m = np.array([[c, -s, cx - c * cx + s * cy + dx],
                  [s, c, cy - s * cx - c * cy + dy]], dtype=np.float32)
    # Lanczos 取樣:雙線性在放大時會把細節抹掉約三成,Lanczos 只損失約一成。
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------- 色調錨定
class ToneAnchor:
    """整條處理鏈的最後一步:讓輸出的亮度分佈與色彩貼近處理前的影格。

    除霧、局部對比(CLAHE)、銳化會讓畫面變暗/變亮、拉開對比、色彩變濃。只對齊平均亮度
    補不回來(暗部更暗、亮部爆白、平均卻沒變)。這裡對 L 做直方圖匹配:把輸出的亮度分佈
    朝原片的分佈拉回 ``match`` 的比例;映射是單調的,所以局部細節與銳利度不受影響。
    色度(a、b)大部分取自原片。LUT 做時間平滑,避免逐格閃爍。
    """

    def __init__(self, match: float = 1.0, chroma: float = 0.0, smooth: float = 0.9):
        self.match = match    # 0 = 不處理,1 = 亮度分佈完全等於原片
        self.chroma = chroma  # 色度取自處理後影格的比例(0 = 完全沿用原片色度)
        self.smooth = smooth
        self._lut: np.ndarray | None = None

    def __call__(self, ref: np.ndarray, out: np.ndarray) -> np.ndarray:
        lr = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
        lo = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        cdf_ref = np.cumsum(np.bincount(lr[..., 0].ravel(), minlength=256)).astype(np.float64)
        cdf_out = np.cumsum(np.bincount(lo[..., 0].ravel(), minlength=256)).astype(np.float64)
        cdf_ref /= cdf_ref[-1]
        cdf_out /= cdf_out[-1]
        levels = np.arange(256, dtype=np.float64)
        # 輸出的每個亮度 → 它在輸出分佈中的分位 → 原片同分位的亮度。
        q = np.clip(cdf_out - 0.5 * np.diff(cdf_out, prepend=0.0), 0.0, 1.0)
        target = np.interp(q, cdf_ref, levels)
        lut = (1 - self.match) * levels + self.match * target
        self._lut = lut if self._lut is None else self.smooth * self._lut + (1 - self.smooth) * lut
        lut8 = np.clip(np.maximum.accumulate(self._lut), 0, 255).round().astype(np.uint8)  # 保持單調
        res = np.empty_like(lo)
        res[..., 0] = cv2.LUT(lo[..., 0], lut8)
        res[..., 1:] = (lr[..., 1:].astype(np.float32) * (1 - self.chroma)
                        + lo[..., 1:].astype(np.float32) * self.chroma).round().astype(np.uint8)
        return cv2.cvtColor(res, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------- 提高畫質(新版)
def estimate_noise(gray: np.ndarray) -> float:
    """估計影像雜訊標準差(Immerkær 1996),只取最平坦的 30% 區域,避免把紋理當成雜訊。"""
    g = gray.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(cv2.filter2D(g, -1, k))
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = grad <= np.percentile(grad, 30)
    return float(np.sqrt(np.pi / 2) / 6 * resp[flat].mean())


def _soft(d: np.ndarray, t: float) -> np.ndarray:
    """軟閾值:振幅小於 t 的部分視為雜訊歸零,其餘往 0 收 t。"""
    return np.sign(d) * np.maximum(np.abs(d) - t, 0)


class DetailEnhancer:
    """不傷原片的畫質增強:只增強亮度細節,色彩與整體亮度不變。

    1. 用自導向濾波把亮度 Y 拆成「基底 + 中尺度細節 + 細尺度細節」。導向濾波會把強邊緣
       留在基底,所以放大細節時不會在邊緣產生光暈(反銳化遮罩會)。
    2. 依估計的雜訊強度做軟閾值:低於雜訊門檻的振幅不放大;雜訊明顯時把它扣掉(降噪)。
    3. 紋理遮罩:只在局部細節能量明顯高於雜訊的區域增強,皮膚、牆面等平坦區不動。
    4. 把亮度增量同時加到 B、G、R,色相與飽和度幾乎不變;細節層平均值約為 0,整體亮度不變。
    """

    def __init__(self, fine: float = 0.6, medium: float = 0.35, smooth: float = 0.9,
                 max_delta: float = 14.0, noise_every: int = 15):
        self.fine = fine          # 細尺度(紋理、毛髮)增強量
        self.medium = medium      # 中尺度(輪廓、局部立體感)增強量
        self.smooth = smooth
        self.max_delta = max_delta  # 單一像素最大亮度變化,避免過度銳化
        self.noise_every = noise_every  # 每幾格重新估計一次雜訊(與 App 端相同)
        self._sigma: float | None = None
        self._frame = 0

    @staticmethod
    def _base(y: np.ndarray, r: int, eps: float) -> np.ndarray:
        return _guided_filter(y, y, r, eps)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        y = f @ np.array([0.114, 0.587, 0.299], np.float32)
        if self._sigma is None or self._frame % self.noise_every == 0:
            small = cv2.resize(y, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            s = estimate_noise(small) * 1.4  # 縮半會降低雜訊,補回約 √2
            self._sigma = s if self._sigma is None else self.smooth * self._sigma + (1 - self.smooth) * s
        self._frame += 1
        sigma = self._sigma

        b1 = self._base(y, 2, (0.04 * 255) ** 2)
        b2 = self._base(b1, 8, (0.08 * 255) ** 2)
        d1, d2 = y - b1, b1 - b2
        t = 1.5 * sigma
        d1c, d2c = _soft(d1, t), _soft(d2, 0.5 * t)
        # 雜訊明顯(σ > 2)才降噪:扣掉被判定為雜訊的細尺度振幅。
        dn = float(np.clip((sigma - 2.0) / 4.0, 0.0, 1.0))
        # 紋理遮罩:局部細節能量明顯高於雜訊的地方(毛髮、網子、草地)才增強,
        # 平坦區(皮膚、牆面、天空)權重趨近 0,避免把顆粒放大成斑駁感。
        energy = np.sqrt(cv2.boxFilter(d1 * d1, -1, (9, 9)))
        w = np.clip((energy - 1.5 * sigma - 0.8) / (1.5 * sigma + 1.0), 0.0, 1.0)
        w = cv2.GaussianBlur(w, (0, 0), 3)
        delta = w * (self.fine * d1c + self.medium * d2c) - dn * (d1 - d1c)
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        return np.clip(f + delta[..., None], 0, 255).astype(np.uint8)


def stabilization_zoom(corrections: np.ndarray, w: int, h: int, max_zoom: float = 1.08) -> float:
    """剛好蓋住修正後黑邊所需的放大倍率(取 99 百分位,避免單一極端值放大整支影片)。

    固定放大 5% 會無謂地損失解析度;晃動小的影片只需要放大 1~2%。
    """
    if len(corrections) == 0:
        return 1.0
    dx = np.percentile(np.abs(corrections[:, 0]), 99) / w
    dy = np.percentile(np.abs(corrections[:, 1]), 99) / h
    da = np.percentile(np.abs(corrections[:, 2]), 99)
    # 平移每邊需要 2×位移比例;旋轉 θ 時角落外露約 θ×(長邊/短邊)。
    need = 1.0 + 2.0 * max(dx, dy) + da * max(w, h) / min(w, h)
    return float(np.clip(need, 1.0, max_zoom))


def limit_corrections(corr: np.ndarray, w: int, h: int, budget: float = 0.015) -> np.ndarray:
    """把每格修正量限制在畫面每邊 budget 比例以內,控制防手震需要的裁切放大。

    在 test5 實測:視窗 15 格、上限 1.5% 時只需放大 3.5%,仍保有約 60% 的抖動降低;
    不設上限則要放大近 9%,細節損失會抵銷畫質增強。超出上限的多半是快速運鏡,本來就不該修。
    """
    c = corr.copy()
    c[:, 0] = np.clip(c[:, 0], -budget * w, budget * w)
    c[:, 1] = np.clip(c[:, 1], -budget * h, budget * h)
    c[:, 2] = np.clip(c[:, 2], -budget, budget)
    return c


def lock_trajectory(motions: np.ndarray, w: int, h: int, max_zoom: float = 1.25
                    ) -> tuple[np.ndarray, float, int] | None:
    """三腳架鎖定:把鏡頭路徑換成一條完全平滑的多項式(0 次 = 固定不動、1 次 = 等速運鏡…),
    抵銷所有晃動與不規則漂移。在 0~3 次中選需要放大最少的;超過 max_zoom 就回傳 None。

    回傳 (每格補償量, 放大倍率, 多項式次數)。
    """
    traj = np.cumsum(motions, axis=0)
    t = np.arange(len(traj))
    best = None
    for deg in range(4):
        if len(traj) <= deg + 1:
            break
        fit = np.stack([np.polyval(np.polyfit(t, traj[:, i], deg), t) for i in range(3)], axis=1)
        corr = fit - traj
        zoom = stabilization_zoom(corr, w, h, max_zoom=4.0)
        if best is None or zoom < best[1] - 0.005:  # 次數越低越穩,除非能明顯少裁
            best = (corr, zoom, deg)
    if best is None or best[1] > max_zoom:
        return None
    return best
