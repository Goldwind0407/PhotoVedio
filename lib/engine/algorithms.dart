/// 影片修復演算法(opencv_dart),與 python_engine/algorithms.py 逐函式對應。
///
/// 所有回傳 [cv.Mat] 的函式,回傳值由呼叫端負責 dispose;
/// 函式內部產生的暫存 Mat 一律由 [_Scope] 釋放,避免逐格處理時記憶體持續增長。
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

/// 收集暫存物件,離開時一次釋放。
class _Scope {
  final _disposers = <void Function()>[];

  cv.Mat keep(cv.Mat m) {
    _disposers.add(m.dispose);
    return m;
  }

  cv.VecMat keepVec(cv.VecMat v) {
    _disposers.add(v.dispose);
    return v;
  }

  /// 把 [m] 從待釋放清單移除並交給呼叫端(用於回傳值)。
  cv.Mat take(cv.Mat m) {
    _disposers.remove(m.dispose);
    return m;
  }

  void dispose() {
    for (final d in _disposers) {
      d();
    }
    _disposers.clear();
  }
}

Float32List _f32(cv.Mat m) {
  final d = m.data;
  return d.buffer.asFloat32List(d.offsetInBytes, d.lengthInBytes ~/ 4);
}

// --------------------------------------------------------------- 除霧

/// 灰階導向濾波(He et al.),把粗糙的透射率貼合影像邊緣。
cv.Mat _guidedFilter(_Scope s, cv.Mat guide, cv.Mat src, int r, double eps) {
  final k = (2 * r + 1, 2 * r + 1);
  cv.Mat box(cv.Mat m) => s.keep(cv.boxFilter(m, -1, k));
  final meanI = box(guide);
  final meanP = box(src);
  final cov = s.keep(cv.subtract(
      box(s.keep(cv.multiply(guide, src))), s.keep(cv.multiply(meanI, meanP))));
  final varI = s.keep(cv.subtract(
      box(s.keep(cv.multiply(guide, guide))), s.keep(cv.multiply(meanI, meanI))));
  final varEps = s.keep(varI.convertTo(cv.MatType.CV_32FC1, beta: eps));
  final a = s.keep(cv.divide(cov, varEps));
  final b = s.keep(cv.subtract(meanP, s.keep(cv.multiply(a, meanI))));
  return s.keep(
      cv.add(s.keep(cv.multiply(box(a), guide)), box(b)));
}

cv.Mat _darkChannel(_Scope s, cv.Mat img, cv.Mat kernel) {
  final ch = s.keepVec(cv.split(img));
  final m = s.keep(cv.min(ch[0], ch[1]));
  final m2 = s.keep(cv.min(m, ch[2]));
  return s.keep(cv.erode(m2, kernel));
}

/// 暗通道先驗去霧(He et al. 2009)。
///
/// 影片要避免閃爍,所以大氣光 A 在時間上做指數平滑;
/// 為了速度,透射率在縮小的影像上估計,再放大回原尺寸。
class Dehazer {
  Dehazer({
    double strength = 0.6,
    this.patch = 15,
    this.t0 = 0.3,
    this.workWidth = 480,
    this.smooth = 0.9,
    this.keepBrightness = true,
    this.chroma = 0.35,
  }) : omega = strength.clamp(0.0, 1.0) * 0.95 {
    _kernel = cv.getStructuringElement(cv.MORPH_RECT, (patch, patch));
  }

  final double omega;
  final int patch;
  final double t0;
  final int workWidth;
  final double smooth;
  final bool keepBrightness;

  /// 色度取自去霧結果的比例;其餘取自原片,避免色彩失真、飽和度暴增。
  final double chroma;
  double _gain = 1.0;
  int _n = 0;
  late final cv.Mat _kernel;
  List<double>? _a; // BGR

  void dispose() => _kernel.dispose();

  List<double> _estimateA(cv.Mat smallF, cv.Mat dark) {
    final d = _f32(dark);
    final c = _f32(smallF);
    final n = math.max(1, (d.length * 0.001).toInt());
    final sorted = Float32List.fromList(d)..sort();
    final thr = sorted[d.length - n];
    // 暗通道最亮 0.1% 的像素中,取亮度總和最高者的顏色。
    var best = -1.0;
    var bgr = const [1.0, 1.0, 1.0];
    for (var i = 0; i < d.length; i++) {
      if (d[i] < thr) continue;
      final sum = c[3 * i] + c[3 * i + 1] + c[3 * i + 2];
      if (sum > best) {
        best = sum;
        bgr = [c[3 * i], c[3 * i + 1], c[3 * i + 2]];
      }
    }
    final prev = _a;
    return _a = prev == null
        ? bgr
        : [for (var i = 0; i < 3; i++) smooth * prev[i] + (1 - smooth) * bgr[i]];
  }

  /// [frame] 為 8UC3 BGR。回傳新的 Mat。
  cv.Mat call(cv.Mat frame) {
    final s = _Scope();
    try {
      final w = frame.cols, h = frame.rows;
      final scale = math.min(1.0, workWidth / w);
      final small = scale < 1
          ? s.keep(cv.resize(frame, ((w * scale).toInt(), (h * scale).toInt()),
              interpolation: cv.INTER_AREA))
          : frame;
      final sf = s.keep(small.convertTo(cv.MatType.CV_32FC3, alpha: 1 / 255));

      final dark = _darkChannel(s, sf, _kernel);
      final a = _estimateA(sf, dark).map((v) => math.max(v, 1e-3)).toList();
      final aSmall = s.keep(cv.Mat.fromScalar(
          sf.rows, sf.cols, cv.MatType.CV_32FC3, cv.Scalar(a[0], a[1], a[2], 0)));
      final normed = s.keep(cv.divide(sf, aSmall));
      final dark2 = _darkChannel(s, normed, _kernel);
      var t = s.keep(dark2.convertTo(cv.MatType.CV_32FC1, alpha: -omega, beta: 1));

      final gray = s.keep(cv.cvtColor(small, cv.COLOR_BGR2GRAY));
      final grayF = s.keep(gray.convertTo(cv.MatType.CV_32FC1, alpha: 1 / 255));
      t = _guidedFilter(s, grayF, t, math.max(8, patch * 2), 1e-3);

      final tLo = s.keep(cv.Mat.fromScalar(
          t.rows, t.cols, cv.MatType.CV_32FC1, cv.Scalar(t0, 0, 0, 0)));
      final hi = s.keep(cv.Mat.fromScalar(
          t.rows, t.cols, cv.MatType.CV_32FC1, cv.Scalar(1, 0, 0, 0)));
      t = s.keep(cv.min(s.keep(cv.max(t, tLo)), hi));

      // 暗通道去霧一定會把畫面壓暗。在縮小圖上量出壓暗幅度,把平均亮度補回原片,
      // 並做時間平滑避免閃爍。乘上同一個增益不會改變色彩比例。
      var gain = 1.0;
      if (keepBrightness) {
        final t3 = s.keep(cv.merge(s.keepVec(cv.VecMat.fromList([t, t, t]))));
        final js = s.keep(cv.add(
            s.keep(cv.divide(s.keep(cv.subtract(sf, aSmall)), t3)), aSmall));
        final js8 = s.keep(js.convertTo(cv.MatType.CV_8UC3, alpha: 255));
        double luma(cv.Scalar m) => 0.114 * m.val1 + 0.587 * m.val2 + 0.299 * m.val3;
        final yIn = luma(cv.mean(sf));
        final yOut = luma(cv.mean(js8)) / 255;
        final g = (yIn / math.max(yOut, 1e-3)).clamp(1.0, 2.5);
        _gain = _n == 0 ? g : smooth * _gain + (1 - smooth) * g;
        _n++;
        gain = _gain;
      }

      if (scale < 1) {
        t = s.keep(cv.resize(t, (w, h), interpolation: cv.INTER_LINEAR));
      }

      final f = s.keep(frame.convertTo(cv.MatType.CV_32FC3, alpha: 1 / 255));
      final aFull = s.keep(cv.Mat.fromScalar(
          h, w, cv.MatType.CV_32FC3, cv.Scalar(a[0], a[1], a[2], 0)));
      final tt = s.keep(cv.merge(s.keepVec(cv.VecMat.fromList([t, t, t]))));
      // J = (I - A) / t + A
      final j = s.keep(cv.add(
          s.keep(cv.divide(s.keep(cv.subtract(f, aFull)), tt)), aFull));
      // convertTo 對 8U 自動飽和,等同 clip(0,255)。
      final out = s.keep(j.convertTo(cv.MatType.CV_8UC3, alpha: 255 * gain));
      if (chroma >= 1.0) return s.take(out);

      // 明度取自去霧結果,色度(a、b)大部分取自原片。
      final lo = s.keep(cv.cvtColor(out, cv.COLOR_BGR2Lab));
      final li = s.keep(cv.cvtColor(frame, cv.COLOR_BGR2Lab));
      final mix = s.keep(cv.addWeighted(li, 1 - chroma, lo, chroma, 0));
      final loCh = s.keepVec(cv.split(lo));
      final mixCh = s.keepVec(cv.split(mix));
      final lab = s.keep(
          cv.merge(s.keepVec(cv.VecMat.fromList([loCh[0], mixCh[1], mixCh[2]]))));
      return cv.cvtColor(lab, cv.COLOR_Lab2BGR);
    } finally {
      s.dispose();
    }
  }
}

// --------------------------------------------------------------- 防手震

class Motion {
  const Motion(this.dx, this.dy, this.da);
  static const zero = Motion(0, 0, 0);
  final double dx, dy, da;
}

/// 兩張相鄰灰階影格間的位移與旋轉角(特徵點光流 + RANSAC 穩健仿射)。
Motion estimateMotion(cv.Mat prevGray, cv.Mat gray) {
  final corners = cv.goodFeaturesToTrack(prevGray, 300, 0.01, 20, blockSize: 3);
  if (corners.length < 6) {
    corners.dispose();
    return Motion.zero;
  }
  final (next, status, err) =
      cv.calcOpticalFlowPyrLK(prevGray, gray, corners, cv.VecPoint2f());
  final from = <cv.Point2f>[], to = <cv.Point2f>[];
  for (var i = 0; i < corners.length; i++) {
    if (status![i] == 1) {
      from.add(corners[i]);
      to.add(next[i]);
    }
  }
  Motion result = Motion.zero;
  if (from.length >= 6) {
    final f = cv.VecPoint2f.fromList(from);
    final t = cv.VecPoint2f.fromList(to);
    final (m, inliers) = cv.estimateAffinePartial2D(f, t,
        method: cv.RANSAC, ransacReprojThreshold: 3.0);
    if (!m.isEmpty) {
      result = Motion(
        m.atF64(0, i1: 2),
        m.atF64(1, i1: 2),
        math.atan2(m.atF64(1, i1: 0), m.atF64(0, i1: 0)),
      );
    }
    m.dispose();
    inliers.dispose();
    f.dispose();
    t.dispose();
  }
  corners.dispose();
  next.dispose();
  status?.dispose();
  err?.dispose();
  return result;
}

/// numpy 'reflect' 邊界(鏡射但不重複邊緣點)。
int _reflect(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
  }
  return i;
}

/// 累積成軌跡後做移動平均,回傳每格要補償的量(平滑軌跡 - 原軌跡)。
List<Motion> smoothTrajectory(List<Motion> motions, int radius) {
  final n = motions.length;
  if (radius <= 0 || n < 3) return List.filled(n, Motion.zero);

  final traj = List.generate(3, (_) => Float64List(n));
  var x = 0.0, y = 0.0, a = 0.0;
  for (var i = 0; i < n; i++) {
    x += motions[i].dx;
    y += motions[i].dy;
    a += motions[i].da;
    traj[0][i] = x;
    traj[1][i] = y;
    traj[2][i] = a;
  }
  double avg(Float64List v, int i) {
    var sum = 0.0;
    for (var k = -radius; k <= radius; k++) {
      sum += v[_reflect(i + k, n)];
    }
    return sum / (2 * radius + 1);
  }

  return [
    for (var i = 0; i < n; i++)
      Motion(avg(traj[0], i) - traj[0][i], avg(traj[1], i) - traj[1][i],
          avg(traj[2], i) - traj[2][i]),
  ];
}

/// 剛好蓋住修正後黑邊所需的放大倍率(取 99 百分位,避免單一極端值放大整支影片)。
/// 固定放大 5% 會無謂地損失解析度;晃動小的影片只需要放大 1~2%。
/// 對應 python_engine/algorithms.py 的 stabilization_zoom。
double stabilizationZoom(List<Motion> c, int w, int h, {double maxZoom = 1.08}) {
  if (c.isEmpty) return 1.0;
  double p99(Iterable<double> v) {
    final s = v.map((e) => e.abs()).toList()..sort();
    final pos = 0.99 * (s.length - 1);
    final lo = pos.floor(), hi = pos.ceil();
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }

  final dx = p99(c.map((m) => m.dx)) / w;
  final dy = p99(c.map((m) => m.dy)) / h;
  final da = p99(c.map((m) => m.da));
  // 平移每邊需要 2×位移比例;旋轉 θ 時角落外露約 θ×(長邊/短邊)。
  final need = 1.0 + 2.0 * math.max(dx, dy) + da * math.max(w, h) / math.min(w, h);
  return need.clamp(1.0, maxZoom);
}

/// 套用補償量並以影格中心放大 [zoom] 倍,裁掉黑邊。回傳新的 Mat。
cv.Mat applyStabilization(cv.Mat frame, Motion c, double zoom) {
  final w = frame.cols, h = frame.rows;
  final cs = math.cos(c.da) * zoom, sn = math.sin(c.da) * zoom;
  final cx = w / 2, cy = h / 2;
  final m = cv.Mat.fromList(2, 3, cv.MatType.CV_64FC1, [
    cs, -sn, cx - cs * cx + sn * cy + c.dx, //
    sn, cs, cy - sn * cx - cs * cy + c.dy,
  ]);
  try {
    // Lanczos 取樣:雙線性在放大時會把細節抹掉約三成,Lanczos 只損失約一成。
    return cv.warpAffine(frame, m, (w, h),
        flags: cv.INTER_LANCZOS4, borderMode: cv.BORDER_REPLICATE);
  } finally {
    m.dispose();
  }
}

// --------------------------------------------------------------- 色調錨定

List<double> _cdf(Uint8List data) {
  final h = List<double>.filled(256, 0);
  for (var i = 0; i < data.length; i++) {
    h[data[i]]++;
  }
  var acc = 0.0;
  for (var i = 0; i < 256; i++) {
    acc += h[i];
    h[i] = acc / data.length;
  }
  return h;
}

/// 整條處理鏈的最後一步:讓輸出的亮度分佈與色彩貼近處理前的影格。
///
/// 除霧、局部對比(CLAHE)、銳化會讓畫面變暗/變亮、拉開對比、色彩變濃。只對齊平均亮度
/// 補不回來(暗部更暗、亮部爆白、平均卻沒變)。這裡對 L 做直方圖匹配:把輸出的亮度分佈
/// 朝原片的分佈拉回 [match] 的比例;映射是單調的,所以局部細節與銳利度不受影響。
/// 色度(a、b)大部分取自原片。LUT 做時間平滑,避免逐格閃爍。
/// 對應 python_engine/algorithms.py 的 ToneAnchor。
class ToneAnchor {
  ToneAnchor({this.match = 1.0, this.chroma = 0.0, this.smooth = 0.9});

  /// 0 = 不處理,1 = 亮度分佈完全等於原片。
  final double match;

  /// 色度取自處理後影格的比例(0 = 完全沿用原片色度)。
  final double chroma;
  final double smooth;
  List<double>? _lut;

  /// [ref] 為處理前的影格,[out] 為處理後的影格,尺寸須相同(8UC3 BGR)。回傳新的 Mat。
  cv.Mat call(cv.Mat ref, cv.Mat out) {
    final s = _Scope();
    try {
      final lr = s.keep(cv.cvtColor(ref, cv.COLOR_BGR2Lab));
      final lo = s.keep(cv.cvtColor(out, cv.COLOR_BGR2Lab));
      final chR = s.keepVec(cv.split(lr));
      final chO = s.keepVec(cv.split(lo));
      final cdfRef = _cdf(chR[0].data);
      final cdfOut = _cdf(chO[0].data);

      // 輸出的每個亮度 → 它在輸出分佈中的分位 → 原片同分位的亮度(線性內插)。
      final lut = List<double>.filled(256, 0);
      for (var i = 0; i < 256; i++) {
        final q = (cdfOut[i] - 0.5 * (cdfOut[i] - (i > 0 ? cdfOut[i - 1] : 0.0)))
            .clamp(0.0, 1.0);
        var target = 255.0;
        if (q <= cdfRef[0]) {
          target = 0;
        } else {
          for (var j = 1; j < 256; j++) {
            if (cdfRef[j] >= q) {
              final span = cdfRef[j] - cdfRef[j - 1];
              target = span <= 0 ? j.toDouble() : (j - 1) + (q - cdfRef[j - 1]) / span;
              break;
            }
          }
        }
        lut[i] = (1 - match) * i + match * target;
      }
      final prev = _lut;
      if (prev != null) {
        for (var i = 0; i < 256; i++) {
          lut[i] = smooth * prev[i] + (1 - smooth) * lut[i];
        }
      }
      _lut = lut;

      var running = 0.0; // 保持單調
      final lutData = List<int>.generate(256, (i) {
        running = math.max(running, lut[i]);
        return running.round().clamp(0, 255);
      });
      final lutMat = s.keep(cv.Mat.fromList(1, 256, cv.MatType.CV_8UC1, lutData));
      final l = s.keep(cv.LUT(chO[0], lutMat));

      final mix = s.keep(cv.addWeighted(lr, 1 - chroma, lo, chroma, 0));
      final chM = s.keepVec(cv.split(mix));
      final lab = s.keep(cv.merge(s.keepVec(cv.VecMat.fromList([l, chM[1], chM[2]]))));
      return cv.cvtColor(lab, cv.COLOR_Lab2BGR);
    } finally {
      s.dispose();
    }
  }
}

// --------------------------------------------------------------- 提高畫質(新版)

/// 估計影像雜訊標準差(Immerkær 1996),只取最平坦的 30% 區域,避免把紋理當成雜訊。
/// [gray] 為 32FC1。對應 python_engine/algorithms.py 的 estimate_noise。
double estimateNoise(cv.Mat gray) {
  final s = _Scope();
  try {
    final k = s.keep(cv.Mat.fromList(3, 3, cv.MatType.CV_32FC1, [1.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 1.0]));
    final resp = s.keep(cv.filter2D(gray, -1, k));
    final zeros = s.keep(cv.Mat.zeros(gray.rows, gray.cols, cv.MatType.CV_32FC1));
    final absResp = s.keep(cv.absDiff(resp, zeros));
    final gx = s.keep(cv.sobel(gray, cv.MatType.CV_32F, 1, 0));
    final gy = s.keep(cv.sobel(gray, cv.MatType.CV_32F, 0, 1));
    final grad = s.keep(cv.magnitude(gx, gy));

    // 梯度第 30 百分位(直方圖近似,256 格)。
    final g = _f32(grad);
    var maxG = 0.0;
    for (final v in g) {
      if (v > maxG) maxG = v;
    }
    if (maxG <= 0) return 0;
    final bins = List<int>.filled(256, 0);
    for (final v in g) {
      bins[(v / maxG * 255).floor().clamp(0, 255)]++;
    }
    var acc = 0, p30 = maxG;
    for (var i = 0; i < 256; i++) {
      acc += bins[i];
      if (acc >= g.length * 0.3) {
        p30 = (i + 1) / 256 * maxG;
        break;
      }
    }
    final (_, flat) = cv.threshold(grad, p30, 255, cv.THRESH_BINARY_INV);
    final mask = s.keep(s.keep(flat).convertTo(cv.MatType.CV_8UC1));
    final meanResp = cv.mean(absResp, mask: mask).val1;
    return math.sqrt(math.pi / 2) / 6 * meanResp;
  } finally {
    s.dispose();
  }
}

/// 軟閾值:振幅小於 t 的部分歸零,其餘往 0 收 t。等同 d - clip(d, -t, t)。
cv.Mat _soft(_Scope s, cv.Mat d, double t) {
  final lo = s.keep(cv.Mat.fromScalar(d.rows, d.cols, cv.MatType.CV_32FC1, cv.Scalar(-t, 0, 0, 0)));
  final hi = s.keep(cv.Mat.fromScalar(d.rows, d.cols, cv.MatType.CV_32FC1, cv.Scalar(t, 0, 0, 0)));
  final clipped = s.keep(cv.min(s.keep(cv.max(d, lo)), hi));
  return s.keep(cv.subtract(d, clipped));
}

cv.Mat _clamp(_Scope s, cv.Mat m, double lo, double hi) {
  final l = s.keep(cv.Mat.fromScalar(m.rows, m.cols, cv.MatType.CV_32FC1, cv.Scalar(lo, 0, 0, 0)));
  final h = s.keep(cv.Mat.fromScalar(m.rows, m.cols, cv.MatType.CV_32FC1, cv.Scalar(hi, 0, 0, 0)));
  return s.keep(cv.min(s.keep(cv.max(m, l)), h));
}

/// 不傷原片的畫質增強:只增強亮度細節,色彩與整體亮度不變。
///
/// 1. 用自導向濾波把亮度 Y 拆成「基底 + 中尺度細節 + 細尺度細節」。導向濾波會把強邊緣
///    留在基底,所以放大細節時不會在邊緣產生光暈(反銳化遮罩會)。
/// 2. 依估計的雜訊強度做軟閾值:低於雜訊門檻的振幅不放大;雜訊明顯時把它扣掉(降噪)。
/// 3. 紋理遮罩:只在局部細節能量明顯高於雜訊的區域增強,皮膚、牆面等平坦區不動。
/// 4. 把亮度增量同時加到 B、G、R,色相與飽和度幾乎不變;細節層平均值約為 0,整體亮度不變。
///
/// 4K 的浮點中間結果很大,所以分條處理:每條上下多取 [overlap] 像素(大於濾波器作用範圍),
/// 只寫回中間部分,結果與整張處理相同,峰值記憶體約為 1/[strips]。
/// 對應 python_engine/algorithms.py 的 DetailEnhancer。
class DetailEnhancer {
  DetailEnhancer({
    this.fine = 0.6,
    this.medium = 0.35,
    this.smooth = 0.9,
    this.maxDelta = 14.0,
    this.strips = 4,
    this.overlap = 48,
    this.noiseEvery = 15,
  });

  final double fine; // 細尺度(紋理、毛髮)增強量
  final double medium; // 中尺度(輪廓、局部立體感)增強量
  final double smooth;
  final double maxDelta; // 單一像素最大亮度變化,避免過度銳化
  final int strips, overlap;
  final int noiseEvery; // 每幾格重新估計一次雜訊(雜訊在短時間內幾乎不變)
  double? _sigma;
  int _frame = 0;

  static const _luma = [0.114, 0.587, 0.299]; // BGR

  cv.Mat _lumaOf(_Scope s, cv.Mat bgr32) {
    final ch = s.keepVec(cv.split(bgr32));
    final b = s.keep(cv.addWeighted(ch[0], _luma[0], ch[1], _luma[1], 0));
    return s.keep(cv.addWeighted(b, 1, ch[2], _luma[2], 0));
  }

  void _updateNoise(cv.Mat frame) {
    final s = _Scope();
    try {
      final half = s.keep(cv.resize(frame, (frame.cols ~/ 2, frame.rows ~/ 2),
          interpolation: cv.INTER_AREA));
      final y = _lumaOf(s, s.keep(half.convertTo(cv.MatType.CV_32FC3)));
      final est = estimateNoise(y) * 1.4; // 縮半會降低雜訊,補回約 √2
      final prev = _sigma;
      _sigma = prev == null ? est : smooth * prev + (1 - smooth) * est;
    } finally {
      s.dispose();
    }
  }

  /// [frame] 為 8UC3 BGR。回傳新的 Mat。
  cv.Mat call(cv.Mat frame) {
    if (_sigma == null || _frame % noiseEvery == 0) _updateNoise(frame);
    _frame++;
    final sigma = _sigma!;
    final out = cv.Mat.zeros(frame.rows, frame.cols, cv.MatType.CV_8UC3);
    final h = frame.rows, w = frame.cols;
    final step = (h / strips).ceil();
    for (var y0 = 0; y0 < h; y0 += step) {
      final y1 = math.min(h, y0 + step);
      final top = math.max(0, y0 - overlap), bottom = math.min(h, y1 + overlap);
      final s = _Scope();
      try {
        final src = s.keep(frame.region(cv.Rect(0, top, w, bottom - top)));
        final res = _enhanceStrip(s, src, sigma);
        final inner = s.keep(res.region(cv.Rect(0, y0 - top, w, y1 - y0)));
        final dst = s.keep(out.region(cv.Rect(0, y0, w, y1 - y0)));
        inner.copyTo(dst);
      } finally {
        s.dispose();
      }
    }
    return out;
  }

  cv.Mat _enhanceStrip(_Scope s, cv.Mat strip, double sigma) {
    final f = s.keep(strip.convertTo(cv.MatType.CV_32FC3));
    final y = _lumaOf(s, f);
    final b1 = _guidedFilter(s, y, y, 2, math.pow(0.04 * 255, 2).toDouble());
    final b2 = _guidedFilter(s, b1, b1, 8, math.pow(0.08 * 255, 2).toDouble());
    final d1 = s.keep(cv.subtract(y, b1));
    final d2 = s.keep(cv.subtract(b1, b2));
    final t = 1.5 * sigma;
    final d1c = _soft(s, d1, t);
    final d2c = _soft(s, d2, 0.5 * t);

    // 紋理遮罩:局部細節能量明顯高於雜訊的地方才增強。
    final energy = s.keep(cv.sqrt(s.keep(cv.boxFilter(s.keep(cv.multiply(d1, d1)), -1, (9, 9)))));
    final ramp = 1.5 * sigma + 1.0;
    final wRaw = s.keep(energy.convertTo(cv.MatType.CV_32FC1,
        alpha: 1 / ramp, beta: -(1.5 * sigma + 0.8) / ramp));
    final wMask = s.keep(cv.gaussianBlur(_clamp(s, wRaw, 0, 1), (0, 0), 3));

    final boost = s.keep(cv.addWeighted(d1c, fine, d2c, medium, 0));
    var delta = s.keep(cv.multiply(wMask, boost));
    // 雜訊明顯(σ > 2)才降噪:扣掉被判定為雜訊的細尺度振幅。
    final dn = ((sigma - 2.0) / 4.0).clamp(0.0, 1.0);
    if (dn > 0) {
      final noise = s.keep(cv.subtract(d1, d1c));
      delta = s.keep(cv.addWeighted(delta, 1, noise, -dn, 0));
    }
    delta = _clamp(s, delta, -maxDelta, maxDelta);

    final delta3 = s.keep(cv.merge(s.keepVec(cv.VecMat.fromList([delta, delta, delta]))));
    final sum = s.keep(cv.add(f, delta3));
    return s.keep(sum.convertTo(cv.MatType.CV_8UC3)); // 8U 自動飽和裁切
  }
}
