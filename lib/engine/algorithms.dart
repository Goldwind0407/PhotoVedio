/// 影片修復演算法(opencv_dart),與 python_engine/algorithms.py 逐函式對應。
///
/// 所有回傳 [cv.Mat] 的函式,回傳值由呼叫端負責 dispose;
/// 函式內部產生的暫存 Mat 一律由 [_Scope] 釋放,避免逐格處理時記憶體持續增長。
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'trajectory.dart';

export 'trajectory.dart';

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

  /// 不再需要的大型暫存立刻釋放,不等到離開範圍(4K 時每張浮點圖就有數十 MB)。
  void drop(cv.Mat m) => take(m).dispose();

  /// 在內層範圍執行 [body]:它的暫存一結束就釋放,只有回傳值留在本範圍。
  cv.Mat inner(cv.Mat Function(_Scope s) body) {
    final s = _Scope();
    try {
      return keep(s.take(body(s)));
    } finally {
      s.dispose();
    }
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
/// 暫存用完立刻釋放,同時存在的浮點圖最多約 5 張(4K 分條處理時峰值記憶體的主要來源)。
cv.Mat _guidedFilter(_Scope s, cv.Mat guide, cv.Mat src, int r, double eps) {
  final k = (2 * r + 1, 2 * r + 1);
  cv.Mat box(_Scope t, cv.Mat m) => t.keep(cv.boxFilter(m, -1, k));
  final meanI = box(s, guide);
  final meanP = box(s, src);
  // cov = mean(I·p) - meanI·meanP
  final cov = s.inner((t) => t.keep(cv.subtract(
      box(t, t.keep(cv.multiply(guide, src))), t.keep(cv.multiply(meanI, meanP)))));
  // a = cov / (var(I) + eps)
  final a = s.inner((t) {
    final varI = t.keep(cv.subtract(
        box(t, t.keep(cv.multiply(guide, guide))), t.keep(cv.multiply(meanI, meanI))));
    final varEps = t.keep(varI.convertTo(cv.MatType.CV_32FC1, beta: eps));
    return t.keep(cv.divide(cov, varEps));
  });
  s.drop(cov);
  final b = s.inner((t) => t.keep(cv.subtract(meanP, t.keep(cv.multiply(a, meanI)))));
  s
    ..drop(meanP)
    ..drop(meanI);
  final meanA = box(s, a);
  s.drop(a);
  final meanB = box(s, b);
  s.drop(b);
  final q = s.inner((t) => t.keep(cv.add(t.keep(cv.multiply(meanA, guide)), meanB)));
  s
    ..drop(meanA)
    ..drop(meanB);
  return q;
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

      // 原尺寸的合成全是逐像素運算,分條處理結果與整張相同;
      // 12MP 照片整張做浮點運算要好幾百 MB,分條後峰值只剩一小部分。
      final out = cv.Mat.zeros(h, w, cv.MatType.CV_8UC3);
      final step = math.max(1, _stripPixels ~/ w);
      for (var y0 = 0; y0 < h; y0 += step) {
        final rect = cv.Rect(0, y0, w, math.min(step, h - y0));
        final ss = _Scope();
        try {
          final res = _compose(ss, ss.keep(frame.region(rect)), ss.keep(t.region(rect)), a, gain);
          res.copyTo(ss.keep(out.region(rect)));
        } finally {
          ss.dispose();
        }
      }
      return out;
    } finally {
      s.dispose();
    }
  }

  static const _stripPixels = 1 << 20;

  /// J = (I - A) / t + A,再把色度大部分換回原片。回傳的 Mat 由 [s] 管理。
  cv.Mat _compose(_Scope s, cv.Mat frame, cv.Mat t, List<double> a, double gain) {
    final h = frame.rows, w = frame.cols;
    final f = s.keep(frame.convertTo(cv.MatType.CV_32FC3, alpha: 1 / 255));
    final aFull = s.keep(cv.Mat.fromScalar(
        h, w, cv.MatType.CV_32FC3, cv.Scalar(a[0], a[1], a[2], 0)));
    final tt = s.keep(cv.merge(s.keepVec(cv.VecMat.fromList([t, t, t]))));
    final j = s.keep(cv.add(
        s.keep(cv.divide(s.keep(cv.subtract(f, aFull)), tt)), aFull));
    // convertTo 對 8U 自動飽和,等同 clip(0,255)。
    final out = s.keep(j.convertTo(cv.MatType.CV_8UC3, alpha: 255 * gain));
    if (chroma >= 1.0) return out;

    // 明度取自去霧結果,色度(a、b)大部分取自原片。
    final lo = s.keep(cv.cvtColor(out, cv.COLOR_BGR2Lab));
    final li = s.keep(cv.cvtColor(frame, cv.COLOR_BGR2Lab));
    final mix = s.keep(cv.addWeighted(li, 1 - chroma, lo, chroma, 0));
    final loCh = s.keepVec(cv.split(lo));
    final mixCh = s.keepVec(cv.split(mix));
    final lab = s.keep(
        cv.merge(s.keepVec(cv.VecMat.fromList([loCh[0], mixCh[1], mixCh[2]]))));
    return s.keep(cv.cvtColor(lab, cv.COLOR_Lab2BGR));
  }
}

// --------------------------------------------------------------- 防手震

/// 兩張相鄰灰階影格間的運動(繞畫面中心)與各條帶的果凍殘差(見 [fitBands])。
/// 對應 python_engine/algorithms.py 的 estimate_motion_bands。
(Motion, Float64List) estimateMotionBands(cv.Mat prevGray, cv.Mat gray) {
  final zero = (Motion.zero, Float64List(2 * rsBands));
  final corners = cv.goodFeaturesToTrack(prevGray, 400, 0.01, 16, blockSize: 3);
  if (corners.length < 6) {
    corners.dispose();
    return zero;
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
  var result = zero;
  if (from.length >= 6) {
    final f = cv.VecPoint2f.fromList(from);
    final t = cv.VecPoint2f.fromList(to);
    // 部分相似變換 + RANSAC,只用來排除移動物體;真正的運動用 fitBands 的聯合模型解。
    final (m, inliers) = cv.estimateAffinePartial2D(f, t,
        method: cv.RANSAC, ransacReprojThreshold: 3.0);
    if (!m.isEmpty) {
      final a00 = m.atF64(0, i1: 0), a01 = m.atF64(0, i1: 1), a02 = m.atF64(0, i1: 2);
      final a10 = m.atF64(1, i1: 0), a11 = m.atF64(1, i1: 1), a12 = m.atF64(1, i1: 2);
      final inl = inliers.isEmpty ? null : inliers.data;
      final p0 = <double>[], p1 = <double>[];
      for (var i = 0; i < from.length; i++) {
        final x = from[i].x, y = from[i].y;
        // 果凍位移可能超過 RANSAC 的 3px 門檻,殘差在 12px 內的點也加回來。
        final ex = to[i].x - (a00 * x + a01 * y + a02);
        final ey = to[i].y - (a10 * x + a11 * y + a12);
        if ((inl != null && inl[i] == 1) || ex * ex + ey * ey < 144) {
          p0..add(x)..add(y);
          p1..add(to[i].x)..add(to[i].y);
        }
      }
      if (p0.length >= 12) {
        result = fitBands(Float64List.fromList(p0), Float64List.fromList(p1),
            prevGray.cols, prevGray.rows);
      }
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

/// 套用補償量並以影格中心放大 [zoom] 倍,裁掉黑邊。回傳新的 Mat。
///
/// [rs] 為各條帶的果凍偏移(見 [rollingShutterOffsets]):每 [strip] 列各自多平移 -偏移,
/// 偏移在條帶中心之間線性內插,條帶交界不會有斷層。
/// 對應 python_engine/algorithms.py 的 apply_stabilization。
cv.Mat applyStabilization(cv.Mat frame, Motion c, double zoom, {Float64List? rs, int strip = 16}) {
  final w = frame.cols, h = frame.rows;
  final cs = math.cos(c.da) * zoom, sn = math.sin(c.da) * zoom;
  final cx = w / 2, cy = h / 2;
  final tx = cx - cs * cx + sn * cy + c.dx, ty = cy - sn * cx - cs * cy + c.dy;
  cv.Mat warp(double ox, double oy, int y0, int rows) {
    final m = cv.Mat.fromList(2, 3, cv.MatType.CV_64FC1, [
      cs, -sn, tx - ox, //
      sn, cs, ty - oy - y0,
    ]);
    try {
      // Lanczos 取樣:雙線性在放大時會把細節抹掉約三成,Lanczos 只損失約一成。
      return cv.warpAffine(frame, m, (w, rows),
          flags: cv.INTER_LANCZOS4, borderMode: cv.BORDER_REPLICATE);
    } finally {
      m.dispose();
    }
  }

  if (rs == null || rs.every((v) => v == 0)) return warp(0, 0, 0, h);
  final bands = rs.length ~/ 2;
  double interp(double y, int axis) {
    // 條帶中心 (i + 0.5) * h / bands 之間線性內插,兩端外持平(同 np.interp)。
    final pos = y / h * bands - 0.5;
    if (pos <= 0) return rs[axis];
    if (pos >= bands - 1) return rs[2 * (bands - 1) + axis];
    final i = pos.floor(), f = pos - i;
    return rs[2 * i + axis] * (1 - f) + rs[2 * (i + 1) + axis] * f;
  }

  final out = cv.Mat.zeros(h, w, frame.type);
  for (var y0 = 0; y0 < h; y0 += strip) {
    final rows = math.min(strip, h - y0);
    // 輸出列對應的原影格列(忽略微小旋轉):果凍偏移是按原影格的列量測的。
    final yc = cy + (y0 + rows / 2 - cy - c.dy) / zoom;
    final part = warp(interp(yc, 0), interp(yc, 1), y0, rows);
    final dst = out.region(cv.Rect(0, y0, w, rows));
    part.copyTo(dst);
    dst.dispose();
    part.dispose();
  }
  return out;
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

// --------------------------------------------------------------- 提高畫質

/// 由位元率估計壓縮程度:0 = 畫質好(原檔),1 = 重度壓縮(例如通訊軟體傳過的影片)。
///
/// 每像素分到的位元數(HEVC 效率約為 H.264 的兩倍,換算成 H.264 等級)。
/// 經 LINE 等軟體轉傳的 H.264 約 0.1 bpp;iPhone 4K HEVC 原檔約 0.4。
/// 方塊度、雜訊估計都分不出壓縮程度(H.264 內建的去方塊濾波會把方塊邊界抹平),位元率最可靠。
/// 位元率未知時回傳 0.5。對應 python_engine/algorithms.py 的 compression_level。
double compressionLevel(double bitrate, int w, int h, double fps, {required bool hevc}) {
  if (bitrate <= 0 || w <= 0 || h <= 0 || fps <= 0) return 0.5;
  final bpp = bitrate / (w * h * fps) * (hevc ? 2.0 : 1.0);
  return ((0.35 - bpp) / (0.35 - 0.12)).clamp(0.0, 1.0);
}

/// 銳化量。
enum EnhanceLevel {
  low(0.3),
  medium(0.5),
  high(0.8);

  const EnhanceLevel(this.amount);
  final double amount;
}

/// 提高畫質:先去掉壓縮雜訊,再輕度銳化輪廓。
///
/// 手機影片(尤其是轉傳過的)最大的問題是壓縮造成的方塊、蚊狀雜訊與色塊;直接加強細節會連雜訊一起放大。
/// 這裡先用雙邊濾波(保留邊緣的平滑)去掉小幅度的壓縮雜訊,再用反銳化遮罩讓輪廓清楚。
/// 降噪強度依壓縮程度調整([compressionLevel]):原檔只輕微處理,不會抹掉真正的細紋。
///
/// 以「乾淨畫面 → H.264 重度壓縮」實測(SSIM 越高越接近壓縮前):
/// 不處理 0.9557;舊版細節增強 0.9443(把雜訊也放大);本方法(中)0.9572、(弱)0.9588。
/// 全部在 8 位元影像上運算,沒有浮點中間圖,記憶體與速度都比舊版好得多。
/// 雙邊濾波範圍固定 5 像素:壓縮雜訊以像素為單位(8×8 區塊),與解析度無關;
/// 銳化半徑則依解析度放大,4K 在手機上縮小觀看時才看得出來。
/// 對應 python_engine/algorithms.py 的 DetailEnhancer。
class DetailEnhancer {
  DetailEnhancer({this.amount = 0.5, double compression = 0.5})
      : sigmaColor = 10.0 + 20.0 * compression.clamp(0.0, 1.0);

  DetailEnhancer.level(EnhanceLevel l, {double compression = 0.5})
      : this(amount: l.amount, compression: compression);

  final double amount; // 銳化量
  final double sigmaColor; // 雙邊濾波的色彩容差(越大降噪越強)

  /// [frame] 為 8UC3 BGR。回傳新的 Mat。
  cv.Mat call(cv.Mat frame) {
    final k = math.max(1.0, math.min(frame.cols, frame.rows) / 1080);
    final d = cv.bilateralFilter(frame, 5, sigmaColor, 3);
    try {
      final blur = cv.gaussianBlur(d, (0, 0), 0.8 * k);
      try {
        return cv.addWeighted(d, 1 + amount, blur, -amount, 0);
      } finally {
        blur.dispose();
      }
    } finally {
      d.dispose();
    }
  }
}
