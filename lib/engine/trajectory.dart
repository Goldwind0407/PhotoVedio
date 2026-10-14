/// 防手震的鏡頭路徑計算(純 Dart,不依賴 OpenCV),與 python_engine/algorithms.py 逐函式對應。
library;

import 'dart:math' as math;
import 'dart:typed_data';

class Motion {
  const Motion(this.dx, this.dy, this.da);
  static const zero = Motion(0, 0, 0);
  final double dx, dy, da;
}

/// 防手震強度。
enum StabilizeLevel {
  /// 每邊最多修正 3%、旋轉 0.02 rad,約放大 6~10%。
  standard(0.03, 0.02, 1.12),

  /// 每邊最多修正 5%、旋轉 0.03 rad,約放大 10~16%,大幅晃動(走路)時明顯更穩。
  strong(0.05, 0.03, 1.20),

  /// 三腳架鎖定:鏡頭路徑換成完全平滑的曲線,裁切最多;放大超過上限時改用 [strong]。
  lock(0.05, 0.03, 1.25);

  const StabilizeLevel(this.budget, this.rotBudget, this.maxZoom);
  final double budget, rotBudget, maxZoom;
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

List<Float64List> _cumulative(List<Motion> motions) {
  final n = motions.length;
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
  return traj;
}

Float64List _gaussSmooth(Float64List v, double sigma) {
  final n = v.length;
  final r = (3 * sigma).toInt();
  final k = Float64List(2 * r + 1);
  var sum = 0.0;
  for (var i = -r; i <= r; i++) {
    sum += k[i + r] = math.exp(-0.5 * (i / sigma) * (i / sigma));
  }
  final out = Float64List(n);
  for (var i = 0; i < n; i++) {
    var acc = 0.0;
    for (var j = -r; j <= r; j++) {
      acc += k[j + r] * v[_reflect(i + j, n)];
    }
    out[i] = acc / sum;
  }
  return out;
}

/// 平滑時間尺度約 0.27 秒,不隨幀率改變。
double smoothingSigma(double fps) => (8.0 * fps / 30.0).clamp(4.0, 16.0);

/// 在裁切上限內找最平滑的鏡頭路徑,回傳每格補償量(平滑路徑 - 原路徑)。
///
/// 反覆「高斯平滑 → 夾回原路徑 ± 上限」:路徑保持平滑,又不會超出裁切範圍。
/// 舊做法(移動平均後逐格硬截斷)在修正量碰到上限時會讓抖動原封不動漏過去。
/// 對應 python_engine/algorithms.py 的 stabilize_path(附合成測試數據)。
List<Motion> stabilizePath(List<Motion> motions, int w, int h,
    {double budget = 0.03, double rotBudget = 0.02, double sigma = 8.0, int iters = 60}) {
  final n = motions.length;
  if (n < 3) return List.filled(n, Motion.zero);
  final traj = _cumulative(motions);
  final bounds = [budget * w, budget * h, rotBudget];
  final path = <Float64List>[];
  for (var c = 0; c < 3; c++) {
    final t = traj[c], b = bounds[c];
    var p = _gaussSmooth(t, sigma);
    for (var it = 0; it < iters; it++) {
      // 迭代時用 0.8 倍上限,最後一次平滑後才不會超出上限太多
      for (var i = 0; i < n; i++) {
        p[i] = p[i].clamp(t[i] - 0.8 * b, t[i] + 0.8 * b);
      }
      p = _gaussSmooth(p, sigma);
    }
    for (var i = 0; i < n; i++) {
      p[i] = p[i].clamp(t[i] - b, t[i] + b) - t[i];
    }
    path.add(p);
  }
  return [for (var i = 0; i < n; i++) Motion(path[0][i], path[1][i], path[2][i])];
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

/// 最小平方多項式擬合,回傳擬合值。t 先正規化到 [-1, 1],避免高次項數值不穩。
Float64List _polyFit(Float64List v, int deg) {
  final n = v.length, m = deg + 1;
  final ts = Float64List.fromList([for (var i = 0; i < n; i++) n == 1 ? 0.0 : 2.0 * i / (n - 1) - 1]);
  // 正規方程 A^T A c = A^T v
  final a = List.generate(m, (_) => Float64List(m + 1));
  for (var i = 0; i < n; i++) {
    final pw = Float64List(m);
    pw[0] = 1;
    for (var k = 1; k < m; k++) {
      pw[k] = pw[k - 1] * ts[i];
    }
    for (var r = 0; r < m; r++) {
      for (var c = 0; c < m; c++) {
        a[r][c] += pw[r] * pw[c];
      }
      a[r][m] += pw[r] * v[i];
    }
  }
  // 高斯消去(部分選主元)
  for (var col = 0; col < m; col++) {
    var piv = col;
    for (var r = col + 1; r < m; r++) {
      if (a[r][col].abs() > a[piv][col].abs()) piv = r;
    }
    final tmp = a[col];
    a[col] = a[piv];
    a[piv] = tmp;
    for (var r = 0; r < m; r++) {
      if (r == col || a[col][col] == 0) continue;
      final f = a[r][col] / a[col][col];
      for (var c = col; c <= m; c++) {
        a[r][c] -= f * a[col][c];
      }
    }
  }
  final coef = [for (var r = 0; r < m; r++) a[r][r] == 0 ? 0.0 : a[r][m] / a[r][r]];
  return Float64List.fromList([
    for (var i = 0; i < n; i++)
      () {
        var y = 0.0, p = 1.0;
        for (var k = 0; k < m; k++) {
          y += coef[k] * p;
          p *= ts[i];
        }
        return y;
      }(),
  ]);
}

/// 三腳架鎖定:把鏡頭路徑換成一條完全平滑的多項式(0 次 = 固定不動、1 次 = 等速運鏡…),
/// 抵銷所有晃動與不規則漂移。在 0~3 次中選需要放大最少的;超過 [maxZoom] 就回傳 null。
/// 回傳(每格補償量, 放大倍率, 多項式次數)。對應 python_engine/algorithms.py 的 lock_trajectory。
(List<Motion>, double, int)? lockTrajectory(List<Motion> motions, int w, int h,
    {double maxZoom = 1.25}) {
  final n = motions.length;
  final traj = _cumulative(motions);
  (List<Motion>, double, int)? best;
  for (var deg = 0; deg < 4; deg++) {
    if (n <= deg + 1) break;
    final fit = [for (var c = 0; c < 3; c++) _polyFit(traj[c], deg)];
    final corr = [
      for (var i = 0; i < n; i++)
        Motion(fit[0][i] - traj[0][i], fit[1][i] - traj[1][i], fit[2][i] - traj[2][i]),
    ];
    final zoom = stabilizationZoom(corr, w, h, maxZoom: 4.0);
    if (best == null || zoom < best.$2 - 0.005) best = (corr, zoom, deg); // 次數越低越穩,除非能明顯少裁
  }
  if (best == null || best.$2 > maxZoom) return null;
  return best;
}
