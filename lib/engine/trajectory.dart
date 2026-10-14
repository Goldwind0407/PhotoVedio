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

/// 剛好讓每一格都蓋滿畫面的放大倍率(精確計算,所有影格取最大值)。
///
/// 輸出像素 X 取樣自原影格 c + R(-θ)(X - c - d ± o) / zoom(o 為果凍修正的條帶偏移),
/// 四個角落都要落在原影格內。舊做法取 99 百分位、且沒算果凍偏移,
/// 約有 1~3% 的影格邊緣會露出被拉長的像素。
/// [rs] 為各格的果凍偏移(見 [rollingShutterOffsets])。
/// 對應 python_engine/algorithms.py 的 stabilization_zoom。
double stabilizationZoom(List<Motion> c, int w, int h,
    {double maxZoom = 1.08, List<Float64List>? rs}) {
  final cx = w / 2, cy = h / 2;
  var need = 1.0;
  for (var i = 0; i < c.length; i++) {
    final m = c[i];
    var ox = 0.0, oy = 0.0;
    if (rs != null && i < rs.length) {
      for (var b = 0; b + 1 < rs[i].length; b += 2) {
        ox = math.max(ox, rs[i][b].abs());
        oy = math.max(oy, rs[i][b + 1].abs());
      }
    }
    final cs = math.cos(m.da), sn = math.sin(m.da);
    for (final vx in [-cx - m.dx - ox, -cx - m.dx + ox, cx - m.dx - ox, cx - m.dx + ox]) {
      for (final vy in [-cy - m.dy - oy, -cy - m.dy + oy, cy - m.dy - oy, cy - m.dy + oy]) {
        final ux = cs * vx + sn * vy, uy = -sn * vx + cs * vy; // R(-θ) v
        need = math.max(need, math.max(ux.abs() / cx, uy.abs() / cy));
      }
    }
  }
  return math.min(need, maxZoom);
}

/// 解線性方程組 a x = b(高斯消去、部分選主元)。a、b 會被改寫。
Float64List _solve(List<Float64List> a, Float64List b) {
  final m = b.length;
  for (var col = 0; col < m; col++) {
    var piv = col;
    for (var r = col + 1; r < m; r++) {
      if (a[r][col].abs() > a[piv][col].abs()) piv = r;
    }
    if (piv != col) {
      final t = a[col];
      a[col] = a[piv];
      a[piv] = t;
      final tb = b[col];
      b[col] = b[piv];
      b[piv] = tb;
    }
    final d = a[col][col];
    if (d == 0) continue;
    for (var r = col + 1; r < m; r++) {
      final f = a[r][col] / d;
      if (f == 0) continue;
      for (var c = col; c < m; c++) {
        a[r][c] -= f * a[col][c];
      }
      b[r] -= f * b[col];
    }
  }
  final x = Float64List(m);
  for (var r = m - 1; r >= 0; r--) {
    var acc = b[r];
    for (var c = r + 1; c < m; c++) {
      acc -= a[r][c] * x[c];
    }
    x[r] = a[r][r] == 0 ? 0 : acc / a[r][r];
  }
  return x;
}

/// 最小平方多項式擬合,回傳擬合值。t 先正規化到 [-1, 1],避免高次項數值不穩。
Float64List _polyFit(Float64List v, int deg) {
  final n = v.length, m = deg + 1;
  final ts = Float64List.fromList([for (var i = 0; i < n; i++) n == 1 ? 0.0 : 2.0 * i / (n - 1) - 1]);
  // 正規方程 A^T A c = A^T v
  final ata = List.generate(m, (_) => Float64List(m));
  final atb = Float64List(m);
  final pw = Float64List(m);
  for (var i = 0; i < n; i++) {
    pw[0] = 1;
    for (var k = 1; k < m; k++) {
      pw[k] = pw[k - 1] * ts[i];
    }
    for (var r = 0; r < m; r++) {
      for (var c = 0; c < m; c++) {
        ata[r][c] += pw[r] * pw[c];
      }
      atb[r] += pw[r] * v[i];
    }
  }
  final coef = _solve(ata, atb);
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

// --------------------------------------------------------------- 果凍效應(捲簾快門)

/// 果凍效應修正:畫面切成幾條水平帶分別估計位移。
const rsBands = 8;

/// 聯合最小平方:位移 = 平移 + 繞畫面中心的小角度旋轉 + 各條帶的平移(總和為 0)。
///
/// 手機感光元件逐行讀出,同一格的上下兩端相差約一格的時間,手震時各條帶的位移不同。
/// 旋轉與果凍的水平錯位在 x 方向上很像(都隨列線性變化),分開估計時果凍會被誤當成旋轉;
/// 旋轉同時會產生隨行變化的垂直位移,放進同一個模型才分得開。
/// 條帶間加平滑項(果凍扭曲沿畫面高度是連續的),特徵點少的條帶也有合理的值。
/// 用 Huber 權重重複解 3 次,降低少數錯誤配對的影響。
///
/// [p0]、[p1] 為前後兩格對應點,交錯排列 [x0, y0, x1, y1, …]。
/// 回傳(繞中心的運動, 各條帶殘差 [bx0, by0, bx1, by1, …])。
/// 對應 python_engine/algorithms.py 的 _fit_bands。
(Motion, Float64List) fitBands(Float64List p0, Float64List p1, int w, int h, {int bands = rsBands}) {
  final n = p0.length ~/ 2;
  final k = 3 + 2 * bands;
  final cx = w / 2, cy = h / 2;
  final band = Int32List(n);
  for (var j = 0; j < n; j++) {
    band[j] = (p0[2 * j + 1] / h * bands).floor().clamp(0, bands - 1);
  }
  final wx = Float64List(n)..fillRange(0, n, 1.0);
  final wy = Float64List(n)..fillRange(0, n, 1.0);
  final sumW = 10.0 * math.sqrt(n), diffW = 0.3 * math.sqrt(n / bands);
  var u = Float64List(k);
  for (var it = 0; it < 3; it++) {
    final ata = List.generate(k, (_) => Float64List(k));
    final atb = Float64List(k);
    void addRow(List<int> idx, List<double> val, double rhs, double wt) {
      final w2 = wt * wt;
      for (var a = 0; a < idx.length; a++) {
        for (var b = 0; b < idx.length; b++) {
          ata[idx[a]][idx[b]] += w2 * val[a] * val[b];
        }
        atb[idx[a]] += w2 * val[a] * rhs;
      }
    }

    for (var j = 0; j < n; j++) {
      final x = p0[2 * j], y = p0[2 * j + 1];
      // dx = tx - θ(y-cy) + bx[band]、dy = ty + θ(x-cx) + by[band]
      addRow([0, 2, 3 + band[j]], [1, -(y - cy), 1], p1[2 * j] - x, wx[j]);
      addRow([1, 2, 3 + bands + band[j]], [1, x - cx, 1], p1[2 * j + 1] - y, wy[j]);
    }
    // 約束列:條帶總和為 0(權重大),條帶二次差分為小(平滑)
    for (final off in [3, 3 + bands]) {
      addRow([for (var i = 0; i < bands; i++) off + i], List.filled(bands, sumW), 0, 1);
      for (var i = 1; i < bands - 1; i++) {
        addRow([off + i - 1, off + i, off + i + 1], [diffW, -2 * diffW, diffW], 0, 1);
      }
    }
    for (var i = 0; i < k; i++) {
      ata[i][i] += 1e-9;
    }
    u = _solve(ata, atb);
    for (var j = 0; j < n; j++) {
      final x = p0[2 * j], y = p0[2 * j + 1];
      final rx = (u[0] - u[2] * (y - cy) + u[3 + band[j]] - (p1[2 * j] - x)).abs();
      final ry = (u[1] + u[2] * (x - cx) + u[3 + bands + band[j]] - (p1[2 * j + 1] - y)).abs();
      wx[j] = math.sqrt(math.min(1.0, 1.0 / math.max(rx, 1e-9))); // Huber,門檻 1px
      wy[j] = math.sqrt(math.min(1.0, 1.0 / math.max(ry, 1e-9)));
    }
  }
  final res = Float64List(2 * bands);
  for (var i = 0; i < bands; i++) {
    res[2 * i] = u[3 + i];
    res[2 * i + 1] = u[3 + bands + i];
  }
  return (Motion(u[0], u[1], u[2]), res);
}

/// 由每格各條帶的殘差位移算出每格各條帶相對整體的偏移(格式同 [fitBands] 的殘差)。
///
/// 累加殘差得到各條帶的相對位置,再減去約 1 秒的移動平均(高通):只留下快速的抖動與扭曲,
/// 去掉估計誤差累積的漂移,以及等速運鏡造成的固定傾斜(那種傾斜看起來很自然,不修)。
/// 修正時把各條帶往反方向移回去。對應 python_engine/algorithms.py 的 rolling_shutter_offsets。
List<Float64List> rollingShutterOffsets(List<Float64List> residuals, double sigma) {
  final n = residuals.length;
  if (n < 3) return [for (final r in residuals) Float64List(r.length)];
  final cols = residuals[0].length;
  final out = List.generate(n, (_) => Float64List(cols));
  final pos = Float64List(n);
  for (var c = 0; c < cols; c++) {
    var acc = 0.0;
    for (var i = 0; i < n; i++) {
      acc += residuals[i][c];
      pos[i] = acc;
    }
    final sm = _gaussSmooth(pos, sigma);
    for (var i = 0; i < n; i++) {
      out[i][c] = pos[i] - sm[i];
    }
  }
  return out;
}
