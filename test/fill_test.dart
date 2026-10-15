import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:photo_video_editor/engine/algorithms.dart';

void main() {
  test('邊緣帶:整條左邊露出時只回傳左帶,不退化成整張', () {
    const w = 20, h = 10;
    final cov = Uint8List(w * h)..fillRange(0, w * h, 1);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < 3; x++) {
        cov[y * w + x] = 0; // 左邊 3 欄整條露出
      }
    }
    cov[0 * w + 10] = 0; // 上緣中間露一點
    cov[1 * w + 10] = 0;
    expect(borderBands(cov, w, h), [(0, 0, 3, 10), (3, 0, 17, 2)]);
    expect(borderBands(Uint8List(w * h)..fillRange(0, w * h, 1), w, h), isEmpty);
  });

  test('補邊:平移後露出的邊用相鄰格補上', () {
    // 一張有紋理的圖;相鄰格是同一場景(鏡頭沒動),目前格往右平移 6 像素 → 左邊露出 6 欄
    final img = cv.Mat.zeros(40, 60, cv.MatType.CV_8UC3);
    final d = img.data;
    for (var i = 0; i < d.length; i++) {
      d[i] = (i * 7) % 255;
    }
    final frames = {0: img.clone(), 1: img, 2: img.clone()};
    const zero = Motion(0, 0, 0);
    final traj = [zero, zero, zero];
    // 第 1 格的修正量 +6:路徑 = 軌跡 + 修正
    final path = [zero, const Motion(6, 0, 0), zero];
    final (out, cov) = applyStabilizationWithCoverage(img, const Motion(6, 0, 0), 1.0);
    expect(cv.countNonZero(cov), 40 * 54);
    // 相鄰格畫到第 1 格的座標(修正 = 路徑_1 − 軌跡_j = +6)也露出同樣的邊 → 補不到,維持延伸邊緣
    expect(fillBorders(out, cov, 1, frames, path, traj, 1.0), 40 * 6);
    // 相鄰格的鏡頭往左移過 6 像素(軌跡 +6):畫過來剛好蓋住 → 全部補上
    final traj2 = [const Motion(6, 0, 0), zero, const Motion(6, 0, 0)];
    final (out2, cov2) = applyStabilizationWithCoverage(img, const Motion(6, 0, 0), 1.0);
    expect(fillBorders(out2, cov2, 1, frames, path, traj2, 1.0), 0);
  });
}
