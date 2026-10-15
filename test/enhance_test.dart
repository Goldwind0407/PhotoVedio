import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:photo_video_editor/engine/algorithms.dart';

void main() {
  test('壓縮程度:轉傳過的影片接近 1,原檔接近 0', () {
    // 592×1280 H.264 2.7 Mbps(LINE 轉傳)
    expect(compressionLevel(2.714e6, 592, 1280, 29.9, hevc: false), closeTo(1.0, 1e-9));
    // iPhone 4K HEVC 約 45 Mbps
    expect(compressionLevel(45e6, 3840, 2160, 30, hevc: true), 0);
    // 位元率未知
    expect(compressionLevel(0, 1920, 1080, 30, hevc: false), 0.5);
  });

  test('提高畫質:輸出尺寸與格式不變', () {
    final img = cv.Mat.zeros(120, 160, cv.MatType.CV_8UC3);
    final d = img.data;
    for (var i = 0; i < d.length; i++) {
      d[i] = (i * 37) % 251;
    }
    final out = DetailEnhancer.level(EnhanceLevel.high, compression: 1)(img);
    expect((out.cols, out.rows, out.type), (160, 120, cv.MatType.CV_8UC3));
    final dump = Platform.environment['ENH_DUMP'];
    if (dump != null) {
      final src = cv.imread('$dump/in.png');
      cv.imwrite('$dump/out_dart.png', DetailEnhancer.level(EnhanceLevel.medium, compression: 1)(src));
    }
  });
}
