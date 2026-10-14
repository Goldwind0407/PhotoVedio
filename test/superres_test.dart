import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:photo_video_editor/engine/superres.dart';

/// 需要 onnxruntime 原生函式庫:Linux 上執行前設定
/// LD_LIBRARY_PATH=~/.pub-cache/hosted/pub.dev/onnxruntime-<版本>/linux
void main() {
  final model = File(srModelAsset).readAsBytesSync();

  cv.Mat testImage() {
    // 有漸層、邊緣與細紋的合成圖
    final m = cv.Mat.zeros(150, 200, cv.MatType.CV_8UC3);
    final d = m.data;
    for (var y = 0; y < 150; y++) {
      for (var x = 0; x < 200; x++) {
        final i = (y * 200 + x) * 3;
        d[i] = ((x * 255) ~/ 200) ^ ((y ~/ 7) * 37 & 0xff);
        d[i + 1] = (x ~/ 10 + y ~/ 10).isEven ? 220 : 30;
        d[i + 2] = (y * 255) ~/ 150;
      }
    }
    return m;
  }

  test('放大 4 倍,分塊結果與整張相同', () {
    final img = testImage();
    final whole = SuperResolver(model, accelerate: false, tile: 1000);
    final tiled = SuperResolver(model, accelerate: false, tile: 64);
    final a = whole(img), b = tiled(img);
    expect((a.cols, a.rows), (800, 600));
    final diff = cv.absDiff(a, b);
    expect(cv.minMaxLoc(diff.reshape(1, 0)).$2, 0); // 最大差異為 0
    final dump = Platform.environment['SR_DUMP'];
    if (dump != null) {
      cv.imwrite('$dump/in.png', img);
      cv.imwrite('$dump/out_dart.png', a);
    }
    whole.release();
    tiled.release();
  });

  test('輸出尺寸:長邊對齊目標、最多 4 倍、取偶數', () {
    expect(srTargetSize(592, 1280, 3840), (1776, 3840));
    expect(srTargetSize(1080, 1920, 3840), (2160, 3840));
    expect(srTargetSize(300, 200, 3840), (1200, 800));
  });
}
