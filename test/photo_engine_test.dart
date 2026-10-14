import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:photo_video_editor/engine/photo.dart';

void main() {
  test('超過安全上限的照片會縮小並回報', () {
    final m = cv.Mat.fromScalar(3000, 6000, cv.MatType.CV_8UC3, cv.Scalar(90, 120, 150, 0));
    final (_, png) = cv.imencode('.png', m);
    m.dispose();
    final r = processPhoto(png, const PhotoOptions(dehaze: true, enhance: true));
    expect(r.downscaled, isTrue);
    expect(r.width, photoMaxLongSide);
    expect(r.height, 2500);
    final p = processPhoto(png, const PhotoOptions(dehaze: true, maxLongSide: 1600));
    expect(p.width, 1600);
    expect(p.downscaled, isTrue);
  });

  test('小照片維持原尺寸', () {
    final m = cv.Mat.fromScalar(300, 400, cv.MatType.CV_8UC3, cv.Scalar(90, 120, 150, 0));
    final (_, png) = cv.imencode('.png', m);
    m.dispose();
    final r = processPhoto(png, const PhotoOptions(enhance: true));
    expect((r.width, r.height, r.downscaled), (400, 300, false));
  });

  test('PhotoOptions 相等比較', () {
    const a = PhotoOptions(dehaze: true, dehazeStrength: 0.5);
    expect(a, const PhotoOptions(dehaze: true, dehazeStrength: 0.5));
    expect(a == a.copyWith(enhance: true), isFalse);
  });
}
