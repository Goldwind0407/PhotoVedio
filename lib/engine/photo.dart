/// 照片修復:除霧、提高畫質。演算法與影片相同(algorithms.dart),處理順序也與
/// pipeline.dart 的 _renderEntry 一致:除霧 → 提高畫質 → 色調錨定(只有除霧時)。
///
/// 函式是同步的,呼叫端應放在背景 isolate(Isolate.run)執行。
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'algorithms.dart';

class PhotoOptions {
  const PhotoOptions({
    this.enhance = false,
    this.dehaze = false,
    this.dehazeStrength = 0.6,
    this.maxLongSide = 0,
  });

  final bool enhance;
  final bool dehaze;
  final double dehazeStrength; // 0 ~ 1
  final int maxLongSide; // 先縮小到此長邊再處理(預覽用),0 = 只套用安全上限

  bool get any => enhance || dehaze;

  PhotoOptions copyWith({bool? enhance, int? maxLongSide}) => PhotoOptions(
        enhance: enhance ?? this.enhance,
        dehaze: dehaze,
        dehazeStrength: dehazeStrength,
        maxLongSide: maxLongSide ?? this.maxLongSide,
      );

  @override
  bool operator ==(Object other) =>
      other is PhotoOptions &&
      other.enhance == enhance &&
      other.dehaze == dehaze &&
      other.dehazeStrength == dehazeStrength &&
      other.maxLongSide == maxLongSide;

  @override
  int get hashCode => Object.hash(enhance, dehaze, dehazeStrength, maxLongSide);
}

class PhotoResult {
  const PhotoResult(this.bytes, this.width, this.height, {required this.downscaled});

  final Uint8List bytes; // PNG
  final int width, height;

  /// 原圖長邊超過 [photoMaxLongSide],輸出已縮小(不含 [PhotoOptions.maxLongSide] 的預覽縮小)。
  final bool downscaled;
}

/// 記憶體安全上限:除霧與色調錨定的峰值記憶體與像素數成正比,實測 12MP 約多用 350MB、
/// 20MP 約 550MB;48MP 原圖會超過 1.5GB,手機一定被系統砍掉,所以長邊超過此值先縮小。
const photoMaxLongSide = 5000;

/// [bytes] 為 JPG/PNG 等編碼過的圖片。回傳處理後的 PNG(無損,方便後續再調色、裁切)。
/// 解碼時已依 EXIF 轉正,所以輸出不帶方向資訊。
PhotoResult processPhoto(Uint8List bytes, PhotoOptions o) {
  final mats = <cv.Mat>[];
  cv.Mat keep(cv.Mat m) {
    mats.add(m);
    return m;
  }

  Dehazer? dehazer;
  try {
    var cur = keep(cv.imdecode(bytes, cv.IMREAD_COLOR));
    if (cur.isEmpty) throw Exception('無法讀取圖片');

    final long = math.max(cur.cols, cur.rows);
    final cap = o.maxLongSide > 0 ? math.min(o.maxLongSide, photoMaxLongSide) : photoMaxLongSide;
    final downscaled = long > photoMaxLongSide;
    if (long > cap) {
      final s = cap / long;
      cur = keep(
        cv.resize(cur, (
          (cur.cols * s).round(),
          (cur.rows * s).round(),
        ), interpolation: cv.INTER_AREA),
      );
    }

    final ref = cur;
    if (o.dehaze) {
      dehazer = Dehazer(strength: o.dehazeStrength);
      cur = keep(dehazer(cur));
    }
    if (o.enhance) {
      // 照片無法從檔案判斷壓縮程度,用中等降噪。
      cur = keep(DetailEnhancer()(cur));
    }
    if (o.dehaze) {
      cur = keep(ToneAnchor()(ref, cur));
    }

    final (ok, out) = cv.imencode(
      '.png',
      cur,
      params: cv.VecI32.fromList([cv.IMWRITE_PNG_COMPRESSION, 1]),
    );
    if (!ok) throw Exception('圖片編碼失敗');
    return PhotoResult(out, cur.cols, cur.rows, downscaled: downscaled);
  } finally {
    dehazer?.dispose();
    for (final m in mats) {
      m.dispose();
    }
  }
}
