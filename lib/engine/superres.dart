/// AI 超解析度(Real-ESRGAN realesr-general-x4v3,4 倍放大),對應 python_engine/superres.py。
///
/// 模型很小(120 萬參數),但每個輸入像素約要 240 萬次浮點運算,592×1280 的影格約 1.8 TFLOP,
/// 只用 CPU 會非常慢。手機上優先交給硬體加速:Android 用 NNAPI(GPU/NPU)、iPhone 用 CoreML;
/// 不支援時退回 CPU。
///
/// 大畫面分塊處理:每塊多取 [_overlap] 像素(大於網路 34 層 3×3 卷積的作用範圍),只寫回中間,
/// 結果與整張處理相同,記憶體只跟塊大小有關。
///
/// 函式是同步的,應在背景 isolate 執行;ONNX session 不能跨 isolate,要在使用的 isolate 裡建立。
library;

import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime/onnxruntime.dart';
// 輸出張量直接讀原生記憶體:套件的 value 會轉成巢狀 List,4 倍放大後有數千萬個數值,太慢也太耗記憶體。
// ignore: implementation_imports
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart' as bg;
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// App 內建的模型(與電腦版共用)。
const srModelAsset = 'assets/models/realesr-general-x4v3.onnx';
const srScale = 4;
const _overlap = 40;

/// 超解析度後的輸出尺寸:長邊為 [longSide](最多放大 4 倍),取偶數。對應 superres.py 的 target_size。
(int, int) srTargetSize(int w, int h, int longSide) {
  final s = math.min(srScale.toDouble(), longSide / math.max(w, h));
  return (math.max(2, (w * s).round() ~/ 2 * 2), math.max(2, (h * s).round() ~/ 2 * 2));
}

class SuperResolver {
  SuperResolver._(this._session, this.backend, this._tile);

  final OrtSession _session;
  final int _tile;

  /// 實際使用的運算後端(NNAPI / CoreML / XNNPACK / CPU),顯示給使用者參考。
  final String backend;

  /// [model] 為 ONNX 檔內容。[accelerate] 為 false 時只用 CPU(測試用)。
  /// [tile] 為分塊大小(輸入像素),一般不需要改。
  factory SuperResolver(Uint8List model, {bool accelerate = true, int tile = 320}) {
    OrtEnv.instance.init();
    final opts = OrtSessionOptions();
    var backend = 'CPU';
    if (accelerate) {
      if (Platform.isAndroid) {
        if (opts.appendNnapiProvider(NnapiFlags.useFp16)) {
          backend = 'NNAPI';
        } else if (opts.appendXnnpackProvider()) {
          backend = 'XNNPACK';
        }
      } else if (Platform.isIOS || Platform.isMacOS) {
        if (opts.appendCoreMLProvider(CoreMLFlags.useNone)) backend = 'CoreML';
      }
    }
    try {
      return SuperResolver._(OrtSession.fromBuffer(model, opts), backend, tile);
    } finally {
      opts.release();
    }
  }

  void release() => _session.release();

  /// 8UC3 BGR → 4 倍大的 8UC3 BGR。回傳新的 Mat,由呼叫端 dispose。
  cv.Mat call(cv.Mat bgr) {
    final h = bgr.rows, w = bgr.cols;
    final out = cv.Mat.zeros(h * srScale, w * srScale, cv.MatType.CV_8UC3);
    for (var y0 = 0; y0 < h; y0 += _tile) {
      for (var x0 = 0; x0 < w; x0 += _tile) {
        final y1 = math.min(h, y0 + _tile), x1 = math.min(w, x0 + _tile);
        final ya = math.max(0, y0 - _overlap), xa = math.max(0, x0 - _overlap);
        final yb = math.min(h, y1 + _overlap), xb = math.min(w, x1 + _overlap);
        final src = bgr.region(cv.Rect(xa, ya, xb - xa, yb - ya));
        final res = _run(src);
        final inner = res.region(cv.Rect((x0 - xa) * srScale, (y0 - ya) * srScale,
            (x1 - x0) * srScale, (y1 - y0) * srScale));
        final dst = out.region(cv.Rect(x0 * srScale, y0 * srScale, (x1 - x0) * srScale, (y1 - y0) * srScale));
        inner.copyTo(dst);
        for (final m in [src, res, inner, dst]) {
          m.dispose();
        }
      }
    }
    return out;
  }

  /// 執行一塊:8UC3 BGR → 4 倍大的 8UC3 BGR。
  cv.Mat _run(cv.Mat bgr) {
    final h = bgr.rows, w = bgr.cols, n = h * w;
    // HWC BGR 0~255 → NCHW RGB 0~1
    final input = Float32List(3 * n);
    final f = bgr.convertTo(cv.MatType.CV_32FC3, alpha: 1 / 255);
    final ch = cv.split(f);
    for (var c = 0; c < 3; c++) {
      input.setRange(c * n, (c + 1) * n, _f32(ch[2 - c]));
    }
    ch.dispose();
    f.dispose();

    final tensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, h, w]);
    final runOpts = OrtRunOptions();
    final outs = _session.run(runOpts, {'input': tensor});
    tensor.release();
    runOpts.release();
    final out = outs.first!;
    try {
      final pp = calloc<ffi.Pointer<ffi.Void>>();
      final status = OrtEnv.instance.ortApiPtr.ref.GetTensorMutableData.asFunction<
          bg.OrtStatusPtr Function(
              ffi.Pointer<bg.OrtValue>, ffi.Pointer<ffi.Pointer<ffi.Void>>)>()(out.ptr, pp);
      OrtStatus.checkOrtStatus(status);
      final oh = h * srScale, ow = w * srScale, on = oh * ow;
      final data = pp.value.cast<ffi.Float>().asTypedList(3 * on);
      calloc.free(pp);
      // NCHW RGB → BGR 三個平面合成一張
      final planes = <cv.Mat>[];
      for (final c in [2, 1, 0]) {
        final m = cv.Mat.zeros(oh, ow, cv.MatType.CV_32FC1);
        _f32(m).setRange(0, on, data, c * on);
        planes.add(m);
      }
      final vec = cv.VecMat.fromList(planes);
      final merged = cv.merge(vec);
      // convertTo 到 8U 會四捨五入並飽和裁切,等同 clip(x·255 + 0.5)
      final result = merged.convertTo(cv.MatType.CV_8UC3, alpha: 255);
      merged.dispose();
      vec.dispose();
      for (final p in planes) {
        p.dispose();
      }
      return result;
    } finally {
      for (final o in outs) {
        o?.release();
      }
    }
  }
}

Float32List _f32(cv.Mat m) {
  final d = m.data;
  return d.buffer.asFloat32List(d.offsetInBytes, d.lengthInBytes ~/ 4);
}
