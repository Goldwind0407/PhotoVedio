/// 影片串流處理,對應 python_engine/pipeline.py。
///
/// 影片的解碼/編碼交給 FFmpeg(穩定、支援 HEVC/旋轉旗標/音軌),
/// OpenCV 只負責影格運算,兩者之間用 FFmpegKit 的具名管道傳原始 BGR 影格:
///
///   [防手震才有] FFmpeg 解碼並縮小成灰階 ──管道──▶ isolate 估計晃動
///   FFmpeg 解碼 ──管道──▶ isolate 防手震/除霧/畫質 ──管道──▶ FFmpeg 編碼(H.264 + 原音軌)
///
/// (OpenCV 內建的 VideoCapture 在 Android 上遇到解碼失敗會直接讓整個程序 SIGSEGV,故不使用。)
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffmpeg_kit_min_gpl/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_min_gpl/ffmpeg_kit_config.dart';
import 'package:ffmpeg_kit_min_gpl/ffprobe_kit.dart';
import 'package:ffmpeg_kit_min_gpl/return_code.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'algorithms.dart';
import 'superres.dart';

class EngineOptions {
  const EngineOptions({
    this.enhance = false,
    this.enhanceLevel = EnhanceLevel.medium,
    this.dehaze = false,
    this.dehazeStrength = 0.6,
    this.dehazeChroma = 0.35,
    this.toneAnchor = true,
    this.anchorChroma = 0.0,
    this.stabilize = false,
    this.stabilizeLevel = StabilizeLevel.standard,
    this.zoom = 0,
    this.maxFrames = 0,
    this.shakeThreshold = 0.4e-3,
    this.maxLongSide = 0,
    this.superRes = false,
    this.srLongSide = 3840,
    this.srModel,
  });

  final bool enhance; // 提高畫質:雜訊感知、邊緣感知的亮度細節增強(DetailEnhancer)
  final EnhanceLevel enhanceLevel;
  final bool dehaze;
  final double dehazeStrength;
  final bool toneAnchor; // 最後把亮度/色彩拉回原片(建議開啟)
  final double anchorChroma; // 錨定時色度取自處理後影格的比例
  final double dehazeChroma; // 去霧後色度取自去霧結果的比例(越小越接近原片色彩)
  final bool stabilize;
  final StabilizeLevel stabilizeLevel;
  final double zoom; // 防手震裁切放大;0 = 依修正量自動計算(剛好蓋住黑邊)
  final int maxFrames; // 只處理前 N 格(測試/預覽用),0 = 全部
  // 晃動門檻(高頻抖動 / 畫面長邊)。低於此值視為穩定、不做防手震。
  // 依實測:明顯手震約 1.6e-3,一般手持約 0.2~0.6e-3;0.4e-3 約等於 1080p 上 0.8px 的抖動。
  final double shakeThreshold;
  // 工作解析度上限(長邊像素)。4K 單格 BGR 就 25 MB、浮點中間結果更大,
  // 手機記憶體撐不住,所以解碼時就先縮小。0 = 保持原尺寸。
  final int maxLongSide;
  // AI 超解析度:放大 4 倍後縮到長邊 [srLongSide](3840 = 4K)。開啟時取代 [enhance]。
  // [srModel] 為 ONNX 模型內容(見 superres.dart 的 srModelAsset)。
  final bool superRes;
  final int srLongSide;
  final Uint8List? srModel;

  bool get any => enhance || dehaze || stabilize || superRes;
}

typedef ProgressCallback = void Function(String stage, double fraction);

/// 處理結果摘要,讓畫面告訴使用者防手震實際做了什麼。
class VideoReport {
  const VideoReport({
    this.shakeScore,
    this.stabilized = false,
    this.zoom = 1.0,
    this.locked = false,
    this.outWidth = 0,
    this.outHeight = 0,
  });

  final double? shakeScore; // null = 沒開防手震
  final bool stabilized; // false = 偵測為穩定而略過
  final double zoom; // 防手震的放大倍率
  final bool locked; // 用了三腳架鎖定
  final int outWidth, outHeight; // 輸出影片尺寸

  VideoReport withSize(int w, int h) => VideoReport(
      shakeScore: shakeScore, stabilized: stabilized, zoom: zoom, locked: locked, outWidth: w, outHeight: h);
}

/// 晃動分數:軌跡偏離其移動平均(視窗 15 格)的平均距離,除以畫面長邊。
/// 只反映高頻抖動;緩慢平移、刻意運鏡不會被算進去。
double shakeScore(List<Motion> motions, int longSide) {
  final n = motions.length;
  if (n < 16) return 0;
  final xs = Float64List(n), ys = Float64List(n);
  var x = 0.0, y = 0.0;
  for (var i = 0; i < n; i++) {
    x += motions[i].dx;
    y += motions[i].dy;
    xs[i] = x;
    ys[i] = y;
  }
  const r = 7;
  int refl(int i) => i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i);
  var sum = 0.0;
  for (var i = 0; i < n; i++) {
    var mx = 0.0, my = 0.0;
    for (var k = -r; k <= r; k++) {
      mx += xs[refl(i + k)];
      my += ys[refl(i + k)];
    }
    mx /= 2 * r + 1;
    my /= 2 * r + 1;
    sum += math.sqrt(math.pow(xs[i] - mx, 2) + math.pow(ys[i] - my, 2));
  }
  return sum / n / longSide;
}

/// 晃動分析用的長邊像素數(約 720p)。輸出仍用原解析度,
/// 4K 影片的光流運算量因此降到約 1/9。
const analysisLongSide = 1280;

// ------------------------------------------------------------------ 影片資訊

class VideoInfo {
  const VideoInfo(this.width, this.height, this.fps, this.frames, this.color);
  final int width, height; // 已轉正、且為偶數
  final double fps;
  final int frames;
  final ColorInfo color;
}

/// 原片的色彩規格。iPhone 預設錄的是 HDR(HEVC 10-bit、BT.2020、HLG 曲線),
/// 解碼/編碼都必須沿用同一套矩陣與標註,否則輸出會被播放器當成一般 SDR 影片,
/// 看起來偏暗、色彩跑掉。
class ColorInfo {
  const ColorInfo({required this.primaries, required this.transfer, required this.space});
  final String primaries; // 例:bt2020、bt709
  final String transfer; // 例:arib-std-b67(HLG)、smpte2084(PQ)、bt709
  final String space; // 例:bt2020nc、bt709

  bool get isHdr => transfer == 'arib-std-b67' || transfer == 'smpte2084';

  /// swscale 的 YUV↔RGB 矩陣名稱。
  String get swsMatrix => space.startsWith('bt2020')
      ? 'bt2020'
      : (space == 'bt709' ? 'bt709' : (space == 'smpte170m' || space == 'bt470bg' ? 'bt601' : 'bt709'));

  static ColorInfo fromProps(Map<dynamic, dynamic> p) {
    String pick(String key, String fallback) {
      final v = '${p[key] ?? ''}';
      return v.isEmpty || v == 'unknown' || v == 'reserved' ? fallback : v;
    }

    final space = pick('color_space', 'bt709');
    final hdrSpace = space.startsWith('bt2020');
    return ColorInfo(
      primaries: pick('color_primaries', hdrSpace ? 'bt2020' : 'bt709'),
      transfer: pick('color_transfer', 'bt709'),
      space: space,
    );
  }
}

double _parseRate(Object? r) {
  final parts = '$r'.split('/');
  final n = double.tryParse(parts[0]) ?? 0;
  final d = parts.length > 1 ? (double.tryParse(parts[1]) ?? 1) : 1;
  return d == 0 ? 0 : n / d;
}

Future<VideoInfo> _probe(String src) async {
  final session = await FFprobeKit.getMediaInformation(src);
  final info = session.getMediaInformation();
  if (info == null) throw Exception('無法讀取影片資訊');
  final v = info.getStreams().where((s) => s.getType() == 'video').firstOrNull;
  if (v == null) throw Exception('檔案中沒有影像軌');

  var w = v.getWidth() ?? 0, h = v.getHeight() ?? 0;
  final props = v.getAllProperties() ?? const {};
  // 旋轉旗標:新版 FFmpeg 放在 side_data_list,舊版在 tags.rotate。
  num rot = 0;
  final sd = props['side_data_list'];
  if (sd is List) {
    for (final e in sd) {
      if (e is Map && e['rotation'] != null) {
        rot = num.tryParse('${e['rotation']}') ?? rot;
      }
    }
  }
  final tags = props['tags'];
  if (rot == 0 && tags is Map && tags['rotate'] != null) {
    rot = num.tryParse('${tags['rotate']}') ?? 0;
  }
  if (rot.abs().round() % 180 == 90) (w, h) = (h, w); // FFmpeg 會自動轉正

  var fps = _parseRate(props['avg_frame_rate']);
  if (fps <= 0 || fps > 240) fps = _parseRate(props['r_frame_rate']);
  if (fps <= 0 || fps > 240) fps = 30;
  final duration = double.tryParse(info.getDuration() ?? '') ?? 0;
  final nb = int.tryParse('${props['nb_frames']}');
  final frames = nb ?? (duration * fps).round();
  if (w <= 0 || h <= 0 || frames <= 0) throw Exception('影片資訊不完整');
  return VideoInfo(w ~/ 2 * 2, h ~/ 2 * 2, fps, frames, ColorInfo.fromProps(props));
}

// ------------------------------------------------------------------ FFmpeg / 管道

String _lastFfmpegError = '';

// ------------------------------------------------------------------ 取消

/// 使用者取消處理時 [processVideo] 丟出的例外。
class VideoCancelled implements Exception {
  const VideoCancelled();
  @override
  String toString() => '已取消';
}

bool _cancelRequested = false;

/// 取消進行中的 [processVideo]:停止所有 FFmpeg 工作,背景 isolate 會因管道關閉而結束。
Future<void> cancelVideoProcessing() async {
  _cancelRequested = true;
  await FFmpegKit.cancel();
}

void _checkCancelled() {
  if (_cancelRequested) throw const VideoCancelled();
}

/// 啟動 FFmpeg(非同步),完成時回傳 session 的結果。
Future<ReturnCode?> _ffmpeg(List<String> args, {void Function()? onDone}) {
  final c = Completer<ReturnCode?>();
  FFmpegKit.executeWithArgumentsAsync(args, (session) async {
    final code = await session.getReturnCode();
    if (!ReturnCode.isSuccess(code)) {
      final logs = await session.getAllLogsAsString();
      final tail = (logs ?? '').split('\n').where((l) => l.trim().isNotEmpty);
      _lastFfmpegError = tail.toList().reversed.take(4).toList().reversed.join('\n');
    }
    onDone?.call();
    c.complete(code);
  });
  return c.future;
}

/// FFmpeg 若提早失敗,isolate 會卡在開啟管道。從另一端開一下再關,讓它讀到 EOF/寫入失敗。
void _unblockPipe(String path, {required bool readerIsStuck}) {
  Isolate.run(() {
    try {
      File(path)
          .openSync(mode: readerIsStuck ? FileMode.writeOnly : FileMode.read)
          .closeSync();
    } catch (_) {}
  });
}

// ------------------------------------------------------------------ isolate 工作

class _Progress {
  const _Progress(this.stage, this.fraction);
  final String stage;
  final double fraction;
}

class _Result {
  const _Result(this.value);
  final Object? value;
}

class _Failure {
  const _Failure(this.message);
  final String message;
}

typedef _Entry = void Function((Object, SendPort));

/// 在背景 isolate 執行 [entry];回傳它送回的結果。
/// entry 必須是頂層函式,job 只能含純資料(閉包會捕獲不可傳送的物件)。
Future<T> _runIsolate<T>(_Entry entry, Object job, void Function(_Progress) onProgress) async {
  final port = ReceivePort();
  final done = Completer<T>();
  port.listen((msg) {
    if (msg is _Progress) {
      onProgress(msg);
    } else if (msg is _Result) {
      done.complete(msg.value as T);
    } else if (msg is _Failure) {
      done.completeError(Exception(msg.message));
    }
  });
  await Isolate.spawn(entry, (job, port.sendPort));
  try {
    return await done.future;
  } finally {
    port.close();
  }
}

/// 阻塞式讀滿 [buf];遇到 EOF 回傳 false。
bool _readFully(RandomAccessFile f, Uint8List buf) {
  var off = 0;
  while (off < buf.length) {
    final n = f.readIntoSync(buf, off, buf.length);
    if (n <= 0) return false;
    off += n;
  }
  return true;
}

// ---- 階段 1:晃動分析 ----

const _motionStride = 3 + 2 * rsBands;

class _AnalysisJob {
  const _AnalysisJob(this.pipe, this.w, this.h, this.frames);
  final String pipe;
  final int w, h, frames;
}

void _analysisEntry((Object, SendPort) args) {
  final (job, port) = (args.$1 as _AnalysisJob, args.$2);
  try {
    final f = File(job.pipe).openSync();
    var prev = cv.Mat.zeros(job.h, job.w, cv.MatType.CV_8UC1);
    var cur = cv.Mat.zeros(job.h, job.w, cv.MatType.CV_8UC1);
    // 每格 3 + 2×rsBands 個值:dx, dy, da, 各條帶果凍殘差;第一格沒有前一格,全為 0。
    final out = List<double>.filled(_motionStride, 0, growable: true);
    if (_readFully(f, prev.data)) {
      for (var i = 1; i < job.frames; i++) {
        if (!_readFully(f, cur.data)) break;
        final (m, res) = estimateMotionBands(prev, cur);
        out
          ..addAll([m.dx, m.dy, m.da])
          ..addAll(res);
        (prev, cur) = (cur, prev);
        if (i % 10 == 0) port.send(_Progress('分析晃動', i / job.frames));
      }
    }
    f.closeSync();
    prev.dispose();
    cur.dispose();
    port.send(_Result(Float64List.fromList(out)));
  } catch (e) {
    port.send(_Failure('$e'));
  }
}

// ---- 階段 2:套用處理 ----

class _RenderJob {
  const _RenderJob({
    required this.inPipe,
    required this.outPipe,
    required this.w,
    required this.h,
    required this.outW,
    required this.outH,
    required this.frames,
    required this.options,
    required this.corrections,
    required this.rollingShutter,
    required this.zoom,
  });
  final String inPipe, outPipe;
  final int w, h, outW, outH, frames;
  final EngineOptions options;
  final Float64List? corrections; // 每格 3 個值:dx, dy, da
  final Float64List? rollingShutter; // 每格 2×rsBands 個值:各條帶的果凍偏移
  final double zoom; // 防手震放大倍率
}

void _renderEntry((Object, SendPort) args) {
  final (job, port) = (args.$1 as _RenderJob, args.$2);
  final o = job.options;
  Dehazer? dehazer;
  ToneAnchor? anchor;
  SuperResolver? sr;
  try {
    final fin = File(job.inPipe).openSync();
    final fout = File(job.outPipe).openSync(mode: FileMode.writeOnly);
    // 新版畫質增強不改變色調;只有除霧會改變亮度分佈,才需要錨定回原片。
    anchor = o.toneAnchor && o.dehaze ? ToneAnchor(chroma: o.anchorChroma) : null;
    sr = o.superRes ? SuperResolver(o.srModel!) : null;
    // AI 放大本身就會補細節、讓邊緣銳利,再疊細節增強反而過頭,所以兩者擇一。
    final enhancer = o.enhance && sr == null
        ? DetailEnhancer.level(o.enhanceLevel)
        : null;
    dehazer = o.dehaze ? Dehazer(strength: o.dehazeStrength, chroma: o.dehazeChroma) : null;
    final frame = cv.Mat.zeros(job.h, job.w, cv.MatType.CV_8UC3);
    final c = job.corrections;
    final rsAll = job.rollingShutter;
    const rsn = 2 * rsBands;

    for (var i = 0; i < job.frames; i++) {
      if (!_readFully(fin, frame.data)) break;
      var cur = frame;
      cv.Mat step(cv.Mat next) {
        if (!identical(cur, frame)) cur.dispose();
        return cur = next;
      }

      if (c != null && 3 * i + 2 < c.length) {
        final rs = rsAll != null && rsn * (i + 1) <= rsAll.length
            ? Float64List.sublistView(rsAll, rsn * i, rsn * (i + 1))
            : null;
        step(applyStabilization(cur, Motion(c[3 * i], c[3 * i + 1], c[3 * i + 2]), job.zoom,
            rs: rs));
      }
      // 色調錨定的參照:防手震之後、除霧與畫質之前的影格(位置對得上,亮度色彩是原片的)。
      final ref = anchor != null ? cur.clone() : null;
      if (dehazer != null) step(dehazer(cur));
      if (enhancer != null) step(enhancer(cur));
      if (sr != null) step(sr(cur));
      if (cur.cols != job.outW || cur.rows != job.outH) {
        // AI 放大 4 倍後縮到目標尺寸用 INTER_AREA(縮小不產生鋸齒);其餘放大用 Lanczos。
        final shrink = cur.cols > job.outW;
        step(cv.resize(cur, (job.outW, job.outH),
            interpolation: shrink ? cv.INTER_AREA : cv.INTER_LANCZOS4));
      }
      if (ref != null) {
        // 開「放大」時輸出比參照圖大,參照圖也要放大到相同尺寸。
        final r = ref.cols == cur.cols && ref.rows == cur.rows
            ? ref
            : cv.resize(ref, (cur.cols, cur.rows), interpolation: cv.INTER_LINEAR);
        step(anchor!(r, cur));
        if (!identical(r, ref)) r.dispose();
        ref.dispose();
      }
      fout.writeFromSync(cur.data);
      if (!identical(cur, frame)) cur.dispose();
      // AI 放大每格要好幾秒,每格都回報進度
      if (sr != null || i % 5 == 0) {
        port.send(_Progress(sr != null ? '輸出影片(AI 放大)' : '輸出影片', (i + 1) / job.frames));
      }
    }
    fin.closeSync();
    fout.closeSync(); // 關閉寫入端 → 編碼器讀到 EOF 後收尾
    frame.dispose();
    port.send(const _Result(null));
  } catch (e) {
    port.send(_Failure('$e'));
  } finally {
    sr?.release();
    dehazer?.dispose();
  }
}

// ------------------------------------------------------------------ 公開入口

Future<VideoReport> processVideo(
  String src,
  String dst,
  EngineOptions options, {
  ProgressCallback? onProgress,
}) async {
  void progress(_Progress p) => onProgress?.call(p.stage, p.fraction);
  _cancelRequested = false;
  final info = await _probe(src);
  _checkCancelled();
  final frames =
      options.maxFrames > 0 ? math.min(info.frames, options.maxFrames) : info.frames;
  // 工作解析度:長邊超過上限就等比縮小(取偶數)。
  final cap = options.maxLongSide;
  final fit = cap > 0 ? math.min(1.0, cap / math.max(info.width, info.height)) : 1.0;
  final workW = math.max(2, (info.width * fit).round() ~/ 2 * 2);
  final workH = math.max(2, (info.height * fit).round() ~/ 2 * 2);
  final pipes = <String>[];
  Future<String> newPipe() async {
    final p = await FFmpegKitConfig.registerNewFFmpegPipe();
    if (p == null) throw Exception('無法建立 FFmpeg 管道');
    pipes.add(p);
    return p;
  }

  try {
    Float64List? corrections, rollingShutter;
    var zoom = 1.0;
    var report = const VideoReport();

    // ---- 階段 1:防手震才需要。FFmpeg 直接輸出縮小後的灰階影格,不必在 Dart 端縮圖。 ----
    if (options.stabilize) {
      final s = math.min(1.0, analysisLongSide / math.max(workW, workH));
      final aw = math.max(2, (workW * s).round() ~/ 2 * 2);
      final ah = math.max(2, (workH * s).round() ~/ 2 * 2);
      final pipe = await newPipe();
      var finished = false;
      final decode = _ffmpeg([
        '-y', '-threads', '2', '-i', src, '-an', '-frames:v', '$frames',
        '-vf', 'scale=$aw:$ah,format=gray',
        '-f', 'rawvideo', '-pix_fmt', 'gray', pipe, //
      ], onDone: () {
        if (!finished) _unblockPipe(pipe, readerIsStuck: true);
      });
      final motions = await _runIsolate<Float64List>(
        _analysisEntry,
        _AnalysisJob(pipe, aw, ah, frames),
        progress,
      ).whenComplete(() => finished = true);
      final code = await decode;
      if (!ReturnCode.isSuccess(code) && motions.length < 2 * _motionStride) {
        throw Exception('解碼失敗:$_lastFfmpegError');
      }
      // 位移是縮小圖上的像素,放大回原解析度;旋轉角與尺度無關。
      final sx = workW / aw, sy = workH / ah;
      final n = motions.length ~/ _motionStride;
      Motion motionAt(int i) {
        final o = i * _motionStride;
        return Motion(motions[o] * sx, motions[o + 1] * sy, motions[o + 2]);
      }

      final list = [for (var i = 0; i < n; i++) motionAt(i)];
      final residuals = [
        for (var i = 0; i < n; i++)
          Float64List.fromList([
            for (var b = 0; b < rsBands; b++) ...[
              motions[i * _motionStride + 3 + 2 * b] * sx,
              motions[i * _motionStride + 4 + 2 * b] * sy,
            ],
          ]),
      ];
      // 自動判斷:不晃的影片不做修正(裁切放大與重新取樣本身會損失畫質)。
      final jitter = shakeScore(list, math.max(workW, workH));
      // 鎖定模式連緩慢漂移也要消除,不看晃動門檻。
      final shaky =
          jitter >= options.shakeThreshold || options.stabilizeLevel == StabilizeLevel.lock;
      onProgress?.call(shaky ? '偵測到晃動,套用防手震' : '畫面穩定,略過防手震', 1.0);
      report = VideoReport(shakeScore: jitter);
      if (shaky) {
        final level = options.stabilizeLevel;
        var locked = false;
        List<Motion>? smooth;
        if (level == StabilizeLevel.lock) {
          final l = lockTrajectory(list, workW, workH, maxZoom: level.maxZoom);
          if (l != null) {
            (smooth, zoom, _) = l;
            locked = true;
          }
        }
        // 鎖定需要的裁切太多(例如邊走邊拍)時,改用超強平滑。
        final lv = locked || level != StabilizeLevel.lock ? level : StabilizeLevel.extreme;
        smooth ??= stabilizePath(list, workW, workH,
            budget: lv.budget, rotBudget: lv.rotBudget, sigma: smoothingSigma(info.fps));
        // 果凍效應修正:高通的時間尺度約半秒(sigma = fps/2 格)。
        final rs = rollingShutterOffsets(residuals, info.fps / 2);
        // 最終倍率要把果凍修正的條帶偏移也算進去,才不會露邊。
        zoom = options.zoom > 0
            ? options.zoom
            : stabilizationZoom(smooth, workW, workH, maxZoom: lv.maxZoom, rs: rs);
        corrections = Float64List.fromList([for (final m in smooth) ...[m.dx, m.dy, m.da]]);
        rollingShutter = Float64List.fromList([for (final r in rs) ...r]);
        report = VideoReport(shakeScore: jitter, stabilized: true, zoom: zoom, locked: locked);
      }
    }

    _checkCancelled();
    // ---- 階段 2:解碼 → OpenCV → 編碼 ----
    final (outW, outH) =
        options.superRes ? srTargetSize(workW, workH, options.srLongSide) : (workW, workH);
    final inPipe = await newPipe();
    final outPipe = await newPipe();
    var finished = false;

    final decode = _ffmpeg([
      '-y', '-threads', '2', '-i', src, '-an', '-frames:v', '$frames',
      '-vf', 'scale=$workW:$workH:in_color_matrix=${info.color.swsMatrix}:in_range=tv'
          ':flags=accurate_rnd+full_chroma_int+full_chroma_inp',
      '-f', 'rawvideo', '-pix_fmt', 'bgr24', inPipe, //
    ], onDone: () {
      if (!finished) _unblockPipe(inPipe, readerIsStuck: true);
    });
    final c = info.color;
    // 輸出沿用原片的色彩規格:HDR 原片輸出 10-bit HEVC 並保留 HLG/PQ 標註;SDR 原片輸出 H.264。
    final videoCodec = c.isHdr
        ? [
            '-vf', 'scale=out_color_matrix=${c.swsMatrix}:out_range=tv'
                ':flags=accurate_rnd+full_chroma_int,format=yuv420p10le',
            '-c:v', 'libx265', '-preset', 'fast', '-crf', '14', '-tag:v', 'hvc1',
            '-x265-params',
            'colorprim=${c.primaries}:transfer=${c.transfer}:colormatrix=${c.space}:range=limited'
                // 4K 時 x265 預設會預讀十多格、開多條執行緒,每格都要幾十 MB,手機記憶體撐不住。
                ':frame-threads=1:rc-lookahead=5:lookahead-slices=0:ref=2:bframes=2',
          ]
        : [
            '-vf', 'scale=out_color_matrix=${c.swsMatrix}:out_range=tv'
                ':flags=accurate_rnd+full_chroma_int,format=yuv420p',
            '-c:v', 'libx264', '-crf', '14',
          ];
    final encode = _ffmpeg([
      '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '${outW}x$outH',
      '-framerate', '${info.fps}', '-i', outPipe, '-i', src,
      '-map', '0:v:0', '-map', '1:a:0?',
      ...videoCodec,
      '-color_primaries', c.primaries, '-color_trc', c.transfer,
      '-colorspace', c.space, '-color_range', 'tv',
      '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart',
      dst, //
    ], onDone: () {
      if (!finished) _unblockPipe(outPipe, readerIsStuck: false);
    });

    await _runIsolate<Object?>(
      _renderEntry,
      _RenderJob(
        inPipe: inPipe,
        outPipe: outPipe,
        w: workW,
        h: workH,
        outW: outW,
        outH: outH,
        frames: frames,
        options: options,
        corrections: corrections,
        rollingShutter: rollingShutter,
        zoom: zoom,
      ),
      progress,
    ).whenComplete(() => finished = true);

    _checkCancelled();
    onProgress?.call('收尾', 1.0);
    final enc = await encode;
    await decode;
    _checkCancelled();
    if (!ReturnCode.isSuccess(enc)) {
      throw Exception('編碼失敗:$_lastFfmpegError');
    }
    return report.withSize(outW, outH);
  } catch (_) {
    // 取消後各階段會因管道中斷而各自報錯,統一成「已取消」,並刪掉不完整的輸出檔。
    try {
      File(dst).deleteSync();
    } catch (_) {}
    if (_cancelRequested) throw const VideoCancelled();
    rethrow;
  } finally {
    for (final p in pipes) {
      await FFmpegKitConfig.closeFFmpegPipe(p);
    }
  }
}
