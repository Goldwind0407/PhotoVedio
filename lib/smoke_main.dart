// 模擬器實測:影片放在 App 外部儲存目錄 v1..vN.mp4,輸出 out_N.mp4。
import 'dart:io';

import 'package:flutter/widgets.dart';
import 'package:path_provider/path_provider.dart';

import 'package:ffmpeg_kit_min_gpl/ffmpeg_kit.dart';

import 'engine/pipeline.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final ext = (await getExternalStorageDirectory())!.path;
  final tmp = (await getTemporaryDirectory()).path;

  // 偵錯模式:cmd.txt 存在時,直接執行裡面的 FFmpeg 參數(每行一個),{ext}/{tmp} 會被替換。
  final cmdFile = File('$ext/cmd.txt');
  if (cmdFile.existsSync()) {
    final args = cmdFile
        .readAsLinesSync()
        .where((l) => l.trim().isNotEmpty)
        .map((l) => l.trim().replaceAll('{ext}', ext).replaceAll('{tmp}', tmp))
        .toList();
    final sw = Stopwatch()..start();
    final session = await FFmpegKit.executeWithArguments(args);
    final code = await session.getReturnCode();
    final logs = await session.getAllLogsAsString() ?? '';
    final lines = logs.split('\n').where((l) => l.trim().isNotEmpty).toList();
    for (final l in lines.skip(lines.length > 25 ? lines.length - 25 : 0)) {
      debugPrint('SMOKE CMD $l');
    }
    debugPrint('SMOKE CMD rc=${code?.getValue()} ${sw.elapsedMilliseconds}ms '
        'maxRssMB=${ProcessInfo.maxRss ~/ (1024 * 1024)}');
    debugPrint('SMOKE DONE');
    return;
  }
  const options = EngineOptions(
    enhance: true,
    stabilize: true, // 自動判斷:只有偵測到晃動才修正
  );
  // 每次啟動只處理 only.txt 指定的那一支,避免同一程序連續處理累積記憶體壓力。
  final onlyFile = File('$ext/only.txt');
  final only = onlyFile.existsSync() ? int.parse(onlyFile.readAsStringSync().trim()) : 0;
  for (var n = only > 0 ? only : 1; File('$ext/v$n.mp4').existsSync(); n++) {
    final sw = Stopwatch()..start();
    try {
      final r = await processVideo('$ext/v$n.mp4', '$tmp/out_$n.mp4', options,
          onProgress: (stage, p) {
        if ((p * 100).round() % 5 == 0) {
          debugPrint('SMOKE v$n $stage ${(p * 100).round()}% rss=${ProcessInfo.currentRss ~/ (1024 * 1024)}MB');
        }
      });
      debugPrint('SMOKE v$n OK ${sw.elapsedMilliseconds}ms '
          'out=${File('$tmp/out_$n.mp4').lengthSync()} '
          'maxRssMB=${ProcessInfo.maxRss ~/ (1024 * 1024)} '
          'shake=${r.shakeScore?.toStringAsExponential(2)} zoom=${r.zoom.toStringAsFixed(3)}');
      if (only > 0) break;
    } catch (err, st) {
      debugPrint('SMOKE v$n FAIL $err $st');
    }
  }
  debugPrint('SMOKE DONE');
}
