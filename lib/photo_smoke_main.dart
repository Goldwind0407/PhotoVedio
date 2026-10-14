// 暫時的驗證入口(不提交):處理 <ext>/photo_in 的照片,輸出到 <ext>/photo_out。
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/widgets.dart';
import 'package:path_provider/path_provider.dart';

import 'engine/photo.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final ext = (await getExternalStorageDirectory())!.path;
  Directory('$ext/photo_out').createSync(recursive: true);
  final files = Directory(ext).listSync().whereType<File>().where((f) => f.path.endsWith('.jpg') && (f.path.contains('big') || f.path.contains('r301'))).toList()
    ..sort((a, b) => a.path.compareTo(b.path));
  for (final f in files) {
    final name = f.uri.pathSegments.last.split('.').first;
    for (final (k, o) in [
      ('dehaze', const PhotoOptions(dehaze: true)),
      ('enhance', const PhotoOptions(enhance: true)),
      ('both', const PhotoOptions(dehaze: true, enhance: true)),
      ('preview', const PhotoOptions(dehaze: true, enhance: true, maxLongSide: 1600)),
    ]) {
      final sw = Stopwatch()..start();
      final bytes = f.readAsBytesSync();
      try {
        final out = await Isolate.run(() => processPhoto(bytes, o));
        File('$ext/photo_out/${name}_$k.png').writeAsBytesSync(out);
        debugPrint('SMOKE $name $k ${sw.elapsedMilliseconds}ms '
            'maxRss=${ProcessInfo.maxRss >> 20}MB');
      } catch (e) {
        debugPrint('SMOKE $name $k FAIL $e');
      }
    }
  }
  debugPrint('SMOKE DONE');
}
