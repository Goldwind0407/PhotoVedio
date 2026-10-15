import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';

import 'engine/algorithms.dart' show EnhanceLevel, StabilizeLevel;
import 'engine/pipeline.dart';
import 'engine/superres.dart' show srModelAsset;

/// 影片修復頁:防手震、提高畫質、除霧。處理完成後可比較原片、再存到相簿。
class VideoPage extends StatefulWidget {
  const VideoPage({super.key, this.initialPath});

  /// 從首頁選好的影片;null 時顯示「選擇影片」。
  final String? initialPath;

  @override
  State<VideoPage> createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  String? _sourcePath;
  String? _outputPath; // 處理結果(暫存檔,離開頁面時刪除)
  VideoPlayerController? _player;
  bool _showOutput = false;
  bool _saved = false;
  bool _saving = false;
  String? _resultNote; // 處理完的說明(例如防手震裁切了多少)
  String? _loadError; // 影片無法播放時的原因

  bool _stabilize = true;
  StabilizeLevel _stabilizeLevel = StabilizeLevel.standard;
  bool _enhance = true;
  EnhanceLevel _enhanceLevel = EnhanceLevel.medium;
  bool _dehaze = false;
  double _dehazeStrength = 0.6; // 0 ~ 1
  bool _superRes = false; // AI 放大到 4K

  bool _busy = false;
  bool _cancelling = false;
  double _progress = 0;
  String _stage = '';
  // 剩餘時間估計:以目前階段的開始時間與進度推算
  String _etaStage = '';
  DateTime _etaStart = DateTime.now();

  // 閃退紀錄:處理中定期寫下目前進度與記憶體用量,正常結束就刪除。
  // 若 App 在處理中被系統關掉(多半是記憶體不足),下次打開這頁會顯示停在哪裡。
  File? _crashLog;
  String _lastLogged = '';

  @override
  void initState() {
    super.initState();
    _checkLastCrash();
    final path = widget.initialPath;
    if (path != null) _open(path);
  }

  @override
  void dispose() {
    _player?.dispose();
    _deleteOutput();
    super.dispose();
  }

  Future<void> _checkLastCrash() async {
    final String info;
    try {
      final dir = await getApplicationSupportDirectory();
      final f = _crashLog = File('${dir.path}${Platform.pathSeparator}video_progress.txt');
      if (!f.existsSync()) return;
      info = f.readAsStringSync().trim();
      f.deleteSync();
    } catch (_) {
      return; // 讀不到紀錄不影響使用
    }
    if (!mounted || info.isEmpty) return;
    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('上次處理意外中斷'),
        content: Text('App 在處理影片時被關閉,通常是手機記憶體不足。\n\n$info'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('知道了')),
        ],
      ),
    );
  }

  /// [key] 不變時不重寫(進度每 2% 才寫一次)。
  void _logProgress(String key, String text) {
    if (key == _lastLogged) return;
    _lastLogged = key;
    try {
      _crashLog?.writeAsStringSync(text, flush: true);
    } catch (_) {}
  }

  void _clearLog() {
    try {
      _crashLog?.deleteSync();
    } catch (_) {}
  }

  void _deleteOutput() {
    final p = _outputPath;
    if (p == null) return;
    try {
      File(p).deleteSync();
    } catch (_) {}
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _load(String path, {Duration? at, bool play = true}) async {
    final old = _player;
    final c = VideoPlayerController.file(File(path));
    await c.initialize();
    await c.setLooping(true);
    if (at != null) await c.seekTo(at);
    if (play) await c.play();
    if (!mounted) {
      await c.dispose();
      return;
    }
    setState(() => _player = c);
    await old?.dispose();
  }

  Future<void> _open(String path) async {
    _deleteOutput();
    setState(() {
      _sourcePath = path;
      _outputPath = null;
      _showOutput = false;
      _saved = false;
      _resultNote = null;
      _loadError = null;
    });
    try {
      await _load(path);
    } catch (e) {
      if (mounted) setState(() => _loadError = '$e');
    }
  }

  Future<void> _pick() async {
    if (_outputPath != null && !_saved && !await _confirmDiscard()) return;
    final file = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (file != null) await _open(file.path);
  }

  Future<void> _process() async {
    final src = _sourcePath;
    final player = _player;
    if (src == null || player == null) return;
    if (!_enhance && !_dehaze && !_stabilize && !_superRes) {
      _toast('請至少開啟一項處理');
      return;
    }
    setState(() {
      _busy = true;
      _cancelling = false;
      _progress = 0;
      _stage = '準備中';
      _etaStage = '';
    });
    // 處理時暫停預覽:播放器自己也要解碼影片,4K 時會和處理搶記憶體。
    await player.pause();
    final size = player.value.size;
    final settings = [
      '${size.width.round()}×${size.height.round()}',
      if (_enhance) '提高畫質(${_enhanceLevel.name})',
      if (_dehaze) '除霧',
      if (_stabilize) '防手震(${_stabilizeLevel.name})',
      if (_superRes) 'AI 放大',
    ].join('、');
    _lastLogged = '';
    _logProgress('準備中', '影片:$settings\n停在:準備中');
    try {
      final model = _superRes
          ? (await rootBundle.load(srModelAsset)).buffer.asUint8List()
          : null;
      final dir = await getTemporaryDirectory();
      final stamp = DateTime.now().millisecondsSinceEpoch;
      final out = '${dir.path}/fixed_$stamp.mp4';
      final report = await processVideo(
        src,
        out,
        EngineOptions(
          enhance: _enhance,
          enhanceLevel: _enhanceLevel,
          dehaze: _dehaze,
          dehazeStrength: _dehazeStrength,
          stabilize: _stabilize,
          stabilizeLevel: _stabilizeLevel,
          superRes: _superRes,
          srModel: model,
        ),
        onProgress: (stage, p) {
          final at = '$stage ${(p * 50).floor() * 2}%';
          _logProgress(at, '影片:$settings\n停在:$at\n記憶體用量:約 ${ProcessInfo.currentRss >> 20} MB');
          if (!mounted) return;
          setState(() {
            // 括號內是即時資訊(例如 AI 每格秒數),不算換階段
            final key = stage.split('(').first;
            if (key != _etaStage) {
              _etaStage = key;
              _etaStart = DateTime.now();
            }
            _stage = stage;
            _progress = p;
          });
        },
      );
      if (!mounted) {
        File(out).deleteSync();
        return;
      }
      _deleteOutput(); // 上一次的處理結果
      setState(() {
        _outputPath = out;
        _showOutput = true;
        _saved = false;
        _resultNote = _describe(report);
      });
      await _load(out);
      _toast('處理完成,可以比較原片後再儲存');
    } on VideoCancelled {
      _toast('已取消');
    } catch (e) {
      _toast('處理失敗:$e');
    } finally {
      _clearLog();
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _cancel() async {
    setState(() => _cancelling = true);
    await cancelVideoProcessing();
  }

  String? _describe(VideoReport r) {
    final notes = <String>[];
    if (r.shakeScore != null) {
      if (!r.stabilized) {
        notes.add('畫面本來就很穩定,沒有套用防手震');
      } else {
        final crop = ((1 - 1 / r.zoom) * 100).round();
        notes.add('${r.locked ? '已鎖定鏡頭' : '已套用防手震'},畫面裁切約 $crop%');
      }
    }
    if (_superRes && r.outWidth > 0) notes.add('AI 放大為 ${r.outWidth}×${r.outHeight}');
    return notes.isEmpty ? null : notes.join(';');
  }

  Future<void> _save() async {
    final path = _outputPath;
    if (path == null) return;
    setState(() => _saving = true);
    try {
      if (Platform.isAndroid || Platform.isIOS) {
        await Gal.putVideo(path);
        _toast('已儲存到相簿');
      } else {
        final dir = await getApplicationDocumentsDirectory();
        final dst = '${dir.path}${Platform.pathSeparator}${path.split(Platform.pathSeparator).last}';
        await File(path).copy(dst);
        _toast('已儲存:$dst');
      }
      _saved = true;
    } catch (e) {
      _toast('儲存失敗:$e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  /// 切換原片/處理後,停在同一個時間點,方便比較。
  Future<void> _showVersion(bool output) async {
    final path = output ? _outputPath : _sourcePath;
    if (path == null || output == _showOutput) return;
    final at = _player?.value.position;
    final playing = _player?.value.isPlaying ?? true;
    setState(() => _showOutput = output);
    await _load(path, at: at, play: playing);
  }

  Future<bool> _confirmDiscard() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('處理結果還沒儲存'),
        content: const Text('離開後處理好的影片就會刪除。'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('留下')),
          TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('不儲存')),
        ],
      ),
    );
    return ok == true;
  }

  Future<void> _onBack() async {
    if (_busy) {
      final stop = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('要停止處理嗎?'),
          content: const Text('影片還在處理中,離開會取消這次處理。'),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('繼續處理')),
            TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('取消並離開')),
          ],
        ),
      );
      if (stop != true || !mounted) return;
      await _cancel();
      if (mounted) Navigator.of(context).pop();
      return;
    }
    if (await _confirmDiscard() && mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final hasResult = _outputPath != null;
    return PopScope(
      canPop: !_busy && (!hasResult || _saved),
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) _onBack();
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('修影片'),
          actions: [
            if (_sourcePath != null)
              TextButton(onPressed: _busy ? null : _pick, child: const Text('換影片')),
            if (hasResult)
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: FilledButton.icon(
                  onPressed: _busy || _saving ? null : _save,
                  icon: Icon(_saved ? Icons.check : Icons.download, size: 18),
                  label: Text(_saving ? '儲存中' : (_saved ? '已儲存' : '儲存')),
                ),
              ),
          ],
        ),
        body: _sourcePath == null ? _empty() : _editor(),
      ),
    );
  }

  Widget _empty() {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.movie_outlined, size: 64, color: scheme.onSurfaceVariant),
            const SizedBox(height: 16),
            Text('選一段影片開始', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text('防手震、果凍修正、提高畫質、除霧',
                style: TextStyle(color: scheme.onSurfaceVariant)),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _pick,
              icon: const Icon(Icons.video_library_outlined),
              label: const Text('選擇影片'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _editor() {
    return Column(
      children: [
        if (_outputPath != null && !_busy)
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: SegmentedButton<bool>(
              segments: const [
                ButtonSegment(value: false, label: Text('原片')),
                ButtonSegment(value: true, label: Text('處理後')),
              ],
              selected: {_showOutput},
              onSelectionChanged: (v) => _showVersion(v.first),
            ),
          ),
        Expanded(child: _videoArea()),
        _bottomPanel(),
      ],
    );
  }

  Widget _videoArea() {
    final p = _player;
    final error = _loadError;
    if (error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 48),
              const SizedBox(height: 12),
              const Text('無法播放這段影片'),
              const SizedBox(height: 4),
              Text(error,
                  textAlign: TextAlign.center,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant)),
              const SizedBox(height: 16),
              FilledButton.tonal(onPressed: _pick, child: const Text('換一段影片')),
            ],
          ),
        ),
      );
    }
    if (p == null) return const Center(child: CircularProgressIndicator());
    return Column(
      children: [
        Expanded(
          child: Center(
            child: AspectRatio(
              aspectRatio: p.value.aspectRatio,
              child: GestureDetector(
                // 播放狀態由下面的 ValueListenableBuilder 反映,不必 setState
                onTap: _busy ? null : () => p.value.isPlaying ? p.pause() : p.play(),
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    VideoPlayer(p),
                    ValueListenableBuilder(
                      valueListenable: p,
                      builder: (context, v, _) => AnimatedOpacity(
                        opacity: v.isPlaying ? 0 : 1,
                        duration: const Duration(milliseconds: 150),
                        child: const CircleAvatar(
                          radius: 28,
                          backgroundColor: Colors.black45,
                          child: Icon(Icons.play_arrow, size: 36, color: Colors.white),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: VideoProgressIndicator(p,
              allowScrubbing: !_busy, padding: const EdgeInsets.symmetric(vertical: 10)),
        ),
      ],
    );
  }

  Widget _bottomPanel() {
    return Container(
      width: double.infinity,
      color: Theme.of(context).colorScheme.surfaceContainer,
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
          child: _busy ? _progressView() : _optionsView(),
        ),
      ),
    );
  }

  Widget _progressView() {
    final scheme = Theme.of(context).colorScheme;
    final steps = _stabilize ? 2 : 1;
    final step = _stage == '分析晃動' || !_stabilize ? 1 : 2;
    final elapsed = DateTime.now().difference(_etaStart);
    String? eta;
    if (_progress > 0.03 && _progress < 1 && elapsed.inSeconds >= 3) {
      final left = elapsed * ((1 - _progress) / _progress);
      eta = left.inMinutes > 0
          ? '約剩 ${left.inMinutes} 分 ${left.inSeconds % 60} 秒'
          : '約剩 ${left.inSeconds} 秒';
    }
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(_cancelling ? '正在取消…' : '步驟 $step/$steps:$_stage',
                  style: Theme.of(context).textTheme.titleSmall),
            ),
            Text('${(_progress * 100).round()}%',
                style: Theme.of(context).textTheme.titleSmall),
          ],
        ),
        const SizedBox(height: 8),
        LinearProgressIndicator(value: _progress, minHeight: 6, borderRadius: BorderRadius.circular(3)),
        const SizedBox(height: 8),
        Text(
          [?eta, '處理中請留在這個畫面'].join(' · '),
          style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
        ),
        const SizedBox(height: 8),
        OutlinedButton(onPressed: _cancelling ? null : _cancel, child: const Text('取消處理')),
      ],
    );
  }

  Widget _optionsView() {
    final note = _resultNote;
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (note != null && _showOutput)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                const Icon(Icons.info_outline, size: 16),
                const SizedBox(width: 6),
                Expanded(child: Text(note, style: const TextStyle(fontSize: 12))),
              ],
            ),
          ),
        ConstrainedBox(
          constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.38),
          child: SingleChildScrollView(
            child: Column(
              children: [
                _OptionTile(
                  icon: Icons.videocam_outlined,
                  title: '防手震',
                  subtitle: switch (_stabilizeLevel) {
                    StabilizeLevel.standard => '含果凍修正與補邊,畫面裁切約 3~5%',
                    StabilizeLevel.strong => '適合走路拍攝,裁切約 5~8%',
                    StabilizeLevel.extreme => '晃很大或跟拍時用,裁切約 8~12%',
                    StabilizeLevel.lock => '像架在腳架上完全不晃,裁切最多',
                  },
                  value: _stabilize,
                  onChanged: (v) => setState(() => _stabilize = v),
                  child: _Levels<StabilizeLevel>(
                    value: _stabilizeLevel,
                    options: const {
                      StabilizeLevel.standard: '標準',
                      StabilizeLevel.strong: '強',
                      StabilizeLevel.extreme: '超強',
                      StabilizeLevel.lock: '鎖定',
                    },
                    onChanged: (v) => setState(() => _stabilizeLevel = v),
                  ),
                ),
                _OptionTile(
                  icon: Icons.hd_outlined,
                  title: 'AI 放大到 4K',
                  subtitle: '補出細節、讓畫面更清楚;非常耗時',
                  value: _superRes,
                  onChanged: (v) => setState(() => _superRes = v),
                  child: Text(
                    '每一格都要經過 AI 運算,手機上每秒影片可能要好幾分鐘(依手機效能而定)。'
                    '處理時會顯示預估剩餘時間,隨時可以取消。開啟時會取代「提高畫質」。',
                    style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant),
                  ),
                ),
                if (!_superRes)
                  _OptionTile(
                  icon: Icons.auto_awesome,
                  title: '提高畫質',
                  subtitle: '去除壓縮雜訊,讓輪廓更清楚',
                  value: _enhance,
                  onChanged: (v) => setState(() => _enhance = v),
                  child: _Levels<EnhanceLevel>(
                    value: _enhanceLevel,
                    options: const {
                      EnhanceLevel.low: '弱',
                      EnhanceLevel.medium: '中',
                      EnhanceLevel.high: '強',
                    },
                    onChanged: (v) => setState(() => _enhanceLevel = v),
                  ),
                ),
                _OptionTile(
                  icon: Icons.blur_off,
                  title: '除霧',
                  subtitle: '只在畫面有明顯霧氣時使用',
                  value: _dehaze,
                  onChanged: (v) => setState(() => _dehaze = v),
                  child: Row(
                    children: [
                      const Text('強度'),
                      Expanded(
                        child: Slider(
                          value: _dehazeStrength,
                          min: 0.1,
                          onChanged: (v) => setState(() => _dehazeStrength = v),
                        ),
                      ),
                      SizedBox(width: 32, child: Text('${(_dehazeStrength * 100).round()}')),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        FilledButton.icon(
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
          onPressed: _player == null ? null : _process,
          icon: const Icon(Icons.auto_fix_high),
          label: Text(_outputPath == null ? '開始處理' : '用目前設定重新處理'),
        ),
      ],
    );
  }
}

/// 一個處理項目:標題列帶開關,開啟時在下方顯示細部設定。
class _OptionTile extends StatelessWidget {
  const _OptionTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.value,
    required this.onChanged,
    required this.child,
  });

  final IconData icon;
  final String title, subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Material(
        color: value ? scheme.surfaceContainerHighest : scheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(14),
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: () => onChanged(!value),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 8, 8),
            child: Column(
              children: [
                Row(
                  children: [
                    Icon(icon, color: value ? scheme.primary : scheme.onSurfaceVariant),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
                          Text(subtitle,
                              style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
                        ],
                      ),
                    ),
                    Switch(value: value, onChanged: onChanged),
                  ],
                ),
                AnimatedSize(
                  duration: const Duration(milliseconds: 150),
                  alignment: Alignment.topCenter,
                  child: value
                      ? Padding(padding: const EdgeInsets.only(top: 6, right: 4), child: child)
                      : const SizedBox(width: double.infinity),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// 強度選擇(撐滿寬度的分段按鈕)。
class _Levels<T> extends StatelessWidget {
  const _Levels({required this.value, required this.options, required this.onChanged});

  final T value;
  final Map<T, String> options;
  final ValueChanged<T> onChanged;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: SegmentedButton<T>(
        showSelectedIcon: false,
        segments: [
          for (final e in options.entries) ButtonSegment(value: e.key, label: Text(e.value)),
        ],
        selected: {value},
        onSelectionChanged: (v) => onChanged(v.first),
      ),
    );
  }
}
