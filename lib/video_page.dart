import 'dart:io';

import 'package:flutter/material.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';

import 'engine/algorithms.dart' show EnhanceLevel, StabilizeLevel;
import 'engine/pipeline.dart';

/// 影片修復頁:提高畫質、除霧、防手震。處理完成後可預覽並存到相簿。
class VideoPage extends StatefulWidget {
  const VideoPage({super.key});

  @override
  State<VideoPage> createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  String? _sourcePath;
  String? _outputPath;
  VideoPlayerController? _player;
  bool _showOutput = false;

  bool _enhance = true;
  EnhanceLevel _enhanceLevel = EnhanceLevel.medium;
  bool _dehaze = false;
  double _dehazeStrength = 0.6; // 0 ~ 1
  bool _stabilize = false;
  StabilizeLevel _stabilizeLevel = StabilizeLevel.standard;

  bool _busy = false;
  double _progress = 0;
  String _stage = '';

  @override
  void dispose() {
    _player?.dispose();
    super.dispose();
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _load(String path) async {
    final old = _player;
    final c = VideoPlayerController.file(File(path));
    await c.initialize();
    await c.setLooping(true);
    await c.play();
    if (!mounted) return;
    setState(() => _player = c);
    await old?.dispose();
  }

  Future<void> _pick() async {
    final file = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (file == null) return;
    _sourcePath = file.path;
    _outputPath = null;
    _showOutput = false;
    await _load(file.path);
  }

  Future<void> _process() async {
    final src = _sourcePath;
    final player = _player;
    if (src == null || player == null) return;
    if (!_enhance && !_dehaze && !_stabilize) {
      _toast('請至少開啟一項處理');
      return;
    }
    setState(() {
      _busy = true;
      _progress = 0;
      _stage = '準備中';
    });
    try {
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
        ),
        onProgress: (stage, p) {
          if (!mounted) return;
          setState(() {
            _stage = stage;
            _progress = p;
          });
        },
      );
      _outputPath = out;
      _showOutput = true;
      await _load(out);
      _toast('處理完成,可預覽後儲存${_describe(report)}');
    } catch (e) {
      _toast('$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _describe(VideoReport r) {
    if (r.shakeScore == null) return '';
    if (!r.stabilized) return '\n畫面本來就很穩定,未套用防手震';
    final crop = ((1 - 1 / r.zoom) * 100).round();
    return '\n已${r.locked ? '鎖定鏡頭' : '套用防手震'},畫面裁切約 $crop%';
  }

  Future<void> _save() async {
    final path = _outputPath;
    if (path == null) return;
    try {
      if (Platform.isAndroid || Platform.isIOS) {
        await Gal.putVideo(path);
        _toast('已儲存到相簿');
      } else {
        _toast('已儲存:$path');
      }
    } catch (e) {
      _toast('儲存失敗:$e');
    }
  }

  Future<void> _toggleCompare() async {
    final path = _showOutput ? _sourcePath : _outputPath;
    if (path == null) return;
    _showOutput = !_showOutput;
    await _load(path);
  }

  @override
  Widget build(BuildContext context) {
    final p = _player;
    return Scaffold(
      appBar: AppBar(
        title: const Text('影片修復'),
        actions: [
          IconButton(
            tooltip: '選擇影片',
            icon: const Icon(Icons.video_library_outlined),
            onPressed: _busy ? null : _pick,
          ),
          if (_outputPath != null)
            IconButton(
              tooltip: _showOutput ? '看原片' : '看處理後',
              icon: Icon(_showOutput ? Icons.compare : Icons.compare_outlined),
              onPressed: _busy ? null : _toggleCompare,
            ),
          IconButton(
            tooltip: '儲存',
            icon: const Icon(Icons.save_alt),
            onPressed: _outputPath != null && !_busy ? _save : null,
          ),
        ],
      ),
      body: p == null
          ? Center(
              child: FilledButton.icon(
                onPressed: _pick,
                icon: const Icon(Icons.video_file_outlined),
                label: const Text('選擇一段影片'),
              ),
            )
          : Column(
              children: [
                Expanded(
                  child: Center(
                    child: AspectRatio(
                      aspectRatio: p.value.aspectRatio,
                      child: Stack(
                        alignment: Alignment.topLeft,
                        children: [
                          VideoPlayer(p),
                          Padding(
                            padding: const EdgeInsets.all(8),
                            child: Chip(
                              label: Text(_showOutput ? '處理後' : '原片'),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                _controls(),
              ],
            ),
    );
  }

  Widget _controls() {
    return Container(
      color: Theme.of(context).colorScheme.surfaceContainer,
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SwitchListTile(
              dense: true,
              title: const Text('提高畫質'),
              subtitle: const Text('增強紋理與輪廓細節,不改變色彩與亮度'),
              value: _enhance,
              onChanged: _busy ? null : (v) => setState(() => _enhance = v),
            ),
            if (_enhance)
              SegmentedButton<EnhanceLevel>(
                segments: const [
                  ButtonSegment(value: EnhanceLevel.low, label: Text('弱')),
                  ButtonSegment(value: EnhanceLevel.medium, label: Text('中')),
                  ButtonSegment(value: EnhanceLevel.high, label: Text('強')),
                ],
                selected: {_enhanceLevel},
                onSelectionChanged:
                    _busy ? null : (v) => setState(() => _enhanceLevel = v.first),
              ),
            SwitchListTile(
              dense: true,
              title: const Text('除霧'),
              subtitle: const Text('只在有明顯霧氣時使用,否則會改變原片色調'),
              value: _dehaze,
              onChanged: _busy ? null : (v) => setState(() => _dehaze = v),
            ),
            if (_dehaze)
              Slider(
                value: _dehazeStrength,
                onChanged:
                    _busy ? null : (v) => setState(() => _dehazeStrength = v),
              ),
            SwitchListTile(
              dense: true,
              title: const Text('防手震'),
              subtitle: Text(switch (_stabilizeLevel) {
                StabilizeLevel.standard => '含果凍效應修正,畫面裁切約 5~10%',
                StabilizeLevel.strong => '適合走路拍攝,裁切約 10~16%',
                StabilizeLevel.lock => '像架在腳架上,完全不晃;裁切最多',
              }),
              value: _stabilize,
              onChanged: _busy ? null : (v) => setState(() => _stabilize = v),
            ),
            if (_stabilize)
              SegmentedButton<StabilizeLevel>(
                segments: const [
                  ButtonSegment(value: StabilizeLevel.standard, label: Text('標準')),
                  ButtonSegment(value: StabilizeLevel.strong, label: Text('強')),
                  ButtonSegment(value: StabilizeLevel.lock, label: Text('鎖定')),
                ],
                selected: {_stabilizeLevel},
                onSelectionChanged:
                    _busy ? null : (v) => setState(() => _stabilizeLevel = v.first),
              ),
            const SizedBox(height: 4),
            if (_busy) ...[
              Text(_stage),
              LinearProgressIndicator(value: _progress),
            ] else
              FilledButton.icon(
                onPressed: _sourcePath == null ? null : _process,
                icon: const Icon(Icons.auto_fix_high),
                label: const Text('開始處理'),
              ),
          ],
        ),
      ),
    );
  }
}
