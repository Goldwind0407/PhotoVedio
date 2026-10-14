import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:crop_your_image/crop_your_image.dart';
import 'package:flutter/material.dart';
import 'package:gal/gal.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'engine/photo.dart';

/// 4x5 色彩矩陣(與 Flutter ColorFilter.matrix 同格式,偏移量為 0~255)。
typedef ColorMatrix = List<double>;

const ColorMatrix identityMatrix = [
  1, 0, 0, 0, 0, //
  0, 1, 0, 0, 0,
  0, 0, 1, 0, 0,
  0, 0, 0, 1, 0,
];

/// 回傳 a 之後再套 b 的合併矩陣(先 a 後 b)。
ColorMatrix multiply(ColorMatrix a, ColorMatrix b) {
  final out = List<double>.filled(20, 0);
  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 5; c++) {
      var v = 0.0;
      for (var k = 0; k < 4; k++) {
        v += b[r * 5 + k] * a[k * 5 + c];
      }
      if (c == 4) v += b[r * 5 + 4];
      out[r * 5 + c] = v;
    }
  }
  return out;
}

ColorMatrix saturationMatrix(double s) {
  const lr = 0.2126, lg = 0.7152, lb = 0.0722;
  return [
    lr * (1 - s) + s, lg * (1 - s), lb * (1 - s), 0, 0, //
    lr * (1 - s), lg * (1 - s) + s, lb * (1 - s), 0, 0,
    lr * (1 - s), lg * (1 - s), lb * (1 - s) + s, 0, 0,
    0, 0, 0, 1, 0,
  ];
}

ColorMatrix contrastBrightnessMatrix(double contrast, double brightness) {
  final t = 128 * (1 - contrast) + brightness * 255;
  return [
    contrast, 0, 0, 0, t, //
    0, contrast, 0, 0, t,
    0, 0, contrast, 0, t,
    0, 0, 0, 1, 0,
  ];
}

class FilterPreset {
  final String name;
  final ColorMatrix matrix;
  const FilterPreset(this.name, this.matrix);
}

final List<FilterPreset> presets = [
  const FilterPreset('原圖', identityMatrix),
  FilterPreset('鮮豔', saturationMatrix(1.5)),
  FilterPreset('黑白', saturationMatrix(0)),
  const FilterPreset('復古', [
    0.393, 0.769, 0.189, 0, 0, //
    0.349, 0.686, 0.168, 0, 0,
    0.272, 0.534, 0.131, 0, 0,
    0, 0, 0, 1, 0,
  ]),
  const FilterPreset('冷色', [
    0.9, 0, 0, 0, 0, //
    0, 1, 0, 0, 0,
    0, 0, 1.15, 0, 10,
    0, 0, 0, 1, 0,
  ]),
  const FilterPreset('暖色', [
    1.15, 0, 0, 0, 10, //
    0, 1, 0, 0, 0,
    0, 0, 0.9, 0, 0,
    0, 0, 0, 1, 0,
  ]),
];

/// 底部工具列的分頁。
enum _Tool { filter, adjust, repair }

/// 「調整」分頁裡的參數:名稱、範圍、中性值,以及顯示用的 -100~100 換算。
enum _Param {
  brightness('亮度', Icons.wb_sunny_outlined, -0.5, 0.5, 0),
  contrast('對比', Icons.contrast, 0.5, 1.5, 1),
  saturation('飽和', Icons.palette_outlined, 0, 2, 1),
  sharpen('銳化', Icons.details, 0, 1, 0);

  const _Param(this.label, this.icon, this.min, this.max, this.neutral);
  final String label;
  final IconData icon;
  final double min, max, neutral;

  /// 換算成 -100~100(銳化 0~100),比小數好懂。
  int display(double v) => v >= neutral
      ? ((v - neutral) / (max - neutral) * 100).round()
      : -((neutral - v) / (neutral - min) * 100).round();
}

/// 照片編輯所有可調的設定;用來比較是否有修改、以及「重設」後復原。
class _Edits {
  const _Edits({
    this.filter = 0,
    this.values = const {
      _Param.brightness: 0,
      _Param.contrast: 1,
      _Param.saturation: 1,
      _Param.sharpen: 0,
    },
    this.enhance = false,
    this.dehaze = false,
    this.dehazeStrength = 0.6,
  });

  final int filter;
  final Map<_Param, double> values;
  final bool enhance, dehaze;
  final double dehazeStrength;

  bool get isDefault =>
      filter == 0 &&
      !enhance &&
      !dehaze &&
      _Param.values.every((p) => values[p] == p.neutral);
}

class PhotoEditorPage extends StatefulWidget {
  const PhotoEditorPage({super.key, required this.bytes});

  final Uint8List bytes;

  @override
  State<PhotoEditorPage> createState() => _PhotoEditorPageState();
}

class _PhotoEditorPageState extends State<PhotoEditorPage> {
  late final Uint8List _original = widget.bytes; // 選進來的原圖(重設時回到這張)
  late Uint8List _bytes = widget.bytes; // 目前的底圖(裁切後會更新)
  int _filter = 0;
  final Map<_Param, double> _values = Map.of(const _Edits().values);
  bool _enhance = false;
  bool _dehaze = false;
  double _dehazeStrength = 0.6; // 0 ~ 1

  _Tool _tool = _Tool.filter;
  _Param _param = _Param.brightness;

  Uint8List? _preview; // 除霧/提高畫質的預覽結果
  bool _processing = false;
  bool _previewRunning = false; // 同一時間只跑一個預覽,避免連點時多個 isolate 同時吃記憶體
  int _generation = 0; // 設定每變一次就加一,用來丟棄過時的預覽結果
  // 原尺寸處理結果的快取:預覽最後顯示的就是它,儲存時直接沿用,不必重算。
  Future<PhotoResult>? _full;
  Uint8List? _fullSrc;
  PhotoOptions? _fullOpts;
  bool _showOriginal = false; // 按住比較鈕或長按畫面時顯示原圖
  bool _saving = false;
  Object? _savedState; // 最後一次儲存時的設定;一樣就不必在離開時確認

  /// 除霧的快速預覽在縮小的圖上算。
  static const _previewLongSide = 1600;

  _Edits get _edits => _Edits(
        filter: _filter,
        values: Map.of(_values),
        enhance: _enhance,
        dehaze: _dehaze,
        dehazeStrength: _dehazeStrength,
      );

  bool get _cropped => !identical(_bytes, _original);
  bool get _edited => _cropped || !_edits.isDefault;
  Object get _stateKey => (_bytes, _filter, _values.values.join(','), _enhance, _dehaze, _dehazeStrength);
  bool get _unsaved => _edited && _savedState != _stateKey;

  PhotoOptions _photoOptions() => PhotoOptions(
        enhance: _enhance,
        dehaze: _dehaze,
        dehazeStrength: _dehazeStrength,
      );

  /// 原尺寸處理;同一張圖、同一組設定只算一次(預覽與儲存共用)。
  Future<PhotoResult> _fullResult(Uint8List bytes, PhotoOptions opts) {
    final cached = _full;
    if (cached != null && identical(_fullSrc, bytes) && _fullOpts == opts) {
      return cached;
    }
    final f = _processInBackground(bytes, opts);
    _full = f;
    _fullSrc = bytes;
    _fullOpts = opts;
    f.then((_) {}, onError: (_) {
      if (identical(_full, f)) _full = null; // 失敗的結果不快取
    });
    return f;
  }

  ColorMatrix get _matrix {
    var m = presets[_filter].matrix;
    m = multiply(m, saturationMatrix(_values[_Param.saturation]!));
    m = multiply(m, contrastBrightnessMatrix(_values[_Param.contrast]!, _values[_Param.brightness]!));
    return m;
  }

  void _toast(String msg, {SnackBarAction? action}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(msg), action: action));
  }

  void _apply(_Edits e, Uint8List bytes) {
    setState(() {
      _bytes = bytes;
      _filter = e.filter;
      _values.addAll(e.values);
      _enhance = e.enhance;
      _dehaze = e.dehaze;
      _dehazeStrength = e.dehazeStrength;
    });
    _updatePreview();
  }

  /// 全部重設(含裁切),可以從提示列「復原」。
  void _reset() {
    final before = _edits, beforeBytes = _bytes;
    _apply(const _Edits(), _original);
    _toast('已重設所有編輯',
        action: SnackBarAction(label: '復原', onPressed: () => _apply(before, beforeBytes)));
  }

  Future<void> _updatePreview() async {
    _generation++;
    // 已有預覽在跑:它這一輪結束後發現設定變了,會用最新設定再跑一次。
    if (_previewRunning) return;
    _previewRunning = true;
    try {
      while (mounted) {
        final gen = _generation;
        final bytes = _bytes;
        final opts = _photoOptions();
        if (!opts.any) {
          setState(() => _preview = null);
        } else {
          setState(() => _processing = true);
          try {
            // 除霧與影像尺寸無關,先在縮小圖上快速預覽。提高畫質是以像素為單位增強細節,
            // 縮小圖上看起來會比原尺寸強很多,所以一律等原尺寸結果出來再顯示,
            // 預覽才會等於存出來的照片。
            if (opts.dehaze) {
              final quick = await _processInBackground(
                  bytes, opts.copyWith(enhance: false, maxLongSide: _previewLongSide));
              if (!mounted) return;
              if (gen == _generation) setState(() => _preview = quick.bytes);
            }
            if (gen == _generation) {
              final full = await _fullResult(bytes, opts);
              if (!mounted) return;
              if (gen == _generation) setState(() => _preview = full.bytes);
            }
          } catch (e) {
            if (!mounted) return;
            if (gen == _generation) _toast('處理失敗:$e');
          }
        }
        if (gen == _generation) break;
      }
    } finally {
      _previewRunning = false;
      if (mounted) setState(() => _processing = false);
    }
  }

  Future<void> _crop() async {
    final result = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(builder: (_) => CropPage(bytes: _bytes)),
    );
    if (result != null) {
      setState(() => _bytes = result);
      _updatePreview();
    }
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      final bytes = _bytes;
      final matrix = _matrix;
      final sharpen = _values[_Param.sharpen]!;
      final opts = _photoOptions();
      final key = _stateKey;
      // 除霧/提高畫質沿用預覽算好的原尺寸結果(還在算就等它),其餘在背景 isolate 套用。
      final processed = opts.any ? await _fullResult(bytes, opts) : null;
      final out = await _renderInBackground(processed?.bytes ?? bytes, matrix, sharpen);
      final note = processed != null && processed.downscaled
          ? '\n原圖超過 $photoMaxLongSide 像素,已縮小為 ${processed.width}×${processed.height}'
          : '';
      if (Platform.isAndroid || Platform.isIOS) {
        await Gal.putImageBytes(out, name: 'edit_${_stamp()}');
        _toast('已儲存到相簿$note');
      } else {
        final dir = await getApplicationDocumentsDirectory();
        final path = '${dir.path}${Platform.pathSeparator}edit_${_stamp()}.jpg';
        await File(path).writeAsBytes(out);
        _toast('已儲存:$path$note');
      }
      _savedState = key;
    } catch (e) {
      _toast('儲存失敗:$e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  String _stamp() => DateTime.now().millisecondsSinceEpoch.toString();

  /// 有未儲存的編輯時,離開前先確認。
  Future<void> _confirmLeave() async {
    final leave = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('放棄編輯?'),
        content: const Text('這張照片的編輯還沒儲存,離開後就會消失。'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('繼續編輯')),
          TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('放棄')),
        ],
      ),
    );
    if (leave == true && mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: !_unsaved && !_saving,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop && !_saving) _confirmLeave();
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('修照片'),
          actions: [
            TextButton(onPressed: _edited && !_saving ? _reset : null, child: const Text('重設')),
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: FilledButton.icon(
                onPressed: _saving ? null : _save,
                icon: _saving
                    ? const SizedBox.square(
                        dimension: 16, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.download, size: 18),
                label: Text(_saving ? '儲存中' : '儲存'),
              ),
            ),
          ],
        ),
        body: Column(
          children: [
            Expanded(child: _imageArea()),
            _toolPanel(),
          ],
        ),
        bottomNavigationBar: NavigationBar(
          height: 64,
          selectedIndex: _tool.index,
          labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
          onDestinationSelected: (i) {
            if (i == 3) {
              _crop(); // 裁切是開新頁面的動作,不切換分頁
            } else {
              setState(() => _tool = _Tool.values[i]);
            }
          },
          destinations: const [
            NavigationDestination(icon: Icon(Icons.filter_vintage_outlined), label: '濾鏡'),
            NavigationDestination(icon: Icon(Icons.tune), label: '調整'),
            NavigationDestination(icon: Icon(Icons.auto_fix_high_outlined), label: '修復'),
            NavigationDestination(icon: Icon(Icons.crop), label: '裁切'),
          ],
        ),
      ),
    );
  }

  Widget _imageArea() {
    final scheme = Theme.of(context).colorScheme;
    final showOriginal = _showOriginal;
    return Stack(
      children: [
        Positioned.fill(
          child: GestureDetector(
            onLongPressStart: (_) => setState(() => _showOriginal = true),
            onLongPressEnd: (_) => setState(() => _showOriginal = false),
            // 雙指縮放檢查細節(提高畫質的效果要放大才看得清楚)
            child: InteractiveViewer(
              maxScale: 8,
              child: Center(
                child: ColorFiltered(
                  colorFilter: ColorFilter.matrix(showOriginal ? identityMatrix : _matrix),
                  child: Image.memory(
                    showOriginal ? _bytes : (_preview ?? _bytes),
                    fit: BoxFit.contain,
                    gaplessPlayback: true,
                    // 原尺寸結果可能上千萬像素,只解碼到螢幕寬度,省記憶體。
                    cacheWidth: (MediaQuery.sizeOf(context).width *
                            MediaQuery.devicePixelRatioOf(context))
                        .round(),
                  ),
                ),
              ),
            ),
          ),
        ),
        if (showOriginal)
          const Positioned(top: 12, left: 0, right: 0, child: Center(child: Chip(label: Text('原圖')))),
        if (_processing)
          Positioned(
            top: 12,
            left: 0,
            right: 0,
            child: Center(
              child: Chip(
                avatar: const SizedBox.square(
                    dimension: 14, child: CircularProgressIndicator(strokeWidth: 2)),
                label: Text(_enhance ? '原尺寸計算中,需要幾秒鐘' : '計算中'),
              ),
            ),
          ),
        // 按住比較:放開就回到編輯後的樣子
        if (_edited)
          Positioned(
            right: 12,
            bottom: 12,
            child: Tooltip(
              message: '按住看原圖',
              child: GestureDetector(
                onTapDown: (_) => setState(() => _showOriginal = true),
                onTapUp: (_) => setState(() => _showOriginal = false),
                onTapCancel: () => setState(() => _showOriginal = false),
                child: CircleAvatar(
                  radius: 22,
                  backgroundColor: scheme.surfaceContainerHighest.withValues(alpha: 0.85),
                  child: Icon(Icons.compare, color: scheme.onSurface),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _toolPanel() {
    return Container(
      width: double.infinity,
      color: Theme.of(context).colorScheme.surfaceContainer,
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
      child: AnimatedSize(
        duration: const Duration(milliseconds: 150),
        alignment: Alignment.topCenter,
        child: switch (_tool) {
          _Tool.filter => _filterPanel(),
          _Tool.adjust => _adjustPanel(),
          _Tool.repair => _repairPanel(),
        },
      ),
    );
  }

  Widget _filterPanel() {
    final scheme = Theme.of(context).colorScheme;
    return SizedBox(
      height: 96,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: presets.length,
        separatorBuilder: (_, _) => const SizedBox(width: 10),
        itemBuilder: (context, i) {
          final selected = _filter == i;
          return GestureDetector(
            onTap: () => setState(() => _filter = i),
            child: Column(
              children: [
                AnimatedContainer(
                  duration: const Duration(milliseconds: 120),
                  width: 68,
                  height: 68,
                  padding: const EdgeInsets.all(2),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                        color: selected ? scheme.primary : Colors.transparent, width: 2),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(7),
                    child: ColorFiltered(
                      colorFilter: ColorFilter.matrix(presets[i].matrix),
                      child: Image.memory(_bytes, fit: BoxFit.cover, cacheWidth: 160),
                    ),
                  ),
                ),
                const SizedBox(height: 4),
                Text(presets[i].name,
                    style: TextStyle(
                        fontSize: 12,
                        color: selected ? scheme.primary : scheme.onSurfaceVariant,
                        fontWeight: selected ? FontWeight.bold : null)),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _adjustPanel() {
    final scheme = Theme.of(context).colorScheme;
    final p = _param;
    final v = _values[p]!;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: [
              for (final q in _Param.values)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    avatar: Icon(q.icon, size: 18),
                    // 有調整過的參數在名稱旁顯示數值,一眼看出改了哪些
                    label: Text(_values[q] == q.neutral
                        ? q.label
                        : '${q.label} ${_signed(q.display(_values[q]!))}'),
                    selected: p == q,
                    showCheckmark: false,
                    onSelected: (_) => setState(() => _param = q),
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            SizedBox(
              width: 44,
              child: Text(_signed(p.display(v)),
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleMedium),
            ),
            Expanded(
              child: GestureDetector(
                onDoubleTap: () => setState(() => _values[p] = p.neutral),
                child: Slider(
                  value: v.clamp(p.min, p.max),
                  min: p.min,
                  max: p.max,
                  onChanged: (x) => setState(() => _values[p] = x),
                ),
              ),
            ),
            IconButton(
              tooltip: '歸零',
              onPressed: v == p.neutral ? null : () => setState(() => _values[p] = p.neutral),
              icon: const Icon(Icons.restart_alt),
            ),
          ],
        ),
        Text(
          p == _Param.sharpen ? '銳化在儲存時套用,預覽看不到;雙擊滑桿可歸零' : '雙擊滑桿可歸零',
          style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
        ),
      ],
    );
  }

  String _signed(int n) => n > 0 ? '+$n' : '$n';

  Widget _repairPanel() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            Expanded(
              child: _ToggleCard(
                icon: Icons.auto_awesome,
                title: '提高畫質',
                subtitle: '增強細節與輪廓',
                selected: _enhance,
                onTap: () {
                  setState(() => _enhance = !_enhance);
                  _updatePreview();
                },
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _ToggleCard(
                icon: Icons.blur_off,
                title: '除霧',
                subtitle: '讓畫面更通透',
                selected: _dehaze,
                onTap: () {
                  setState(() => _dehaze = !_dehaze);
                  _updatePreview();
                },
              ),
            ),
          ],
        ),
        if (_dehaze)
          Row(
            children: [
              const SizedBox(width: 44, child: Text('強度', textAlign: TextAlign.center)),
              Expanded(
                child: Slider(
                  value: _dehazeStrength,
                  min: 0.1,
                  max: 1,
                  onChanged: (v) => setState(() => _dehazeStrength = v),
                  onChangeEnd: (_) => _updatePreview(),
                ),
              ),
              SizedBox(width: 44, child: Text('${(_dehazeStrength * 100).round()}')),
            ],
          ),
      ],
    );
  }
}

/// 可切換開關的大卡片(比小晶片好按,也放得下一行說明)。
class _ToggleCard extends StatelessWidget {
  const _ToggleCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.selected,
    required this.onTap,
  });

  final IconData icon;
  final String title, subtitle;
  final bool selected;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: selected ? scheme.primaryContainer : scheme.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Icon(icon, color: selected ? scheme.onPrimaryContainer : scheme.onSurfaceVariant),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: selected ? scheme.onPrimaryContainer : scheme.onSurface)),
                    Text(subtitle,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                            fontSize: 12,
                            color: selected ? scheme.onPrimaryContainer : scheme.onSurfaceVariant)),
                  ],
                ),
              ),
              Icon(selected ? Icons.check_circle : Icons.circle_outlined,
                  size: 20, color: selected ? scheme.onPrimaryContainer : scheme.outline),
            ],
          ),
        ),
      ),
    );
  }
}

// 背景 isolate 的入口放在頂層,閉包才不會帶到 State(含 BuildContext,無法傳到 isolate)。
Future<PhotoResult> _processInBackground(Uint8List bytes, PhotoOptions opts) =>
    Isolate.run(() => processPhoto(bytes, opts));

Future<Uint8List> _renderInBackground(
        Uint8List bytes, ColorMatrix matrix, double sharpen) =>
    Isolate.run(() => _render(bytes, matrix, sharpen));

/// 輸出:把色彩矩陣與銳化實際套用到像素上,回傳 JPG 位元組。
Uint8List _render(Uint8List bytes, ColorMatrix m, double sharpen) {
  var image = img.decodeImage(bytes);
  if (image == null) throw Exception('無法讀取圖片');
  image = img.bakeOrientation(image);

  for (final p in image) {
    final r = p.r.toDouble(), g = p.g.toDouble(), b = p.b.toDouble();
    p.r = (m[0] * r + m[1] * g + m[2] * b + m[4]).clamp(0, 255);
    p.g = (m[5] * r + m[6] * g + m[7] * b + m[9]).clamp(0, 255);
    p.b = (m[10] * r + m[11] * g + m[12] * b + m[14]).clamp(0, 255);
  }

  if (sharpen > 0) {
    // 銳化核心:中心 1+4a,上下左右 -a。
    final a = sharpen;
    image = img.convolution(
      image,
      filter: [0, -a, 0, -a, 1 + 4 * a, -a, 0, -a, 0],
      div: 1,
    );
  }
  return Uint8List.fromList(img.encodeJpg(image, quality: 95));
}

class CropPage extends StatefulWidget {
  final Uint8List bytes;
  const CropPage({super.key, required this.bytes});

  @override
  State<CropPage> createState() => _CropPageState();
}

class _CropPageState extends State<CropPage> {
  final _controller = CropController();
  bool _working = false;
  double? _aspect;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('裁切'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: FilledButton(
              onPressed: _working
                  ? null
                  : () {
                      setState(() => _working = true);
                      _controller.crop();
                    },
              child: const Text('完成'),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Crop(
              image: widget.bytes,
              controller: _controller,
              aspectRatio: _aspect,
              onCropped: (result) {
                if (result is CropSuccess) {
                  Navigator.of(context).pop(result.croppedImage);
                } else {
                  setState(() => _working = false);
                }
              },
            ),
          ),
          SafeArea(
            child: SizedBox(
              height: 56,
              child: ListView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                children: [
                  for (final o in const [
                    ('自由', null),
                    ('1:1', 1.0),
                    ('4:3', 4 / 3),
                    ('3:4', 3 / 4),
                    ('16:9', 16 / 9),
                    ('9:16', 9 / 16),
                  ])
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ChoiceChip(
                        label: Text(o.$1),
                        selected: _aspect == o.$2,
                        onSelected: (_) => setState(() => _aspect = o.$2),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
