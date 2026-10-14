import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:crop_your_image/crop_your_image.dart';
import 'package:flutter/material.dart';
import 'package:gal/gal.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import 'video_page.dart';

void main() => runApp(const EditorApp());

class EditorApp extends StatelessWidget {
  const EditorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '修圖工具',
      theme: ThemeData(
        colorSchemeSeed: Colors.deepPurple,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: const EditorPage(),
    );
  }
}

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

class EditorPage extends StatefulWidget {
  const EditorPage({super.key});

  @override
  State<EditorPage> createState() => _EditorPageState();
}

class _EditorPageState extends State<EditorPage> {
  Uint8List? _bytes; // 目前的底圖(裁切後會更新)
  int _filter = 0;
  double _brightness = 0; // -1 ~ 1
  double _contrast = 1; // 0.5 ~ 1.5
  double _saturation = 1; // 0 ~ 2
  double _sharpen = 0; // 0 ~ 1
  bool _busy = false;

  ColorMatrix get _matrix {
    var m = presets[_filter].matrix;
    m = multiply(m, saturationMatrix(_saturation));
    m = multiply(m, contrastBrightnessMatrix(_contrast, _brightness));
    return m;
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg)));
  }

  void _reset() => setState(() {
        _filter = 0;
        _brightness = 0;
        _contrast = 1;
        _saturation = 1;
        _sharpen = 0;
      });

  Future<void> _pick() async {
    final file = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    setState(() => _bytes = bytes);
    _reset();
  }

  Future<void> _crop() async {
    final bytes = _bytes;
    if (bytes == null) return;
    final result = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(builder: (_) => CropPage(bytes: bytes)),
    );
    if (result != null) setState(() => _bytes = result);
  }

  Future<void> _save() async {
    final bytes = _bytes;
    if (bytes == null) return;
    setState(() => _busy = true);
    try {
      final matrix = _matrix;
      final sharpen = _sharpen;
      // 在背景 isolate 處理,避免畫面卡住。
      final out = await Isolate.run(() => _render(bytes, matrix, sharpen));
      if (Platform.isAndroid || Platform.isIOS) {
        await Gal.putImageBytes(out, name: 'edit_${_stamp()}');
        _toast('已儲存到相簿');
      } else {
        final dir = await getApplicationDocumentsDirectory();
        final path = '${dir.path}${Platform.pathSeparator}edit_${_stamp()}.jpg';
        await File(path).writeAsBytes(out);
        _toast('已儲存:$path');
      }
    } catch (e) {
      _toast('儲存失敗:$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _stamp() => DateTime.now().millisecondsSinceEpoch.toString();

  @override
  Widget build(BuildContext context) {
    final hasImage = _bytes != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('修圖工具'),
        actions: [
          IconButton(
            tooltip: '修影片',
            icon: const Icon(Icons.movie_filter_outlined),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const VideoPage()),
            ),
          ),
          IconButton(
            tooltip: '選擇照片',
            icon: const Icon(Icons.photo_library_outlined),
            onPressed: _pick,
          ),
          IconButton(
            tooltip: '重設',
            icon: const Icon(Icons.restart_alt),
            onPressed: hasImage ? _reset : null,
          ),
          IconButton(
            tooltip: '儲存',
            icon: const Icon(Icons.save_alt),
            onPressed: hasImage && !_busy ? _save : null,
          ),
        ],
      ),
      body: !hasImage
          ? Center(
              child: FilledButton.icon(
                onPressed: _pick,
                icon: const Icon(Icons.add_photo_alternate_outlined),
                label: const Text('選擇一張照片'),
              ),
            )
          : Column(
              children: [
                Expanded(
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      ColorFiltered(
                        colorFilter: ColorFilter.matrix(_matrix),
                        child: Image.memory(_bytes!, fit: BoxFit.contain),
                      ),
                      if (_busy) const CircularProgressIndicator(),
                    ],
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
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              height: 44,
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: [
                  for (var i = 0; i < presets.length; i++)
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ChoiceChip(
                        label: Text(presets[i].name),
                        selected: _filter == i,
                        onSelected: (_) => setState(() => _filter = i),
                      ),
                    ),
                  ActionChip(
                    avatar: const Icon(Icons.crop, size: 18),
                    label: const Text('裁切'),
                    onPressed: _crop,
                  ),
                ],
              ),
            ),
            _slider('亮度', _brightness, -0.5, 0.5, 0,
                (v) => setState(() => _brightness = v)),
            _slider('對比', _contrast, 0.5, 1.5, 1,
                (v) => setState(() => _contrast = v)),
            _slider('飽和', _saturation, 0, 2, 1,
                (v) => setState(() => _saturation = v)),
            _slider('銳化', _sharpen, 0, 1, 0,
                (v) => setState(() => _sharpen = v),
                note: '儲存時套用'),
          ],
        ),
      ),
    );
  }

  Widget _slider(String label, double value, double min, double max,
      double neutral, ValueChanged<double> onChanged,
      {String? note}) {
    return Row(
      children: [
        SizedBox(width: 40, child: Text(label)),
        Expanded(
          child: GestureDetector(
            onDoubleTap: () => onChanged(neutral),
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              onChanged: onChanged,
            ),
          ),
        ),
        SizedBox(
          width: 64,
          child: Text(
            note ?? (value - neutral).toStringAsFixed(2),
            style: Theme.of(context).textTheme.bodySmall,
            textAlign: TextAlign.end,
          ),
        ),
      ],
    );
  }
}

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
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: _working
                ? null
                : () {
                    setState(() => _working = true);
                    _controller.crop();
                  },
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
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: Wrap(
                spacing: 8,
                children: [
                  for (final o in const [
                    ('自由', null),
                    ('1:1', 1.0),
                    ('4:3', 4 / 3),
                    ('3:4', 3 / 4),
                    ('16:9', 16 / 9),
                    ('9:16', 9 / 16),
                  ])
                    ChoiceChip(
                      label: Text(o.$1),
                      selected: _aspect == o.$2,
                      onSelected: (_) => setState(() => _aspect = o.$2),
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
