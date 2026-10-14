import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'photo_page.dart';
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
      home: const HomePage(),
    );
  }
}

/// 首頁:修照片、修影片兩個入口,點了直接開相簿選檔。
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  Future<void> _photo(BuildContext context) async {
    final file = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (file == null || !context.mounted) return;
    final bytes = await file.readAsBytes();
    if (!context.mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => PhotoEditorPage(bytes: bytes)),
    );
  }

  Future<void> _video(BuildContext context) async {
    final file = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (file == null || !context.mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => VideoPage(initialPath: file.path)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 32, 20, 20),
          children: [
            Text('修圖工具', style: theme.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text('所有處理都在手機上完成,不會上傳。',
                style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
            const SizedBox(height: 28),
            _EntryCard(
              icon: Icons.photo_outlined,
              title: '修照片',
              features: const ['濾鏡', '調色', '裁切', '提高畫質', '除霧'],
              onTap: () => _photo(context),
            ),
            const SizedBox(height: 16),
            _EntryCard(
              icon: Icons.movie_outlined,
              title: '修影片',
              features: const ['防手震', '果凍修正', '提高畫質', '除霧'],
              onTap: () => _video(context),
            ),
          ],
        ),
      ),
    );
  }
}

class _EntryCard extends StatelessWidget {
  const _EntryCard({
    required this.icon,
    required this.title,
    required this.features,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final List<String> features;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: scheme.surfaceContainerHigh,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              CircleAvatar(
                radius: 30,
                backgroundColor: scheme.primaryContainer,
                child: Icon(icon, size: 30, color: scheme.onPrimaryContainer),
              ),
              const SizedBox(width: 18),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: [
                        for (final f in features)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: scheme.surfaceContainerHighest,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(f, style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: scheme.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }
}
