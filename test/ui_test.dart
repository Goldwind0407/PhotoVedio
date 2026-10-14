import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:photo_video_editor/main.dart';
import 'package:photo_video_editor/photo_page.dart';
import 'package:photo_video_editor/video_page.dart';

Widget _app(Widget home) => MaterialApp(home: home);

void main() {
  testWidgets('首頁有修照片與修影片兩個入口', (tester) async {
    await tester.pumpWidget(_app(const HomePage()));
    expect(find.text('修照片'), findsOneWidget);
    expect(find.text('修影片'), findsOneWidget);
  });

  testWidgets('照片編輯:切換分頁、調整後可重設並復原', (tester) async {
    final m = cv.Mat.fromScalar(60, 80, cv.MatType.CV_8UC3, cv.Scalar(90, 120, 150, 0));
    final (_, png) = cv.imencode('.png', m);
    m.dispose();
    await tester.pumpWidget(_app(PhotoEditorPage(bytes: png)));

    // 預設在濾鏡分頁;沒有編輯時「重設」不可按
    expect(find.text('黑白'), findsOneWidget);
    expect(tester.widget<TextButton>(find.widgetWithText(TextButton, '重設')).onPressed, isNull);

    await tester.tap(find.text('調整'));
    await tester.pump();
    expect(find.text('雙擊滑桿可歸零'), findsOneWidget);
    await tester.drag(find.byType(Slider), const Offset(100, 0));
    await tester.pump();
    expect(find.textContaining('亮度 +'), findsOneWidget);

    await tester.tap(find.widgetWithText(TextButton, '重設'));
    await tester.pump();
    expect(find.textContaining('亮度 +'), findsNothing);
    await tester.pump(const Duration(milliseconds: 500)); // 等提示列滑入
    await tester.tap(find.text('復原'));
    await tester.pump();
    expect(find.textContaining('亮度 +'), findsOneWidget);

    await tester.tap(find.text('修復'));
    await tester.pump();
    expect(find.text('提高畫質'), findsOneWidget);
    expect(find.text('除霧'), findsOneWidget);
  });

  testWidgets('影片頁沒有影片時顯示選擇按鈕', (tester) async {
    await tester.pumpWidget(_app(const VideoPage()));
    expect(find.text('選擇影片'), findsOneWidget);
  });
}
