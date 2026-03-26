import 'dart:ui' as ui;

import 'package:flutter/material.dart';

/// Simple overlay painter for OCR/document bounding boxes.
///
/// It can draw:
/// - OCR word/line/entity boxes
/// - an optional label above each box
///
/// The [scaleX] and [scaleY] values should map OCR/image coordinates
/// to the currently displayed image size.
class BoundingBoxPainter extends CustomPainter {
  final List<BoundingBoxItem> items;
  final double scaleX;
  final double scaleY;
  final bool showLabels;

  const BoundingBoxPainter({
    required this.items,
    required this.scaleX,
    required this.scaleY,
    this.showLabels = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (final item in items) {
      final rect = Rect.fromLTRB(
        item.rect.left * scaleX,
        item.rect.top * scaleY,
        item.rect.right * scaleX,
        item.rect.bottom * scaleY,
      );

      boxPaint.color = item.color;
      canvas.drawRect(rect, boxPaint);

      if (showLabels && item.label != null && item.label!.trim().isNotEmpty) {
        _drawLabel(
          canvas,
          rect,
          item.label!,
          item.color,
        );
      }
    }
  }

  void _drawLabel(
    Canvas canvas,
    Rect rect,
    String label,
    Color color,
  ) {
    final textStyle = TextStyle(
      color: color,
      fontSize: 12,
      fontWeight: FontWeight.w600,
      backgroundColor: Colors.white,
    );

    final textSpan = TextSpan(
      text: label,
      style: textStyle,
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: ui.TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: 160);

    final offset = Offset(
      rect.left,
      (rect.top - textPainter.height - 4).clamp(0.0, double.infinity),
    );

    final backgroundRect = Rect.fromLTWH(
      offset.dx - 2,
      offset.dy - 1,
      textPainter.width + 4,
      textPainter.height + 2,
    );

    final backgroundPaint = Paint()..color = Colors.white.withOpacity(0.85);
    canvas.drawRect(backgroundRect, backgroundPaint);

    textPainter.paint(canvas, offset);
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.items != items ||
        oldDelegate.scaleX != scaleX ||
        oldDelegate.scaleY != scaleY ||
        oldDelegate.showLabels != showLabels;
  }
}

/// One item to draw on the image overlay.
class BoundingBoxItem {
  final Rect rect;
  final String? label;
  final Color color;

  const BoundingBoxItem({
    required this.rect,
    this.label,
    this.color = Colors.red,
  });
}
