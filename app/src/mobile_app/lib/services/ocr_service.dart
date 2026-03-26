import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:path_provider/path_provider.dart';

import '../models/ocr_models.dart';

/// Service responsible for:
/// - loading an image from assets (optional helper)
/// - running ML Kit OCR
/// - converting ML Kit output into our own OCR models
class OcrService {
  final TextRecognizer _textRecognizer;

  OcrService({
    TextRecognitionScript script = TextRecognitionScript.latin,
  }) : _textRecognizer = TextRecognizer(script: script);

  /// Copies an asset image into a temporary file so ML Kit can process it.
  Future<File> copyAssetToFile(String assetPath, String fileName) async {
    final byteData = await rootBundle.load(assetPath);
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/$fileName');

    await file.writeAsBytes(byteData.buffer.asUint8List(), flush: true);
    return file;
  }

  /// Runs OCR on an image file path and returns a structured [OcrDocument].
  Future<OcrDocument> processImageFromPath(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final recognizedText = await _textRecognizer.processImage(inputImage);

    final blocks = recognizedText.blocks.map(_mapBlock).toList();

    return OcrDocument(
      rawText: recognizedText.text,
      blocks: blocks,
    );
  }

  /// Convenience helper: copies an asset file and runs OCR on it.
  Future<OcrDocument> processAssetImage(String assetPath, String fileName) async {
    final file = await copyAssetToFile(assetPath, fileName);
    return processImageFromPath(file.path);
  }

  /// Converts an ML Kit text block into our own [OcrBlock] model.
  OcrBlock _mapBlock(TextBlock block) {
    final lines = block.lines.map(_mapLine).toList();

    return OcrBlock(
      text: block.text,
      boundingBox: block.boundingBox,
      lines: lines,
    );
  }

  /// Converts an ML Kit line into our own [OcrLine] model.
  OcrLine _mapLine(TextLine line) {
    final words = line.elements.map(_mapWord).toList();

    return OcrLine(
      text: line.text,
      boundingBox: line.boundingBox,
      words: words,
    );
  }

  /// Converts an ML Kit element/word into our own [OcrWord] model.
  OcrWord _mapWord(TextElement element) {
    return OcrWord(
      text: element.text,
      boundingBox: element.boundingBox,
    );
  }

  /// Closes the underlying ML Kit recognizer.
  Future<void> dispose() async {
    await _textRecognizer.close();
  }
}
