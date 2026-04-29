import 'dart:typed_data';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'dart:io';
import 'package:pdfx/pdfx.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/material.dart';


class OCRClassificationResult {
  final String documentType;
  final double? confidence;
  final Map<String, String> infos;

  OCRClassificationResult({
    required this.documentType,
    this.confidence,
    required this.infos,
  });
}

class OCRMLService {
  final textRecognizer = TextRecognizer();

  Future<String> processImages(List<File> images) async {
    String fullText = '';

    for (final image in images) {
      final inputImage = InputImage.fromFile(image);
      final RecognizedText recognizedText =
          await textRecognizer.processImage(inputImage);

      fullText += recognizedText.text + '\n';
    }

    print("OCR TEXT:\n$fullText");

    return fullText;
  }

  Future<String> processPdf(File pdfFile) async {
    final document = await PdfDocument.openFile(pdfFile.path);

    List<File> images = [];
    final tempDir = await getTemporaryDirectory();

    for (int i = 1; i <= document.pagesCount; i++) {
      final page = await document.getPage(i);

      final pageImage = await page.render(
        width: page.width * 2,
        height: page.height * 2,
        format: PdfPageImageFormat.png,
      );

      final file = File('${tempDir.path}/pdf_page_$i.png');
      await file.writeAsBytes(pageImage!.bytes);

      images.add(file);

      await page.close();
    }

    await document.close();

    // reuse your existing OCR
    return await processImages(images);
  }

  void dispose() {
    textRecognizer.close();
  }
}
