import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'dart:io';
import 'package:pdfx/pdfx.dart';
import 'package:path_provider/path_provider.dart';

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

  /// Process PDF by rendering pages and running OCR.
  /// Improvements for speed:
  /// - Cap rendered page dimensions with [maxDimension] to avoid huge images.
  /// - Use a single `TextRecognizer` instance (already reused).
  /// - Delete temp page files after OCR to reduce I/O pressure.
  Future<String> processPdf(File pdfFile, {int maxDimension = 1200, double scale = 1.0}) async {
    final document = await PdfDocument.openFile(pdfFile.path);

    List<File> images = [];
    final tempDir = await getTemporaryDirectory();

    for (int i = 1; i <= document.pagesCount; i++) {
      final page = await document.getPage(i);
      final pageImage = await page.render(
        width: (page.width * scale) > maxDimension.toDouble() ? maxDimension.toDouble(): (page.width * scale),
        height: (page.height * scale) > maxDimension.toDouble() ? maxDimension.toDouble(): (page.height * scale),
        format: PdfPageImageFormat.png,
      );

      final file = File('${tempDir.path}/pdf_page_$i.png');
      await file.writeAsBytes(pageImage!.bytes);

      images.add(file);

      await page.close();
    }

    await document.close();

    // reuse your existing OCR
    final buffer = StringBuffer();
    for (final file in images) {
      final inputImage = InputImage.fromFile(file);
      final RecognizedText recognizedText = await textRecognizer.processImage(inputImage);
      buffer.writeln(recognizedText.text);
      await file.delete(); // clean up temporary file to save disk and IO
    }
    return buffer.toString();
  }

  void dispose() {
    textRecognizer.close();
  }
}
