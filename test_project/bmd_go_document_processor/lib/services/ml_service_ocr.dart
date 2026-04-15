import 'dart:io';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

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

  Future<OCRClassificationResult> processImages(List<File> images) async {
    String fullText = '';

    for (final image in images) {
      final inputImage = InputImage.fromFile(image);
      final RecognizedText recognizedText =
          await textRecognizer.processImage(inputImage);

      fullText += recognizedText.text + '\n';
    }

    print("OCR TEXT:\n$fullText");

    // Extract structured data
    final extractedData = _extractReceiptData(fullText);

    return OCRClassificationResult(
      documentType: 'receipt',
      confidence: 0.9,
      infos: extractedData,
    );
  }

  Map<String, String> _extractReceiptData(String text) {
    final Map<String, String> data = {};

    // VERY simple parsing (you can improve this later)
    // final lines = text.split('\n');

    // for (var line in lines) {
    //   if (line.contains('€')) {
    //     data['Umsatz'] = line;
    //   }

    //   if (RegExp(r'\d{2}\.\d{2}\.\d{4}').hasMatch(line)) {
    //     data['Datum'] = line;
    //   }
    // }

    data['RawText'] = text;

    return data;
  }

  void dispose() {
    textRecognizer.close();
  }
}