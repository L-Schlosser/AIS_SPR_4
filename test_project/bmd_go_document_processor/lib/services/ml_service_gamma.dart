import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'dart:io';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'services/ml_service_ocr.dart';

class GammaService{
  static const List<String> documentTypes = [
    'recipe',
    'bill',
    'doctor_note',
    'delivery_note',
    'receipt',
    'contract',
  ];

  Future<OCRClassificationResult> processImages(List<File> images) async {
    String fullText = '';

    for (final image in images) {
      final inputImage = InputImage.fromFile(image);
      final RecognizedText recognizedText =
          await textRecognizer.processImage(inputImage);
s
      fullText += recognizedText.text + '\n';
    }

    print("OCR TEXT:\n$fullText");

    // Extract structured data
    final extractedData = _extractReceiptData(fullText);

    return OCRClassificationResult(
      documentType: documentTypes[Random().nextInt(documentTypes.length)],
      confidence: 0.9,
      infos: extractedData,
    );
  }

    Map<String, String> _extractReceiptData(String text) {
    final Map<String, String> data = {};
    data['RawText'] = text;

    return data;
  }

  void dispose() {
    textRecognizer.close();
  }


  static Future<Map<String, dynamic>> classifyDocument(File imageFile) async {
    return OCRClassificationResult(
      documentType: documentTypes[Random().nextInt(documentTypes.length)],
      confidence: 0.9,
      infos: {'RawText': 'Sample extracted text from OCR'},
    ).toJson();
  }
}

class MLServiceTest {
  
}