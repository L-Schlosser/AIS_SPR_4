import 'dart:io';
import 'services/ml_service_ocr.dart';

//LIST OF POSSIBLE TYPES: recipe, bill, doctor_note, delivery_note, receipt, contract


class GammaService{
  Future<OCRClassificationResult> processImages(List<File> images) async {
    final type = 'TYPE FROM GAMMA';
    final extractedData = 'TEXT FROM GAMMA';

    return OCRClassificationResult(
      documentType: type,
      infos: extractedData,
    );
  }
}
