import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class UnknownExtractor extends DocumentExtractor {
  const UnknownExtractor();

  @override
  String get documentType => 'unknown';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final confidence = double.parse((seedConfidence ?? 0.15).toStringAsFixed(2));

    return buildDocument(
      ocrDocument: ocrDocument,
      fields: const [],
      confidence: confidence,
    );
  }
}
