import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class CaregivingLeaveExtractor extends DocumentExtractor {
  const CaregivingLeaveExtractor();

  @override
  String get documentType => 'caregiving_leave_confirmation';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final fields = <ExtractedField>[];
    final lines = ocrDocument.allLines;

    final date = findFirstDate(lines);
    if (date != null) {
      fields.add(ExtractedField(
        key: 'date',
        value: date.value,
        confidence: 0.85,
        boundingBox: date.boundingBox,
        sourceText: date.sourceText,
      ));
    }

    final summary = findBestSummaryLine(
      lines,
      preferredKeywords: [
        'pflegefreistellung',
        'pflege',
        'freistellung',
        'bestaetigung',
        'bestätigung',
      ],
    );
    if (summary != null) {
      fields.add(ExtractedField(
        key: 'summary',
        value: summary.text.trim(),
        confidence: 0.68,
        boundingBox: summary.boundingBox,
        sourceText: summary.text,
      ));
    }

    final confidence = computeConfidence(
      fields: fields,
      strongFieldCount: [
        fields.any((f) => f.key == 'date'),
        fields.any((f) => f.key == 'summary'),
      ].where((v) => v).length,
      weakFieldCount: 0,
      seedConfidence: seedConfidence ?? 0.0,
    );

    return buildDocument(
      ocrDocument: ocrDocument,
      fields: fields,
      confidence: confidence,
    );
  }
}
