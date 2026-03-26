import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class DeliveryNoteExtractor extends DocumentExtractor {
  const DeliveryNoteExtractor();

  @override
  String get documentType => 'delivery_note';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final fields = <ExtractedField>[];
    final lines = ocrDocument.allLines;

    final deliveryNumber = findLabeledValue(
      lines,
      labelPatterns: ['lieferscheinnummer', 'liefernummer', 'lieferschein nr'],
    );
    if (deliveryNumber != null) {
      fields.add(ExtractedField(
        key: 'delivery_number',
        value: deliveryNumber.value,
        confidence: 0.90,
        boundingBox: deliveryNumber.boundingBox,
        sourceText: deliveryNumber.sourceText,
      ));
    }

    final deliveryDate = findLabeledValue(
          lines,
          labelPatterns: ['lieferdatum', 'datum'],
          preferDate: true,
        ) ??
        findFirstDate(lines);
    if (deliveryDate != null) {
      fields.add(ExtractedField(
        key: 'delivery_date',
        value: deliveryDate.value,
        confidence: 0.88,
        boundingBox: deliveryDate.boundingBox,
        sourceText: deliveryDate.sourceText,
      ));
    }

    final recipient = findLabeledValue(
      lines,
      labelPatterns: ['empfaenger', 'empfänger', 'lieferadresse', 'an'],
    );
    if (recipient != null) {
      fields.add(ExtractedField(
        key: 'recipient',
        value: recipient.value,
        confidence: 0.75,
        boundingBox: recipient.boundingBox,
        sourceText: recipient.sourceText,
      ));
    }

    final confidence = computeConfidence(
      fields: fields,
      strongFieldCount: [
        fields.any((f) => f.key == 'delivery_number'),
        fields.any((f) => f.key == 'delivery_date'),
      ].where((v) => v).length,
      weakFieldCount: [
        fields.any((f) => f.key == 'recipient'),
      ].where((v) => v).length,
      seedConfidence: seedConfidence ?? 0.0,
    );

    return buildDocument(
      ocrDocument: ocrDocument,
      fields: fields,
      confidence: confidence,
    );
  }
}
