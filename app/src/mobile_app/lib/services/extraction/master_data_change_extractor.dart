import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class MasterDataChangeExtractor extends DocumentExtractor {
  const MasterDataChangeExtractor();

  @override
  String get documentType => 'master_data_change';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final fields = <ExtractedField>[];
    final lines = ocrDocument.allLines;

    final name = findLabeledValue(
      lines,
      labelPatterns: ['name', 'familienname', 'vorname'],
    );
    if (name != null) {
      fields.add(ExtractedField(
        key: 'name',
        value: name.value,
        confidence: 0.70,
        boundingBox: name.boundingBox,
        sourceText: name.sourceText,
      ));
    }

    final address = findLabeledValue(
      lines,
      labelPatterns: ['adresse', 'wohnadresse', 'hauptwohnsitz', 'nebenwohnsitz'],
    );
    if (address != null) {
      fields.add(ExtractedField(
        key: 'address',
        value: address.value,
        confidence: 0.72,
        boundingBox: address.boundingBox,
        sourceText: address.sourceText,
      ));
    }

    final date = findFirstDate(lines);
    if (date != null) {
      fields.add(ExtractedField(
        key: 'date',
        value: date.value,
        confidence: 0.82,
        boundingBox: date.boundingBox,
        sourceText: date.sourceText,
      ));
    }

    final confidence = computeConfidence(
      fields: fields,
      strongFieldCount: [
        fields.any((f) => f.key == 'name'),
        fields.any((f) => f.key == 'address'),
      ].where((v) => v).length,
      weakFieldCount: [
        fields.any((f) => f.key == 'date'),
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
