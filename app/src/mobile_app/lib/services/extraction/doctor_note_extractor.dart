import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class DoctorNoteExtractor extends DocumentExtractor {
  const DoctorNoteExtractor();

  @override
  String get documentType => 'doctor_note';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final fields = <ExtractedField>[];
    final lines = ocrDocument.allLines;

    final doctor = findLabeledValue(
      lines,
      labelPatterns: ['arzt', 'ärztin', 'ordination', 'dr.', 'doktor'],
    );
    if (doctor != null) {
      fields.add(ExtractedField(
        key: 'doctor_name',
        value: doctor.value,
        confidence: 0.72,
        boundingBox: doctor.boundingBox,
        sourceText: doctor.sourceText,
      ));
    }

    final patient = findLabeledValue(
      lines,
      labelPatterns: ['patient', 'patientin', 'name'],
    );
    if (patient != null) {
      fields.add(ExtractedField(
        key: 'patient_name',
        value: patient.value,
        confidence: 0.72,
        boundingBox: patient.boundingBox,
        sourceText: patient.sourceText,
      ));
    }

    final date = findFirstDate(lines);
    if (date != null) {
      fields.add(ExtractedField(
        key: 'date',
        value: date.value,
        confidence: 0.88,
        boundingBox: date.boundingBox,
        sourceText: date.sourceText,
      ));
    }

    final summary = findBestSummaryLine(
      lines,
      preferredKeywords: ['diagnose', 'befund', 'bestaetigung', 'bestätigung'],
    );
    if (summary != null) {
      fields.add(ExtractedField(
        key: 'summary',
        value: summary.text.trim(),
        confidence: 0.60,
        boundingBox: summary.boundingBox,
        sourceText: summary.text,
      ));
    }

    final confidence = computeConfidence(
      fields: fields,
      strongFieldCount: [
        fields.any((f) => f.key == 'doctor_name'),
        fields.any((f) => f.key == 'patient_name'),
        fields.any((f) => f.key == 'date'),
      ].where((v) => v).length,
      weakFieldCount: [
        fields.any((f) => f.key == 'summary'),
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
