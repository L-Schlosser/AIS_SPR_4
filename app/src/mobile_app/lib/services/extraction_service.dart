import '../models/ocr_models.dart';
import '../models/structured_document.dart';
import 'extraction/caregiving_leave_extractor.dart';
import 'extraction/delivery_note_extractor.dart';
import 'extraction/document_extractor.dart';
import 'extraction/doctor_note_extractor.dart';
import 'extraction/invoice_extractor.dart';
import 'extraction/master_data_change_extractor.dart';
import 'extraction/receipt_extractor.dart';
import 'extraction/unknown_extractor.dart';

class ExtractionService {
  final List<DocumentExtractor> _extractors;

  ExtractionService({List<DocumentExtractor>? extractors})
    : _extractors =
          extractors ??
          const [
            ReceiptExtractor(),
            InvoiceExtractor(),
            DeliveryNoteExtractor(),
            DoctorNoteExtractor(),
            CaregivingLeaveExtractor(),
            MasterDataChangeExtractor(),
          ];

  List<DocumentExtractor> get extractors => List.unmodifiable(_extractors);

  StructuredDocument extractDocument({
    required OcrDocument ocrDocument,
    String? documentType,
    double? classificationConfidence,
  }) {
    if (documentType != null && documentType.trim().isNotEmpty) {
      final extractor = _findExtractor(documentType);
      return extractor.extract(
        ocrDocument: ocrDocument,
        seedConfidence: classificationConfidence,
      );
    }

    return extractBestDocument(
      ocrDocument: ocrDocument,
      seedConfidence: classificationConfidence,
    );
  }

  StructuredDocument extractBestDocument({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    StructuredDocument? best;

    for (final extractor in _extractors) {
      final candidate = extractor.extract(
        ocrDocument: ocrDocument,
        seedConfidence: seedConfidence,
      );

      if (best == null) {
        best = candidate;
        continue;
      }

      final bestConfidence = best.classificationConfidence ?? 0.0;
      final candidateConfidence = candidate.classificationConfidence ?? 0.0;

      if (candidateConfidence > bestConfidence) {
        best = candidate;
      }
    }

    return best ??
        const UnknownExtractor().extract(
          ocrDocument: ocrDocument,
          seedConfidence: 0.0,
        );
  }

  DocumentExtractor _findExtractor(String documentType) {
    final normalized = documentType.trim().toLowerCase();

    for (final extractor in _extractors) {
      if (extractor.documentType == normalized) return extractor;
    }

    return const UnknownExtractor();
  }
}
