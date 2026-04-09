import 'dart:io';

import '../models/editable_field_data.dart';
import '../models/ocr_models.dart';
import '../models/structured_document.dart';
import '../services/classification_service.dart';

class ReviewImageItem {
  final File imageFile;

  OcrDocument? ocrDocument;
  StructuredDocument? structuredDocument;
  ClassificationResult? classificationResult;
  String? selectedDocumentType;

  bool isProcessing;

  final List<EditableFieldData> editableFields;

  ReviewImageItem({
    required this.imageFile,
    this.ocrDocument,
    this.structuredDocument,
    this.classificationResult,
    this.selectedDocumentType,
    this.isProcessing = false,
    List<EditableFieldData>? editableFields,
  }) : editableFields = editableFields ?? [];

  void dispose() {
    for (final field in editableFields) {
      field.dispose();
    }
    editableFields.clear();
  }
}
