import 'dart:ui';

import 'ocr_models.dart';

/// Represents one extracted field in a structured document.
/// Example: "date" -> "24.03.2026"
class ExtractedField {
  final String key;
  final String value;
  final double? confidence;
  final Rect? boundingBox;
  final String? sourceText;

  const ExtractedField({
    required this.key,
    required this.value,
    this.confidence,
    this.boundingBox,
    this.sourceText,
  });

  Map<String, dynamic> toJson() {
    return {
      'key': key,
      'value': value,
      'confidence': confidence,
      'bounding_box': boundingBox != null ? rectToJson(boundingBox!) : null,
      'source_text': sourceText,
    };
  }

  factory ExtractedField.fromJson(Map<String, dynamic> json) {
    return ExtractedField(
      key: json['key'] as String? ?? '',
      value: json['value'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble(),
      boundingBox: json['bounding_box'] != null
          ? rectFromJson(json['bounding_box'] as Map<String, dynamic>)
          : null,
      sourceText: json['source_text'] as String?,
    );
  }
}

/// Represents one extracted entity with type information.
/// Example: type=date, value=24.03.2026
class ExtractedEntity {
  final String type;
  final String value;
  final double? confidence;
  final Rect? boundingBox;
  final String? sourceText;

  const ExtractedEntity({
    required this.type,
    required this.value,
    this.confidence,
    this.boundingBox,
    this.sourceText,
  });

  Map<String, dynamic> toJson() {
    return {
      'type': type,
      'value': value,
      'confidence': confidence,
      'bounding_box': boundingBox != null ? rectToJson(boundingBox!) : null,
      'source_text': sourceText,
    };
  }

  factory ExtractedEntity.fromJson(Map<String, dynamic> json) {
    return ExtractedEntity(
      type: json['type'] as String? ?? '',
      value: json['value'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble(),
      boundingBox: json['bounding_box'] != null
          ? rectFromJson(json['bounding_box'] as Map<String, dynamic>)
          : null,
      sourceText: json['source_text'] as String?,
    );
  }
}

/// Final structured output of the OCR/document understanding pipeline.
class StructuredDocument {
  final String documentType;
  final double? classificationConfidence;
  final String rawText;
  final List<ExtractedField> fields;
  final List<ExtractedEntity> entities;
  final OcrDocument? ocrDocument;

  const StructuredDocument({
    required this.documentType,
    required this.rawText,
    required this.fields,
    required this.entities,
    this.classificationConfidence,
    this.ocrDocument,
  });

  /// Returns the first field value for a given key, or null if it does not exist.
  String? getFieldValue(String key) {
    for (final field in fields) {
      if (field.key == key) return field.value;
    }
    return null;
  }

  /// Returns all fields as a simple key-value map.
  /// If a key appears multiple times, the last one wins.
  Map<String, String> get fieldsAsMap {
    return {
      for (final field in fields) field.key: field.value,
    };
  }

  Map<String, dynamic> toJson() {
    return {
      'document_type': documentType,
      'classification_confidence': classificationConfidence,
      'raw_text': rawText,
      'fields': fields.map((field) => field.toJson()).toList(),
      'entities': entities.map((entity) => entity.toJson()).toList(),
      'ocr_document': ocrDocument?.toJson(),
    };
  }

  factory StructuredDocument.fromJson(Map<String, dynamic> json) {
    final fieldsJson = json['fields'] as List<dynamic>? ?? const [];
    final entitiesJson = json['entities'] as List<dynamic>? ?? const [];

    return StructuredDocument(
      documentType: json['document_type'] as String? ?? 'unknown',
      classificationConfidence:
          (json['classification_confidence'] as num?)?.toDouble(),
      rawText: json['raw_text'] as String? ?? '',
      fields: fieldsJson
          .map((fieldJson) =>
              ExtractedField.fromJson(fieldJson as Map<String, dynamic>))
          .toList(),
      entities: entitiesJson
          .map((entityJson) =>
              ExtractedEntity.fromJson(entityJson as Map<String, dynamic>))
          .toList(),
      ocrDocument: json['ocr_document'] != null
          ? OcrDocument.fromJson(json['ocr_document'] as Map<String, dynamic>)
          : null,
    );
  }

  /// Useful starter for documents that have not been classified yet.
  factory StructuredDocument.empty({OcrDocument? ocrDocument}) {
    return StructuredDocument(
      documentType: 'unknown',
      classificationConfidence: null,
      rawText: ocrDocument?.rawText ?? '',
      fields: const [],
      entities: const [],
      ocrDocument: ocrDocument,
    );
  }

  StructuredDocument copyWith({
    String? documentType,
    double? classificationConfidence,
    String? rawText,
    List<ExtractedField>? fields,
    List<ExtractedEntity>? entities,
    OcrDocument? ocrDocument,
  }) {
    return StructuredDocument(
      documentType: documentType ?? this.documentType,
      classificationConfidence:
          classificationConfidence ?? this.classificationConfidence,
      rawText: rawText ?? this.rawText,
      fields: fields ?? this.fields,
      entities: entities ?? this.entities,
      ocrDocument: ocrDocument ?? this.ocrDocument,
    );
  }
}
