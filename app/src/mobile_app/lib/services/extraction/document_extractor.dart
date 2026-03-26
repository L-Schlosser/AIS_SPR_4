import 'dart:ui';

import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';

abstract class DocumentExtractor {
  const DocumentExtractor();

  String get documentType;

  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  });

  StructuredDocument buildDocument({
    required OcrDocument ocrDocument,
    required List<ExtractedField> fields,
    List<ExtractedEntity>? entities,
    double? confidence,
  }) {
    return StructuredDocument(
      documentType: documentType,
      classificationConfidence: confidence,
      rawText: ocrDocument.rawText,
      fields: fields,
      entities: entities ?? extractGenericEntities(ocrDocument),
      ocrDocument: ocrDocument,
    );
  }

  double computeConfidence({
    required List<ExtractedField> fields,
    int strongFieldCount = 0,
    int weakFieldCount = 0,
    double seedConfidence = 0.0,
  }) {
    final fieldCount = fields.length;
    final fieldScore = (fieldCount * 0.08).clamp(0.0, 0.35);
    final strongScore = (strongFieldCount * 0.14).clamp(0.0, 0.42);
    final weakScore = (weakFieldCount * 0.06).clamp(0.0, 0.18);
    final total =
        (seedConfidence * 0.25) + fieldScore + strongScore + weakScore;
    return double.parse(total.clamp(0.0, 0.99).toStringAsFixed(2));
  }

  List<ExtractedEntity> extractGenericEntities(OcrDocument document) {
    final entities = <ExtractedEntity>[];

    for (final line in document.allLines) {
      final date = extractDateFromText(line.text);
      if (date != null) {
        entities.add(
          ExtractedEntity(
            type: 'date',
            value: date,
            confidence: 0.90,
            boundingBox: line.boundingBox,
            sourceText: line.text,
          ),
        );
      }

      final time = extractTimeFromText(line.text);
      if (time != null) {
        entities.add(
          ExtractedEntity(
            type: 'time',
            value: time,
            confidence: 0.90,
            boundingBox: line.boundingBox,
            sourceText: line.text,
          ),
        );
      }

      final amounts = extractAmountsFromText(line.text);
      for (final amount in amounts) {
        entities.add(
          ExtractedEntity(
            type: 'amount',
            value: amount,
            confidence: 0.82,
            boundingBox: line.boundingBox,
            sourceText: line.text,
          ),
        );
      }

      final uid = extractUidFromText(line.text);
      if (uid != null) {
        entities.add(
          ExtractedEntity(
            type: 'uid',
            value: uid,
            confidence: 0.86,
            boundingBox: line.boundingBox,
            sourceText: line.text,
          ),
        );
      }
    }

    return entities;
  }

  FieldCandidate? findFirstDate(List<OcrLine> lines) {
    for (final line in lines) {
      final date = extractDateFromText(line.text);
      if (date != null) {
        return FieldCandidate(
          value: date,
          boundingBox: line.boundingBox,
          sourceText: line.text,
        );
      }
    }
    return null;
  }

  FieldCandidate? findFirstTime(List<OcrLine> lines) {
    for (final line in lines) {
      final time = extractTimeFromText(line.text);
      if (time != null) {
        return FieldCandidate(
          value: time,
          boundingBox: line.boundingBox,
          sourceText: line.text,
        );
      }
    }
    return null;
  }

  FieldCandidate? findUid(List<OcrLine> lines) {
    for (final line in lines) {
      final uid = extractUidFromText(line.text);
      if (uid != null) {
        return FieldCandidate(
          value: uid,
          boundingBox: line.boundingBox,
          sourceText: line.text,
        );
      }
    }
    return null;
  }

  FieldCandidate? findLargestAmount(List<OcrLine> lines) {
    FieldCandidate? best;
    double? bestValue;

    for (final line in lines) {
      final amounts = extractAmountsFromText(line.text);
      for (final amount in amounts) {
        final numeric = parseEuropeanNumber(amount);
        if (numeric == null) continue;

        if (bestValue == null || numeric > bestValue) {
          bestValue = numeric;
          best = FieldCandidate(
            value: amount,
            boundingBox: line.boundingBox,
            sourceText: line.text,
          );
        }
      }
    }

    return best;
  }

  FieldCandidate? findLabeledValue(
    List<OcrLine> lines, {
    required List<String> labelPatterns,
    bool preferDate = false,
  }) {
    for (int i = 0; i < lines.length; i++) {
      final current = lines[i];
      final normalized = normalize(current.text);

      final hasLabel = labelPatterns.any(
        (label) => normalized.contains(normalize(label)),
      );
      if (!hasLabel) continue;

      final inlineValue = extractValueAfterColon(current.text);
      if (inlineValue != null && inlineValue.isNotEmpty) {
        final value = preferDate
            ? (extractDateFromText(inlineValue) ?? inlineValue)
            : inlineValue;
        return FieldCandidate(
          value: value,
          boundingBox: current.boundingBox,
          sourceText: current.text,
        );
      }

      if (i + 1 < lines.length) {
        final next = lines[i + 1];
        final value = preferDate
            ? (extractDateFromText(next.text) ?? next.text.trim())
            : next.text.trim();
        if (value.isNotEmpty) {
          return FieldCandidate(
            value: value,
            boundingBox: next.boundingBox,
            sourceText: next.text,
          );
        }
      }
    }

    return null;
  }

  String? findLargestAmountInList(List<String> amounts) {
    String? bestAmount;
    double? bestValue;

    for (final amount in amounts) {
      final numeric = parseEuropeanNumber(amount);
      if (numeric == null) continue;

      if (bestValue == null || numeric > bestValue) {
        bestValue = numeric;
        bestAmount = amount;
      }
    }

    return bestAmount;
  }

  OcrLine? findBestSummaryLine(
    List<OcrLine> lines, {
    required List<String> preferredKeywords,
  }) {
    for (final line in lines) {
      final normalized = normalize(line.text);
      final matches = preferredKeywords.any(
        (keyword) => normalized.contains(normalize(keyword)),
      );
      if (matches && line.text.trim().length > 8) {
        return line;
      }
    }

    for (final line in lines) {
      if (line.text.trim().length > 20) {
        return line;
      }
    }

    return null;
  }

  String? extractValueAfterColon(String text) {
    final parts = text.split(':');
    if (parts.length < 2) return null;

    final value = parts.sublist(1).join(':').trim();
    return value.isEmpty ? null : value;
  }

  String? extractDateFromText(String text) {
    final match = RegExp(
      r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',
    ).firstMatch(text);
    return match?.group(0);
  }

  String? extractTimeFromText(String text) {
    final match = RegExp(r'\b\d{1,2}:\d{2}\b').firstMatch(text);
    return match?.group(0);
  }

  List<String> extractAmountsFromText(String text) {
    final normalizedText = text
        .replaceAllMapped(
          RegExp(r'(\d)[ ]*([.,])[ ]*(\d{2})\b'),
          (match) => '${match.group(1)}${match.group(2)}${match.group(3)}',
        )
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();

    final matches = RegExp(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})\b')
        .allMatches(normalizedText)
        .map((m) => m.group(0))
        .whereType<String>()
        .toList();

    return matches;
  }

  String? extractUidFromText(String text) {
    final match = RegExp(
      r'\b[A-Z]{2}U\d{8}\b',
      caseSensitive: false,
    ).firstMatch(text);
    return match?.group(0);
  }

  String? detectCurrency(String text) {
    final normalized = normalize(text);
    if (normalized.contains('eur') || text.contains('€')) return 'EUR';
    if (normalized.contains('usd') || text.contains(r'$')) return 'USD';
    if (normalized.contains('gbp') || text.contains('£')) return 'GBP';
    return null;
  }

  double? parseEuropeanNumber(String value) {
    var normalized = value.trim().replaceAll(RegExp(r'[^0-9,.-]'), '');

    final hasComma = normalized.contains(',');
    final hasDot = normalized.contains('.');

    if (hasComma && hasDot) {
      final lastComma = normalized.lastIndexOf(',');
      final lastDot = normalized.lastIndexOf('.');

      if (lastComma > lastDot) {
        normalized = normalized.replaceAll('.', '');
        normalized = normalized.replaceAll(',', '.');
      } else {
        normalized = normalized.replaceAll(',', '');
      }
    } else if (hasComma) {
      normalized = normalized.replaceAll(',', '.');
    }

    return double.tryParse(normalized);
  }

  String normalize(String input) {
    return input
        .toLowerCase()
        .replaceAll('ä', 'ae')
        .replaceAll('ö', 'oe')
        .replaceAll('ü', 'ue')
        .replaceAll('ß', 'ss')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }
}

class FieldCandidate {
  final String value;
  final Rect? boundingBox;
  final String? sourceText;

  const FieldCandidate({
    required this.value,
    this.boundingBox,
    this.sourceText,
  });
}
