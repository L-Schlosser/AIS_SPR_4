import 'dart:ui';

import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class ReceiptExtractor extends DocumentExtractor {
  const ReceiptExtractor();

  @override
  String get documentType => 'receipt';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final fields = <ExtractedField>[];
    final lines = ocrDocument.allLines;

    final vendorLine = lines.isNotEmpty ? lines.first : null;
    if (vendorLine != null && vendorLine.text.trim().isNotEmpty) {
      fields.add(
        ExtractedField(
          key: 'vendor',
          value: vendorLine.text.trim(),
          confidence: 0.65,
          boundingBox: vendorLine.boundingBox,
          sourceText: vendorLine.text,
        ),
      );
    }

    final dateMatch = findFirstDate(lines);
    if (dateMatch != null) {
      fields.add(
        ExtractedField(
          key: 'date',
          value: dateMatch.value,
          confidence: 0.90,
          boundingBox: dateMatch.boundingBox,
          sourceText: dateMatch.sourceText,
        ),
      );
    }

    final timeMatch = findFirstTime(lines);
    if (timeMatch != null) {
      fields.add(
        ExtractedField(
          key: 'time',
          value: timeMatch.value,
          confidence: 0.90,
          boundingBox: timeMatch.boundingBox,
          sourceText: timeMatch.sourceText,
        ),
      );
    }

    final totalMatch = _findReceiptTotal(lines) ?? findLargestAmount(lines);
    if (totalMatch != null) {
      fields.add(
        ExtractedField(
          key: 'total_amount',
          value: totalMatch.value,
          confidence: 0.92,
          boundingBox: totalMatch.boundingBox,
          sourceText: totalMatch.sourceText,
        ),
      );
    }

    final uidMatch = findUid(lines);
    if (uidMatch != null) {
      fields.add(
        ExtractedField(
          key: 'uid',
          value: uidMatch.value,
          confidence: 0.85,
          boundingBox: uidMatch.boundingBox,
          sourceText: uidMatch.sourceText,
        ),
      );
    }

    final currency = detectCurrency(ocrDocument.rawText);
    if (currency != null) {
      fields.add(
        ExtractedField(key: 'currency', value: currency, confidence: 0.80),
      );
    }

    final confidence = computeConfidence(
      fields: fields,
      strongFieldCount: [
        fields.any((f) => f.key == 'total_amount'),
        fields.any((f) => f.key == 'date'),
        fields.any((f) => f.key == 'vendor'),
      ].where((v) => v).length,
      weakFieldCount: [
        fields.any((f) => f.key == 'time'),
        fields.any((f) => f.key == 'currency'),
        fields.any((f) => f.key == 'uid'),
      ].where((v) => v).length,
      seedConfidence: seedConfidence ?? 0.0,
    );

    return buildDocument(
      ocrDocument: ocrDocument,
      fields: fields,
      confidence: confidence,
    );
  }

  FieldCandidate? _findReceiptTotal(List<OcrLine> lines) {
    return _findLabeledAmount(
          lines,
          labelPatterns: ['summe', 'gesamt', 'total', 'zahlbetrag', 'endsumme'],
        ) ??
        _findLabeledAmount(lines, labelPatterns: ['betrag', 'eur']);
  }

  FieldCandidate? _findLabeledAmount(
    List<OcrLine> lines, {
    required List<String> labelPatterns,
    List<String> blockedPhrases = const [],
    double maxVerticalDistance = 70,
  }) {
    for (int i = lines.length - 1; i >= 0; i--) {
      final labelLine = lines[i];
      final normalized = normalize(labelLine.text);

      final hasLabel = labelPatterns.any(
        (label) => normalized.contains(normalize(label)),
      );

      if (!hasLabel) continue;
      if (_isBadAmountLabelContext(
        normalized,
        blockedPhrases: blockedPhrases,
      )) {
        continue;
      }

      final labelBox = labelLine.boundingBox;
      FieldCandidate? bestCandidate;
      double? bestScore;

      for (final candidateLine in lines) {
        if (candidateLine == labelLine) continue;

        final candidateBox = candidateLine.boundingBox;
        final verticalDistance = (candidateBox.top - labelBox.top).abs();
        if (verticalDistance > maxVerticalDistance) continue;

        final amounts = extractAmountsFromText(candidateLine.text);
        if (amounts.isEmpty) continue;

        for (final amount in amounts) {
          final numericValue = parseEuropeanNumber(amount);
          if (numericValue == null) continue;

          final score = _scoreAmountCandidate(
            labelBox: labelBox,
            candidateBox: candidateBox,
            numericValue: numericValue,
            candidateText: candidateLine.text,
          );

          if (bestScore == null || score > bestScore) {
            bestScore = score;
            bestCandidate = FieldCandidate(
              value: amount,
              boundingBox: candidateBox,
              sourceText: candidateLine.text,
            );
          }
        }
      }

      if (bestCandidate != null) {
        return bestCandidate;
      }
    }

    return null;
  }

  double _scoreAmountCandidate({
    required Rect labelBox,
    required Rect candidateBox,
    required double numericValue,
    required String candidateText,
  }) {
    final labelCenterX = labelBox.left + (labelBox.width / 2);
    final labelCenterY = labelBox.top + (labelBox.height / 2);
    final candidateCenterX = candidateBox.left + (candidateBox.width / 2);
    final candidateCenterY = candidateBox.top + (candidateBox.height / 2);

    final dx = candidateCenterX - labelCenterX;
    final dy = candidateCenterY - labelCenterY;
    final absDx = dx.abs();
    final absDy = dy.abs();
    final normalizedText = normalize(candidateText);

    double score = 0.0;

    if (dx > 0) {
      score += 120.0;
    } else {
      score -= 60.0;
    }

    if (absDy < 18) {
      score += 180.0;
    } else if (absDy < 35) {
      score += 110.0;
    } else if (absDy < 60) {
      score += 50.0;
    } else {
      score -= absDy * 1.2;
    }

    score -= absDx * 0.08;

    if (numericValue >= 1.0) score += 10.0;
    if (normalizedText.contains('eur') || candidateText.contains('€')) {
      score += 60.0;
    }

    if (normalizedText.contains('gegeben') ||
        normalizedText.contains('mastercard') ||
        normalizedText.contains('visa') ||
        normalizedText.contains('kk') ||
        normalizedText.contains('mc') ||
        normalizedText.contains('bezahlt')) {
      score -= 120.0;
    }

    if (normalizedText.contains('mwst') ||
        normalizedText.contains('von') ||
        normalizedText.contains('=')) {
      score -= 80.0;
    }

    return score;
  }

  bool _isBadAmountLabelContext(
    String normalizedLine, {
    List<String> blockedPhrases = const [],
  }) {
    final allBlocked = <String>[
      'betrag dankend erhalten',
      'dankend erhalten',
      'mwst von',
      ...blockedPhrases,
    ];
    return allBlocked.any(normalizedLine.contains);
  }
}
