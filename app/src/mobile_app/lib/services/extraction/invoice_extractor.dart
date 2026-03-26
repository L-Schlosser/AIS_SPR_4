import 'dart:ui';

import '../../models/ocr_models.dart';
import '../../models/structured_document.dart';
import 'document_extractor.dart';

class InvoiceExtractor extends DocumentExtractor {
  const InvoiceExtractor();

  @override
  String get documentType => 'invoice';

  @override
  StructuredDocument extract({
    required OcrDocument ocrDocument,
    double? seedConfidence,
  }) {
    final fields = <ExtractedField>[];
    final lines = ocrDocument.allLines;

    final invoiceNumber = _findInvoiceNumber(lines);
    if (invoiceNumber != null) {
      fields.add(
        ExtractedField(
          key: 'invoice_number',
          value: invoiceNumber.value,
          confidence: 0.92,
          boundingBox: invoiceNumber.boundingBox,
          sourceText: invoiceNumber.sourceText,
        ),
      );
    }

    final issueDate = _findIssueDate(lines) ?? findFirstDate(lines);
    if (issueDate != null) {
      fields.add(
        ExtractedField(
          key: 'issue_date',
          value: issueDate.value,
          confidence: 0.88,
          boundingBox: issueDate.boundingBox,
          sourceText: issueDate.sourceText,
        ),
      );
    }

    final dueDate = _findDueDate(lines);
    if (dueDate != null) {
      fields.add(
        ExtractedField(
          key: 'due_date',
          value: dueDate.value,
          confidence: 0.90,
          boundingBox: dueDate.boundingBox,
          sourceText: dueDate.sourceText,
        ),
      );
    }

    final totalAmount = _findInvoiceTotal(lines) ?? findLargestAmount(lines);
    if (totalAmount != null) {
      fields.add(
        ExtractedField(
          key: 'total_amount',
          value: totalAmount.value,
          confidence: 0.90,
          boundingBox: totalAmount.boundingBox,
          sourceText: totalAmount.sourceText,
        ),
      );
    }

    final uidMatch = findUid(lines);
    if (uidMatch != null) {
      fields.add(
        ExtractedField(
          key: 'uid',
          value: uidMatch.value,
          confidence: 0.90,
          boundingBox: uidMatch.boundingBox,
          sourceText: uidMatch.sourceText,
        ),
      );
    }

    final issuer = _findIssuer(lines);
    if (issuer != null) {
      fields.add(
        ExtractedField(
          key: 'issuer',
          value: issuer.value,
          confidence: 0.60,
          boundingBox: issuer.boundingBox,
          sourceText: issuer.sourceText,
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
        fields.any((f) => f.key == 'invoice_number'),
        fields.any((f) => f.key == 'issue_date'),
        fields.any((f) => f.key == 'total_amount'),
      ].where((v) => v).length,
      weakFieldCount: [
        fields.any((f) => f.key == 'due_date'),
        fields.any((f) => f.key == 'uid'),
        fields.any((f) => f.key == 'issuer'),
        fields.any((f) => f.key == 'currency'),
      ].where((v) => v).length,
      seedConfidence: seedConfidence ?? 0.0,
    );

    return buildDocument(
      ocrDocument: ocrDocument,
      fields: fields,
      confidence: confidence,
    );
  }

  FieldCandidate? _findInvoiceNumber(List<OcrLine> lines) {
    return findLabeledValue(
      lines,
      labelPatterns: const [
        'rechnungsnummer',
        'rechnung nr',
        'rechnung nr.',
        'rechnungs-nr',
        'invoice number',
        'invoice no',
        'invoice #',
      ],
    );
  }

  FieldCandidate? _findIssueDate(List<OcrLine> lines) {
    return findLabeledValue(
      lines,
      labelPatterns: const [
        'rechnungsdatum',
        'issue date',
        'invoice date',
        'datum',
      ],
      preferDate: true,
    );
  }

  FieldCandidate? _findDueDate(List<OcrLine> lines) {
    return findLabeledValue(
      lines,
      labelPatterns: const [
        'faellig',
        'fällig',
        'zahlbar bis',
        'due date',
        'payment due',
      ],
      preferDate: true,
    );
  }

  FieldCandidate? _findIssuer(List<OcrLine> lines) {
    if (lines.isEmpty) return null;

    final firstLine = lines.first;
    if (firstLine.text.trim().isEmpty) return null;

    return FieldCandidate(
      value: firstLine.text.trim(),
      boundingBox: firstLine.boundingBox,
      sourceText: firstLine.text,
    );
  }

  FieldCandidate? _findInvoiceTotal(List<OcrLine> lines) {
    return _findLabeledAmount(
          lines,
          labelPatterns: const [
            'gesamt',
            'gesamtbetrag',
            'bruttobetrag',
            'summe',
            'endsumme',
            'total',
            'amount due',
            'zahlbetrag',
          ],
        ) ??
        _findLabeledAmount(lines, labelPatterns: const ['betrag', 'eur']);
  }

  FieldCandidate? _findLabeledAmount(
    List<OcrLine> lines, {
    required List<String> labelPatterns,
    List<String> blockedPhrases = const [],
    double maxVerticalDistance = 90,
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
      score += 110.0;
    } else {
      score -= 50.0;
    }

    if (absDy < 18) {
      score += 170.0;
    } else if (absDy < 35) {
      score += 100.0;
    } else if (absDy < 60) {
      score += 40.0;
    } else {
      score -= absDy * 1.0;
    }

    score -= absDx * 0.06;

    if (numericValue >= 1.0) {
      score += 12.0;
    }

    if (normalizedText.contains('eur') || candidateText.contains('€')) {
      score += 50.0;
    }

    if (normalizedText.contains('mwst') ||
        normalizedText.contains('ust') ||
        normalizedText.contains('vat') ||
        normalizedText.contains('netto') ||
        normalizedText.contains('=')) {
      score -= 70.0;
    }

    if (normalizedText.contains('bezahlt') ||
        normalizedText.contains('payment received') ||
        normalizedText.contains('paid')) {
      score -= 100.0;
    }

    return score;
  }

  bool _isBadAmountLabelContext(
    String normalizedLine, {
    List<String> blockedPhrases = const [],
  }) {
    final allBlocked = <String>[
      'mwst von',
      'ust von',
      'vat from',
      'payment received',
      ...blockedPhrases,
    ];
    return allBlocked.any(normalizedLine.contains);
  }
}
