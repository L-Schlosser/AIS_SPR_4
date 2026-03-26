import '../models/ocr_models.dart';

/// Result of the rule-based document classification.
class ClassificationResult {
  final String documentType;
  final double confidence;
  final Map<String, int> scores;

  const ClassificationResult({
    required this.documentType,
    required this.confidence,
    required this.scores,
  });

  Map<String, dynamic> toJson() {
    return {
      'document_type': documentType,
      'confidence': confidence,
      'scores': scores,
    };
  }
}

/// Simple rule-based classifier for the supported OCR document types.
///
/// This is intentionally lightweight for a first project version:
/// - it looks at the full OCR text
/// - it scores each document type by keyword hits
/// - it returns the best matching type plus a simple confidence value
class ClassificationService {
  static const String unknownType = 'unknown';

  /// Supported document types and their indicative keywords.
  static const Map<String, List<String>> _keywordsByType = {
    'invoice': [
      'rechnung',
      'invoice',
      'honorarnote',
      'honorar',
      'rechnungsnummer',
      'rechnungs-nr',
      'invoice number',
      'invoice no',
      'uid',
      'ust',
      'ust-id',
      'nettobetrag',
      'bruttobetrag',
      'gesamtbetrag',
      'zahlbar bis',
      'fällig',
      'mwst',
      'umsatzsteuer',
      'umsatzsteuerfrei',
      'ustg',
      'betrag',
      'summe',
      'gesamt',
      'patient',
      'diagnose',
    ],
    'receipt': [
      'kassenbeleg',
      'beleg',
      'bon',
      'summe',
      'gesamt',
      'total',
      'bar',
      'karte',
      'eur',
      'mwst',
      'zahlbetrag',
      'change',
      'cash',
    ],
    'caregiving_leave_confirmation': [
      'pflegefreistellung',
      'pflege',
      'bestätigung',
      'freistellung',
      'angehörige',
      'betreuung',
    ],
    'master_data_change': [
      'stammdaten',
      'meldezettel',
      'heiratsurkunde',
      'melderegister',
      'hauptwohnsitz',
      'nebenwohnsitz',
      'eheschließung',
      'familienstand',
      'adresse',
      'wohnadresse',
    ],
    'delivery_note': [
      'lieferschein',
      'lieferung',
      'lieferdatum',
      'lieferadresse',
      'empfänger',
      'versand',
      'liefernr',
      'liefernummer',
    ],
    'doctor_note': [
      'arzt',
      'ärztlich',
      'arztbestätigung',
      'ordination',
      'patient',
      'diagnose',
      'bestätigung',
      'krankenhaus',
      'ambulanz',
      'befund',
    ],
  };

  /// Optional score boosts if multiple keywords together strongly indicate a type.
  static const Map<String, List<List<String>>> _comboRules = {
    'invoice': [
      ['rechnung', 'uid'],
      ['rechnung', 'rechnungsnummer'],
      ['invoice', 'invoice number'],
      ['honorarnote', 'betrag'],
      ['honorarnote', 'datum'],
      ['umsatzsteuerfrei', 'ustg'],
      ['patient', 'diagnose'],
    ],
    'receipt': [
      ['kassenbeleg', 'summe'],
      ['beleg', 'eur'],
      ['total', 'cash'],
    ],
    'delivery_note': [
      ['lieferschein', 'lieferdatum'],
      ['lieferschein', 'empfänger'],
    ],
    'doctor_note': [
      ['arzt', 'patient'],
      ['diagnose', 'patient'],
      ['arztbestätigung', 'ordination'],
    ],
    'master_data_change': [
      ['meldezettel', 'hauptwohnsitz'],
      ['heiratsurkunde', 'familienstand'],
    ],
    'caregiving_leave_confirmation': [
      ['pflegefreistellung', 'bestätigung'],
      ['pflege', 'freistellung'],
    ],
  };

  ClassificationResult classifyDocument(OcrDocument document) {
    return classifyText(document.rawText);
  }

  ClassificationResult classifyLines(List<OcrLine> lines) {
    final text = lines.map((line) => line.text).join('\n');
    return classifyText(text);
  }

  ClassificationResult classifyText(String text) {
    final normalizedText = _normalize(text);
    final scores = <String, int>{};

    for (final entry in _keywordsByType.entries) {
      final type = entry.key;
      final keywords = entry.value;

      int score = 0;

      for (final keyword in keywords) {
        if (normalizedText.contains(_normalize(keyword))) {
          score += 1;
        }
      }

      final combos = _comboRules[type] ?? const [];
      for (final combo in combos) {
        final allMatch = combo.every(
          (keyword) => normalizedText.contains(_normalize(keyword)),
        );
        if (allMatch) {
          score += 2;
        }
      }

      scores[type] = score;
    }

    final sortedEntries = scores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    if (sortedEntries.isEmpty || sortedEntries.first.value <= 0) {
      return const ClassificationResult(
        documentType: unknownType,
        confidence: 0.0,
        scores: {},
      );
    }

    final best = sortedEntries.first;
    final secondBest = sortedEntries.length > 1 ? sortedEntries[1] : null;

    final confidence = _calculateConfidence(
      bestScore: best.value,
      secondBestScore: secondBest?.value ?? 0,
      totalScore: scores.values.fold<int>(0, (sum, value) => sum + value),
    );

    return ClassificationResult(
      documentType: best.key,
      confidence: confidence,
      scores: scores,
    );
  }

  String _normalize(String input) {
    return input
        .toLowerCase()
        .replaceAll('ä', 'ae')
        .replaceAll('ö', 'oe')
        .replaceAll('ü', 'ue')
        .replaceAll('ß', 'ss')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  double _calculateConfidence({
    required int bestScore,
    required int secondBestScore,
    required int totalScore,
  }) {
    if (bestScore <= 0 || totalScore <= 0) return 0.0;

    final dominance = bestScore / totalScore;
    final separation = bestScore == 0
        ? 0.0
        : (bestScore - secondBestScore) / bestScore;

    final confidence = ((dominance * 0.6) + (separation * 0.4)).clamp(0.0, 1.0);

    return double.parse(confidence.toStringAsFixed(2));
  }
}
