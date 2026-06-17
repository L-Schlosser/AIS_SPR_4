import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:path_provider/path_provider.dart';

class QwenExtractionResult {

  final Map<String, String> infos;

  /// Raw model response before JSON parsing
  final String rawResponse;

  /// Original OCR text
  final String rawText;

  /// Input document category
  final String category;

  QwenExtractionResult({
    required this.infos,
    required this.rawResponse,
    required this.rawText,
    required this.category,
  });

  Map<String, dynamic> toJson() => {
        'infos': infos,
        'raw_response': rawResponse,
        'raw_text': rawText,
        'category': category,
      };
}

class MLServiceQwen {
  late Llama _llama;

  bool _isInitialized = false;

  static const String _assetModelPath =
      'assets/models/qwen-q4.gguf';

  static const String _localModelName =
      'qwen-q4.gguf';

  // ===========================================================================
  // Initialization
  // ===========================================================================

  Future<void> initialize() async {
    if (_isInitialized) return;

    final modelPath = await _prepareModel();

    final modelParams = ModelParams();
    final contextParams = ContextParams()
      ..nPredict = -1
      ..nCtx = 4096
      ..nBatch = 512;
    final samplerParams = SamplerParams()
      ..temp = 0.2
      ..topP = 0.9
      ..topK = 40;

    _llama = Llama(
      modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplerParams: samplerParams,
      verbose: false,
    );

    _isInitialized = true;
  }

  bool get isInitialized => _isInitialized;

  void dispose() {
    _llama.dispose();
  }

  // ===========================================================================
  // Main extraction API
  // ===========================================================================

  Future<QwenExtractionResult> extract({
    required String text,
    required String category,
    int maxTokens = 256,
  }) async {
    if (!_isInitialized) {
      throw Exception(
        'Qwen not initialized - call initialize() first',
      );
    }

    final schemaDescription =
        _buildSchemaDescription(category);

    final prompt = '''
Du bist ein KI-System zur Extraktion von OCR-Dokumentdaten.

Die OCR-Texte sind teilweise fehlerhaft oder unvollständig.

Deine Aufgabe:
- Extrahiere wichtige Informationen aus dem Dokument
- Antworte NUR als valides JSON
- KEIN Markdown
- KEINE Erklärungen
- KEIN zusätzlicher Text
- Nur JSON zurückgeben

Dokumenttyp:
$category

Wichtige Standardfelder:
$schemaDescription

Regeln:
- Verwende deutsche snake_case Feldnamen
- Fehlende Felder NICHT hinzufügen
- Zusätzliche relevante Felder dürfen ergänzt werden
- Werte immer als String speichern
- Datumswerte möglichst original belassen
- Beträge möglichst exakt übernehmen

Beispiel:
{
  "firma": "Muster GmbH",
  "datum": "12.05.2025",
  "gesamtbetrag": "54,20 EUR"
}

OCR TEXT:
$text
''';

    // ------------------------------------------------------------------------
    // Run inference
    // ------------------------------------------------------------------------

    final rawResponse = _generateResponse(
      prompt,
      maxTokens,
    );

    // ------------------------------------------------------------------------
    // Parse JSON
    // ------------------------------------------------------------------------

    final infos = _parseJsonResponse(rawResponse);

    return QwenExtractionResult(
      infos: infos,
      rawResponse: rawResponse,
      rawText: text,
      category: category,
    );
  }

  String _generateResponse(
    String prompt,
    int maxTokens,
  ) {
    _llama.clear();
    _llama.setPrompt(prompt);

    final buffer = StringBuffer();
    var produced = 0;

    while (produced < maxTokens) {
      final (token, done) = _llama.getNext();
      if (done) break;
      buffer.write(token);
      produced++;
    }

    return buffer.toString().trim();
  }

  // ===========================================================================
  // Internal helpers
  // ===========================================================================

  Future<String> _prepareModel() async {
    final dir = await getApplicationDocumentsDirectory();

    final modelPath =
        '${dir.path}/$_localModelName';

    final modelFile = File(modelPath);

    // Copy model from assets only once
    if (!await modelFile.exists()) {
      final byteData =
          await rootBundle.load(_assetModelPath);

      final bytes =
          byteData.buffer.asUint8List();

      await modelFile.writeAsBytes(bytes);
    }

    return modelPath;
  }

  // ===========================================================================
  // JSON parsing
  // ===========================================================================

  Map<String, String> _parseJsonResponse(
    String response,
  ) {
    try {
      // Remove markdown code blocks if model adds them
      String cleaned = response.trim();

      cleaned = cleaned.replaceAll(
        '```json',
        '',
      );

      cleaned = cleaned.replaceAll(
        '```',
        '',
      );

      // Try finding JSON object
      final start = cleaned.indexOf('{');
      final end = cleaned.lastIndexOf('}');

      if (start >= 0 && end > start) {
        cleaned = cleaned.substring(
          start,
          end + 1,
        );
      }

      final decoded = jsonDecode(cleaned);

      if (decoded is! Map) {
        return {};
      }

      final result = <String, String>{};

      decoded.forEach((key, value) {
        if (key == null || value == null) return;

        final k = key.toString().trim();
        final v = value.toString().trim();

        if (k.isEmpty || v.isEmpty) return;

        result[k] = v;
      });

      return result;
    } catch (_) {
      return {};
    }
  }

  // ===========================================================================
  // Schema templates
  // ===========================================================================

  String _buildSchemaDescription(
    String category,
  ) {
    switch (category) {
      case 'invoice':
        return '''
- firma
- rechnungsnummer
- kundennummer
- datum
- faelligkeitsdatum
- gesamtbetrag
- mwst
- adresse
- iban
- bic
''';

      case 'receipt':
        return '''
- firma
- datum
- adresse
- gesamtbetrag
- mwst
- waehrung
- zahlungsart
''';

      case 'doctor_note':
        return '''
- arzt
- patient
- diagnose
- datum
- praxis
- adresse
''';

      case 'care_leave':
        return '''
- patient
- arzt
- beginn
- ende
- datum
- diagnose
''';

      case 'delivery_note':
        return '''
- firma
- lieferscheinnummer
- datum
- adresse
- empfaenger
- artikel
''';

      case 'master_data':
        return '''
- firma
- name
- adresse
- telefon
- email
- iban
- bic
- uid
''';

      default:
        return '''
- name
- datum
- firma
''';
    }
  }
}