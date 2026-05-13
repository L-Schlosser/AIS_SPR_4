import 'dart:convert';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class ExtractedDocumentData {
  final String documentType;
  final Map<String, dynamic> data;
  final List<String> extractionErrors;

  const ExtractedDocumentData({
    required this.documentType,
    required this.data,
    this.extractionErrors = const [],
  });

  bool get isValid => extractionErrors.isEmpty;

  String get summary {
    return '';
  }

  Map<String, dynamic> get keyFacts {
    final filtered = Map<String, dynamic>.from(data)
      ..remove('document_type')
      ..remove('language')
      ..remove('summary')
      ..remove('confidence')
      ..remove('additional_fields')
      ..remove('extracted_data')
      ..remove('raw_text');

    final extractedData = data['extracted_data'];
    if (extractedData is Map<String, dynamic>) {
      return Map<String, dynamic>.from(extractedData);
    }

    return filtered;
  }

  Map<String, dynamic> toJson() => {
        'documentType': documentType,
        'data': data,
        'extractionErrors': extractionErrors,
      };

  static ExtractedDocumentData fromJson(Map<String, dynamic> json) {
    return ExtractedDocumentData(
      documentType: (json['documentType'] ?? 'unknown').toString(),
      data: Map<String, dynamic>.from(json['data'] as Map? ?? const {}),
      extractionErrors: List<String>.from(json['extractionErrors'] as List? ?? const []),
    );
  }
}

class ExtractionSchemas {
  static String getRouterPrompt(String documentType, String ocrText) {
    final type = documentType.trim().isEmpty ? 'unbekannt' : documentType.trim();
    // Keep the prompt intentionally short to reduce "rule echoing" by smaller seq2seq models.
    return '''
Extrahiere aus OCR-Text die wichtigen Informationen als FLACHES JSON-Objekt.
Dokumenttyp: $type

Vorgaben:
- Nur echte Informationen aus dem OCR-Text (nichts erfinden).
- Keine Metadatenfelder (keine confidence/summary/language/raw_text/document_type).
- Zahlen als Zahlen; Datum "YYYY-MM-DD"; Arrays bleiben Arrays; null nur bei unsicher erkanntem Wert.

OCR:
$ocrText

Antworte JETZT NUR mit dem JSON-Objekt.
''';
  }

  static String getKeyValueFallbackPrompt(String documentType, String ocrText) {
    final type = documentType.trim().isEmpty ? 'unbekannt' : documentType.trim();
    return '''
Extrahiere aus OCR-Text die wichtigen Informationen.
Dokumenttyp: $type

Gib die Antwort als mehrere Zeilen im Format "schluessel = wert".
- Wiederhole NICHT die Vorgaben/Regeln und NICHT den OCR-Text.
- KEINE Aufzählungszeichen, KEINE Erklärungen, KEINE Sätze.
- Nur echte Informationen aus dem OCR-Text (nichts erfinden).
- Zahlen als Zahlen; Datum "YYYY-MM-DD"; Listen als komma-getrennte Werte (ohne Klammern/Anführungszeichen).

OCR:
$ocrText
''';
  }
}

class MlServiceExtractor {
  OrtSession? _encoder;
  OrtSession? _decoder;
  OrtSession? _decoderWithPast;
  _UnigramSpTokenizer? _tokenizer;
  bool _isInitialized = false;

  Future<void> init() async {
    if (_isInitialized) return;

    final encoderBytes = await _loadAssetBytes([
      'assets/transformer/encoder_model_int8.onnx',
      'assets/models/transformer/encoder_model_int8.onnx',
    ]);
    final decoderBytes = await _loadAssetBytes([
      'assets/transformer/decoder_model_int8.onnx',
      'assets/models/transformer/decoder_model_int8.onnx',
    ]);
    final decoderWithPastBytes = await _loadAssetBytes([
      'assets/transformer/decoder_with_past_model_int8.onnx',
      'assets/models/transformer/decoder_with_past_model_int8.onnx',
    ]);
    final tokenizerJson = await _loadAssetString([
      'assets/transformer/tokenizer.json',
      'assets/models/transformer/tokenizer.json',
    ]);
    final tokenizerConfigJson = await _loadAssetString([
      'assets/transformer/tokenizer_config.json',
      'assets/models/transformer/tokenizer_config.json',
    ]);

    _tokenizer = _UnigramSpTokenizer.fromTokenizerJson(
      tokenizerJson: tokenizerJson,
      tokenizerConfigJson: tokenizerConfigJson,
    );

    final sessionOptions = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(1);

    try {
      sessionOptions.appendXnnpackProvider();
    } catch (_) {}

    _encoder = OrtSession.fromBuffer(encoderBytes, sessionOptions);
    _decoder = OrtSession.fromBuffer(decoderBytes, sessionOptions);
    _decoderWithPast = OrtSession.fromBuffer(decoderWithPastBytes, sessionOptions);

    _isInitialized = true;
  }

  Future<ExtractedDocumentData> extract({
    required String ocrText,
    required String documentType,
    int maxInputTokens = 512,
    int maxNewTokens = 256,
  }) async {
    if (!_isInitialized) {
      throw Exception('MlServiceExtractor not initialized. Call init() first.');
    }

    try {
      final cleanedOcr = _compactOcrText(ocrText, maxChars: 5000);

      // Attempt 1: JSON prompt
      final prompt1 = ExtractionSchemas.getRouterPrompt(documentType, cleanedOcr);
      final out1 = await _runInference(
        prompt: prompt1,
        maxInputTokens: maxInputTokens,
        maxNewTokens: maxNewTokens,
      );
      print("Model output: $out1");

      var parsed = _parseJsonFromOutput(out1);

      // Attempt 2: if the model is clearly echoing rules (no structure extracted),
      // retry with a key:value format that we can reliably parse into a map.
      if (parsed == null || parsed.isEmpty) {
        final looksLikeRuleEcho = _looksLikeInstructionEcho(out1);
        if (looksLikeRuleEcho) {
          final prompt2 = ExtractionSchemas.getKeyValueFallbackPrompt(
            documentType,
            cleanedOcr,
          );
          final out2 = await _runInference(
            prompt: prompt2,
            maxInputTokens: maxInputTokens,
            maxNewTokens: maxNewTokens,
          );
          print("Model output (retry): $out2");
          parsed = _parseJsonFromOutput(out2);
        }
      }

      // Final guard: never return instruction/rule text as extracted data.
      if (parsed != null && parsed.isNotEmpty && _mapLooksLikeInstructions(parsed)) {
        parsed = const <String, dynamic>{};
      }

      if (parsed == null || parsed.isEmpty) {
        final fallback = _genericFeatureExtract(cleanedOcr, documentType: documentType);
        return ExtractedDocumentData(
          documentType: documentType,
          data: fallback,
          extractionErrors: const [
            'Model output contained no structured fields; used generic OCR feature extraction fallback.'
          ],
        );
      }

      return ExtractedDocumentData(
        documentType: documentType,
        data: parsed,
      );
    } catch (e) {
      return ExtractedDocumentData(documentType: documentType, data: <String, dynamic>{}, extractionErrors: ['Extraction error: $e']);
    }
  }

  Future<String> _runInference({
    required String prompt,
    required int maxInputTokens,
    required int maxNewTokens,
  }) async {
    final tokenizer = _tokenizer!;
    final encoder = _encoder!;
    final decoder = _decoder!;
    final decoderWithPast = _decoderWithPast!;

    final inputIds = tokenizer.encode(prompt, maxLength: maxInputTokens);

    final encoderInputs = <String, OrtValue>{};
    final encoderInputIdsName =
        _pickName(encoder.inputNames, ['input_ids', 'inputIds', 'ids']) ??
            encoder.inputNames.first;

    encoderInputs[encoderInputIdsName] = OrtValueTensor.createTensorWithDataList(
      inputIds,
      [1, inputIds.length],
    );

    final encoderAttnName = _pickName(
      encoder.inputNames,
      ['attention_mask', 'attentionMask', 'mask'],
    );
    if (encoderAttnName != null) {
      final attnMask = List<int>.filled(inputIds.length, 1, growable: false);
      encoderInputs[encoderAttnName] = OrtValueTensor.createTensorWithDataList(
        attnMask,
        [1, attnMask.length],
      );
    }

    final runOptions = OrtRunOptions();
    final encoderOutputs = await encoder.runAsync(runOptions, encoderInputs);
    if (encoderOutputs == null || encoderOutputs.isEmpty) {
      throw Exception('Encoder produced no outputs.');
    }

    final encoderHiddenStates = encoderOutputs.first!;
    final generated = <int>[];
    final decoderStartTokenId = tokenizer.decoderStartTokenId;
    var nextInputId = decoderStartTokenId;
    final pastCache = <String, OrtValue>{};

    // Prevent outputting special tokens that break parseability.
    final bannedTokenIds = <int>{
      tokenizer.unkId,
      tokenizer.padTokenId,
      tokenizer.decoderStartTokenId,
    };

    for (var step = 0; step < maxNewTokens; step++) {
      final isFirst = step == 0;
      final session = isFirst ? decoder : decoderWithPast;
      final inputs = <String, OrtValue>{};

      final inputIdsName =
          _pickName(session.inputNames, ['input_ids', 'inputIds', 'ids']) ??
              session.inputNames.first;
      inputs[inputIdsName] = OrtValueTensor.createTensorWithDataList(
        [nextInputId],
        [1, 1],
      );

      final encMaskName = _pickName(
            session.inputNames,
            ['encoder_attention_mask', 'encoderAttentionMask'],
          ) ??
          (session.inputNames.contains('encoder_attention_mask')
              ? 'encoder_attention_mask'
              : null);
      if (encMaskName != null && encoderAttnName != null) {
        inputs[encMaskName] = encoderInputs[encoderAttnName]!;
      }

      if (isFirst) {
        final encHsName = _pickName(
              session.inputNames,
              ['encoder_hidden_states', 'encoderHiddenStates', 'encoder_outputs'],
            ) ??
            (session.inputNames.contains('encoder_hidden_states')
                ? 'encoder_hidden_states'
                : null);
        if (encHsName != null) {
          inputs[encHsName] = encoderHiddenStates;
        }
      } else {
        for (final name in session.inputNames) {
          if (!name.startsWith('past_key_values.')) continue;
          final v = pastCache[name];
          if (v != null) {
            inputs[name] = v;
          }
        }
      }

      final outputs = await session.runAsync(runOptions, inputs);
      if (outputs == null || outputs.isEmpty || outputs.first == null) {
        throw Exception('Decoder produced no outputs.');
      }

      final logits = outputs.first!.value;
      final nextTokenId = _argmaxLastLogits(logits, bannedIds: bannedTokenIds);

      final outputNames = session.outputNames;
      for (var i = 1; i < min(outputNames.length, outputs.length); i++) {
        final oName = outputNames[i];
        final oVal = outputs[i];
        if (oVal == null) continue;
        if (!oName.startsWith('present.')) continue;
        final cacheKey = oName.replaceFirst('present.', 'past_key_values.');
        pastCache[cacheKey] = oVal;
      }

      if (nextTokenId == tokenizer.eosTokenId) {
        break;
      }

      generated.add(nextTokenId);
      nextInputId = nextTokenId;
    }

    return tokenizer.decode(generated);
  }

  static Map<String, dynamic>? _parseJsonFromOutput(String output) {
    var trimmed = output.trim();

    // Strip common markdown fences (even though we ask for none).
    if (trimmed.startsWith('```')) {
      trimmed = trimmed
          .replaceFirst(RegExp(r'^```(?:json)?\s*', caseSensitive: false), '')
          .replaceFirst(RegExp(r'\s*```$'), '')
          .trim();
    }

    try {
      final parsed = jsonDecode(trimmed);
      if (parsed is Map<String, dynamic>) {
        return parsed;
      }
    } catch (_) {}

    final obj = _extractFirstJsonObject(trimmed);
    if (obj != null) {
      try {
        final parsed = jsonDecode(obj);
        if (parsed is Map<String, dynamic>) {
          return parsed;
        }
      } catch (_) {}
    }

    try {
      final arrayStr = _extractFirstJsonArray(trimmed);
      if (arrayStr != null) {
        final parsed = jsonDecode(arrayStr);
        if (parsed is List) {
          return {'items': parsed};
        }
      }
    } catch (_) {}

    // Fallback: if the model cannot emit JSON braces reliably (e.g. emits <unk>),
    // accept a simple "key: value" (or "- key: value") format and convert it to a map.
    final kv = _parseKeyValueLines(trimmed);
    if (kv.isNotEmpty) return kv;

    return null;
  }

  bool get isInitialized => _isInitialized;

  void dispose() {
    _encoder?.release();
    _decoder?.release();
    _decoderWithPast?.release();
    _encoder = null;
    _decoder = null;
    _decoderWithPast = null;
    _tokenizer = null;
    _isInitialized = false;
  }
}

bool _looksLikeInstructionEcho(String s) {
  final t = s.toLowerCase();
  // Heuristic: model repeats prompt constraints instead of extracting.
  // This stays language-agnostic-ish but catches the common German outputs you showed.
  final badPhrases = <String>[
    'ausgabe',
    'muss',
    'genau',
    'json-objekt',
    'regeln',
    'keine erklärungen',
    'null nur',
    'arrays bleiben arrays',
    'yyyy-mm-dd',
    'vorgaben',
    'nur echte informationen',
    'keine metadaten',
    'ocr-text',
  ];
  var hits = 0;
  for (final p in badPhrases) {
    if (t.contains(p)) hits++;
  }

  // Estimate if there's any document-like signal (digits, currency, IBAN, dates, etc).
  final hasDocSignal = RegExp(r'(\d{2,})|(\biban\b)|(\beur\b)|[€$]|(\bAT\d{2}\b)')
      .hasMatch(s);

  // If it contains many rule-phrases and little/no document signal, treat as echo
  // even if separators like ":" appear (e.g. "Formular: invoice Vorgaben: ...").
  if (hits >= 3 && !hasDocSignal) return true;

  // Otherwise: echo if it has multiple rule-phrases and no JSON braces.
  final hasJsonBraces = s.contains('{') && s.contains('}');
  return hits >= 2 && !hasJsonBraces && !hasDocSignal;
}

String _compactOcrText(String input, {required int maxChars}) {
  // Goal: de-noise OCR without extracting fields. This only reduces repetition/length.
  var s = input.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
  s = s.replaceAll(RegExp(r'[ \t]+'), ' ');
  s = s.replaceAll(RegExp(r'\n{3,}'), '\n\n').trim();

  // Collapse long runs of identical tokens (e.g. "09 09 09 ...").
  final tokens = s.split(RegExp(r'\s+'));
  final out = <String>[];
  String? prev;
  var run = 0;
  const maxRun = 6;
  for (final t in tokens) {
    if (t == prev) {
      run++;
      if (run <= maxRun) out.add(t);
    } else {
      prev = t;
      run = 1;
      out.add(t);
    }
  }
  s = out.join(' ');

  if (s.length <= maxChars) return s;
  // Keep head+tail so receipts with footer data still survive truncation.
  final head = s.substring(0, (maxChars * 0.65).floor());
  final tail = s.substring(s.length - (maxChars * 0.35).floor());
  return '$head\n...\n$tail';
}

String? _extractFirstJsonObject(String s) {
  final start = s.indexOf('{');
  if (start < 0) return null;

  var depth = 0;
  var inStr = false;
  var escape = false;
  for (var i = start; i < s.length; i++) {
    final ch = s[i];

    if (inStr) {
      if (escape) {
        escape = false;
        continue;
      }
      if (ch == r'\') {
        escape = true;
        continue;
      }
      if (ch == '"') {
        inStr = false;
      }
      continue;
    }

    if (ch == '"') {
      inStr = true;
      continue;
    }

    if (ch == '{') depth++;
    if (ch == '}') {
      depth--;
      if (depth == 0) {
        return s.substring(start, i + 1);
      }
    }
  }

  return null;
}

String? _extractFirstJsonArray(String s) {
  final start = s.indexOf('[');
  if (start < 0) return null;

  var depth = 0;
  var inStr = false;
  var escape = false;
  for (var i = start; i < s.length; i++) {
    final ch = s[i];

    if (inStr) {
      if (escape) {
        escape = false;
        continue;
      }
      if (ch == r'\') {
        escape = true;
        continue;
      }
      if (ch == '"') {
        inStr = false;
      }
      continue;
    }

    if (ch == '"') {
      inStr = true;
      continue;
    }

    if (ch == '[') depth++;
    if (ch == ']') {
      depth--;
      if (depth == 0) {
        return s.substring(start, i + 1);
      }
    }
  }

  return null;
}

String? _pickName(List<String> names, List<String> preferred) {
  for (final p in preferred) {
    final match = names.firstWhere(
      (n) => n == p || n.toLowerCase() == p.toLowerCase(),
      orElse: () => '',
    );
    if (match.isNotEmpty) return match;
  }
  return null;
}

int _argmaxLastLogits(Object? logits, {Set<int> bannedIds = const {}}) {
  if (logits is List) {
    dynamic cur = logits;
    while (cur is List && cur.isNotEmpty && cur.first is List) {
      cur = cur.last;
    }
    if (cur is List) {
      final lastVec = cur;
      var bestIdx = 0;
      var bestVal = double.negativeInfinity;
      for (var i = 0; i < lastVec.length; i++) {
        if (bannedIds.contains(i)) continue;
        final v = (lastVec[i] as num).toDouble();
        if (v > bestVal) {
          bestVal = v;
          bestIdx = i;
        }
      }
      return bestIdx;
    }
  }
  throw Exception('Unsupported logits output structure: ${logits.runtimeType}');
}

Map<String, dynamic> _parseKeyValueLines(String s) {
  // Accept lines like:
  // - key: value
  // key: value
  // key = value
  // and turn them into a flat JSON-like map.
  // This is NOT field-regex extraction; it’s just parsing a structured format.
  final normalized = s.replaceAll('\r\n', '\n').replaceAll('\r', '\n');

  // Models often output many "- ..." fragments on one line.
  final splitByBullets = normalized
      .replaceAll(' - ', '\n')
      .replaceAll(' – ', '\n')
      .replaceAll(' — ', '\n');

  final lines = splitByBullets
      .split('\n')
      .map((l) => l.trim())
      .where((l) => l.isNotEmpty)
      .toList(growable: false);

  final out = <String, dynamic>{};
  for (var raw in lines) {
    var line = raw;
    if (line.startsWith('- ')) line = line.substring(2).trim();
    if (line.startsWith('• ')) line = line.substring(2).trim();

    if (_lineLooksLikeInstruction(line)) continue;

    final sepIdx = _firstSeparatorIndex(line);
    if (sepIdx < 0) continue;

    final key = line.substring(0, sepIdx).trim();
    var value = line.substring(sepIdx + 1).trim();
    if (key.isEmpty || value.isEmpty) continue;

    if (_lineLooksLikeInstruction('$key $value')) continue;

    // Strip trailing punctuation that models often add.
    value = value.replaceAll(RegExp(r'[;,.]+$'), '').trim();

    // Try to coerce numbers/bools/null; otherwise keep string.
    final coerced = _coerceScalar(value);
    out[key] = coerced;
  }
  return out;
}

bool _lineLooksLikeInstruction(String line) {
  final t = line.toLowerCase();
  // Reject anything that looks like copied prompt rules instead of document data.
  const needles = <String>[
    'nur echte informationen',
    'nichts erfinden',
    'keine metadaten',
    'keine erklärungen',
    'arrays bleiben arrays',
    'null nur',
    'yyyy-mm-dd',
    'vorgaben',
    'ausgabe',
    'json-objekt',
    'ocr-text',
  ];
  for (final n in needles) {
    if (t.contains(n)) return true;
  }
  return false;
}

bool _mapLooksLikeInstructions(Map<String, dynamic> m) {
  if (m.isEmpty) return false;
  var bad = 0;
  var total = 0;
  for (final e in m.entries) {
    total++;
    final joined = '${e.key} ${e.value}'.toString();
    if (_lineLooksLikeInstruction(joined)) bad++;
  }
  // If most fields look like rules, discard.
  return total > 0 && bad / total >= 0.6;
}

Map<String, dynamic> _genericFeatureExtract(String ocrText, {required String documentType}) {
  // Generic, document-agnostic features (no hardcoded invoice fields),
  // with light document-type-specific tuning.
  final text = ocrText;
  final dt = documentType.toLowerCase();
  final isMedical = dt.contains('krank') || dt.contains('pflege') || dt.contains('arzt');

  if (isMedical) {
    return _genericMedicalExtract(text);
  }

  final dates = <String>{};
  for (final m in RegExp(r'\b(\d{4})[-/.](\d{2})[-/.](\d{2})\b').allMatches(text)) {
    dates.add('${m.group(1)}-${m.group(2)}-${m.group(3)}');
  }
  for (final m in RegExp(r'\b(\d{2})[./](\d{2})[./](\d{4})\b').allMatches(text)) {
    // Convert DD/MM/YYYY -> YYYY-MM-DD (best-effort).
    dates.add('${m.group(3)}-${m.group(2)}-${m.group(1)}');
  }

  final amounts = <double>{};
  for (final m in RegExp(
    r'(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))(?=\s*[€$]|(?:\s|\b)(?:EUR|USD|CHF)\b)?',
    caseSensitive: false,
  ).allMatches(text)) {
    final raw = m.group(1) ?? '';
    final normalized = raw.replaceAll('.', '').replaceAll(',', '.');
    final v = double.tryParse(normalized);
    if (v != null) amounts.add(v);
  }

  final currencies = <String>{};
  if (RegExp(r'€|\bEUR\b', caseSensitive: false).hasMatch(text)) currencies.add('EUR');
  if (RegExp(r'\$|\bUSD\b', caseSensitive: false).hasMatch(text)) currencies.add('USD');
  if (RegExp(r'\bCHF\b', caseSensitive: false).hasMatch(text)) currencies.add('CHF');

  final emails = <String>{};
  for (final m in RegExp(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', caseSensitive: false)
      .allMatches(text)) {
    emails.add(m.group(0)!);
  }

  final phones = <String>{};
  for (final m in RegExp(r'(\+?\d[\d ()-]{7,}\d)').allMatches(text)) {
    final p = m.group(1)!.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (p.length >= 8) phones.add(p);
  }

  final ibans = <String>{};
  for (final m in RegExp(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b').allMatches(text)) {
    ibans.add(m.group(0)!);
  }

  final identifiers = <String>{};
  for (final m in RegExp(r'\b[A-Z]{2,}-\d{2,}\b').allMatches(text)) {
    identifiers.add(m.group(0)!);
  }

  return <String, dynamic>{
    if (dates.isNotEmpty) 'dates': dates.toList()..sort(),
    if (amounts.isNotEmpty) 'amounts': amounts.toList()..sort(),
    if (currencies.isNotEmpty) 'currencies': currencies.toList()..sort(),
    if (emails.isNotEmpty) 'emails': emails.toList()..sort(),
    if (phones.isNotEmpty) 'phones': phones.toList()..sort(),
    if (ibans.isNotEmpty) 'ibans': ibans.toList()..sort(),
    if (identifiers.isNotEmpty) 'identifiers': identifiers.toList()..sort(),
  };
}

Map<String, dynamic> _genericMedicalExtract(String text) {
  final dates = <String>{};
  for (final m in RegExp(r'\b(\d{4})[-/.](\d{2})[-/.](\d{2})\b').allMatches(text)) {
    dates.add('${m.group(1)}-${m.group(2)}-${m.group(3)}');
  }
  for (final m in RegExp(r'\b(\d{2})[./](\d{2})[./](\d{4})\b').allMatches(text)) {
    dates.add('${m.group(3)}-${m.group(2)}-${m.group(1)}');
  }

  // AU / Zeitraum "von ... bis ..."
  String? periodStart;
  String? periodEnd;
  final range = RegExp(
    r'\b(von|ab)\s+(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})\s+(bis|bzw\.|und)\s+(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})\b',
    caseSensitive: false,
  ).firstMatch(text);
  if (range != null) {
    periodStart = _normalizeDate(range.group(2));
    periodEnd = _normalizeDate(range.group(4));
  }

  // Issue date near "Datum"
  String? issueDate;
  final issue = RegExp(
    r'\bdatum\b[^0-9]{0,10}(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})',
    caseSensitive: false,
  ).firstMatch(text);
  if (issue != null) {
    issueDate = _normalizeDate(issue.group(1));
  }

  // SVNR / Versicherungsnummer (AT often 10 digits, sometimes spaced)
  final svnr = <String>{};
  for (final m in RegExp(r'\b(SVNR|SV-NR|Sozialversicherungsnummer)\b[^0-9]{0,20}([0-9][0-9 ]{8,14}[0-9])',
          caseSensitive: false)
      .allMatches(text)) {
    final raw = (m.group(2) ?? '').replaceAll(' ', '');
    if (raw.length >= 9 && raw.length <= 12) svnr.add(raw);
  }

  // Doctor name heuristics
  String? doctor;
  final doc = RegExp(r'\b(Dr\.?\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+){0,3})\b')
      .firstMatch(text);
  if (doc != null) doctor = doc.group(1)?.trim();

  // Patient name heuristics
  String? patient;
  final pat = RegExp(r'\b(Name|Patient|Versicherte[rn]?)\b\s*[:\-]?\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+){1,3})\b',
          caseSensitive: false)
      .firstMatch(text);
  if (pat != null) patient = pat.group(2)?.trim();

  // Contacts
  final phones = <String>{};
  for (final m in RegExp(r'(\+?\d[\d ()-]{7,}\d)').allMatches(text)) {
    final p = m.group(1)!.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (p.length >= 8) phones.add(p);
  }
  final emails = <String>{};
  for (final m in RegExp(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', caseSensitive: false).allMatches(text)) {
    emails.add(m.group(0)!);
  }

  // Address hint: Austrian ZIP + city (4 digits + word)
  String? location;
  final loc = RegExp(r'\b(\d{4})\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]{2,})\b').firstMatch(text);
  if (loc != null) location = '${loc.group(1)} ${loc.group(2)}';

  return <String, dynamic>{
    if (periodStart != null) 'period_start': periodStart,
    if (periodEnd != null) 'period_end': periodEnd,
    if (issueDate != null) 'issue_date': issueDate,
    if (dates.isNotEmpty) 'dates': dates.toList()..sort(),
    if (doctor != null) 'doctor': doctor,
    if (patient != null) 'patient': patient,
    if (svnr.isNotEmpty) 'svnr': svnr.toList()..sort(),
    if (phones.isNotEmpty) 'phones': phones.toList()..sort(),
    if (emails.isNotEmpty) 'emails': emails.toList()..sort(),
    if (location != null) 'location_hint': location,
  };
}

String? _normalizeDate(String? s) {
  if (s == null) return null;
  final v = s.trim();
  final iso = RegExp(r'^(\d{4})[-/.](\d{2})[-/.](\d{2})$').firstMatch(v);
  if (iso != null) return '${iso.group(1)}-${iso.group(2)}-${iso.group(3)}';
  final dmy = RegExp(r'^(\d{2})[./-](\d{2})[./-](\d{4})$').firstMatch(v);
  if (dmy != null) return '${dmy.group(3)}-${dmy.group(2)}-${dmy.group(1)}';
  return null;
}

int _firstSeparatorIndex(String line) {
  // Prefer ":"; fall back to "=".
  final c = line.indexOf(':');
  if (c >= 0) return c;
  final e = line.indexOf('=');
  if (e >= 0) return e;
  return -1;
}

dynamic _coerceScalar(String v) {
  final lower = v.toLowerCase();
  if (lower == 'null') return null;
  if (lower == 'true') return true;
  if (lower == 'false') return false;

  // Normalize decimal comma to dot when it looks numeric.
  final normalized = v.replaceAll(RegExp(r'(?<=\d),(?=\d)'), '.');
  final asInt = int.tryParse(normalized);
  if (asInt != null) return asInt;
  final asDouble = double.tryParse(normalized);
  if (asDouble != null) return asDouble;

  // Remove surrounding quotes if present.
  if ((v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'"))) {
    return v.substring(1, v.length - 1);
  }
  return v;
}

Future<Uint8List> _loadAssetBytes(List<String> candidates) async {
  Object? lastError;
  for (final path in candidates) {
    try {
      final data = await rootBundle.load(path);
      return data.buffer.asUint8List();
    } catch (e) {
      lastError = e;
    }
  }
  throw Exception('Failed to load asset bytes. Tried: $candidates. Last: $lastError');
}

Future<String> _loadAssetString(List<String> candidates) async {
  Object? lastError;
  for (final path in candidates) {
    try {
      return await rootBundle.loadString(path);
    } catch (e) {
      lastError = e;
    }
  }
  throw Exception('Failed to load asset string. Tried: $candidates. Last: $lastError');
}

class _UnigramSpTokenizer {
  final Map<String, _Piece> _pieces;
  final _TrieNode _trie;
  final int unkId;
  final int eosTokenId;
  final int padTokenId;
  final int decoderStartTokenId;
  final String metaspace;
  final int _maxPieceLen;

  _UnigramSpTokenizer._({
    required Map<String, _Piece> pieces,
    required this.unkId,
    required this.eosTokenId,
    required this.padTokenId,
    required this.decoderStartTokenId,
    required this.metaspace,
  })  : _pieces = pieces,
        _trie = _TrieNode.build(pieces.keys),
        _maxPieceLen = pieces.keys.fold<int>(1, (m, s) => max(m, s.length));

  factory _UnigramSpTokenizer.fromTokenizerJson({
    required String tokenizerJson,
    required String tokenizerConfigJson,
  }) {
    final tok = jsonDecode(tokenizerJson) as Map<String, dynamic>;
    final model = tok['model'] as Map<String, dynamic>;
    final type = (model['type'] ?? '').toString();
    if (type != 'Unigram') {
      throw Exception('Unsupported tokenizer model type: $type (expected Unigram)');
    }

    final decoder = tok['decoder'] as Map<String, dynamic>;
    final metaspace = (decoder['replacement'] ?? '▁').toString();

    final vocab = model['vocab'] as List<dynamic>;
    final pieces = <String, _Piece>{};
    for (var i = 0; i < vocab.length; i++) {
      final entry = vocab[i] as List<dynamic>;
      final piece = entry[0].toString();
      final score = (entry[1] as num).toDouble();
      pieces[piece] = _Piece(id: i, score: score);
    }

    final cfg = jsonDecode(tokenizerConfigJson) as Map<String, dynamic>;
    final padToken = (cfg['pad_token'] ?? '<pad>').toString();
    final eosToken = (cfg['eos_token'] ?? '</s>').toString();

    final padId = pieces[padToken]?.id ?? 0;
    final eosId = pieces[eosToken]?.id ?? 1;
    final unkId = (model['unk_id'] as num?)?.toInt() ?? (pieces['<unk>']?.id ?? 2);

    // T5-style exports typically use pad as decoder_start_token_id.
    final decoderStartTokenId = padId;

    return _UnigramSpTokenizer._(
      pieces: pieces,
      unkId: unkId,
      eosTokenId: eosId,
      padTokenId: padId,
      decoderStartTokenId: decoderStartTokenId,
      metaspace: metaspace,
    );
  }

  List<int> encode(String text, {required int maxLength}) {
    final normalized = _basicNormalize(text);
    final metaspaceText = _applyMetaspace(normalized);

    final ids = _viterbiEncode(metaspaceText);
    if (ids.length <= maxLength) return ids;
    return ids.sublist(0, maxLength);
  }

  String decode(List<int> ids) {
    final sb = StringBuffer();
    for (final id in ids) {
      if (id == eosTokenId || id == padTokenId) continue;
      final piece = _pieceForId(id);
      if (piece == null) continue;
      sb.write(piece);
    }
    var out = sb.toString();
    out = out.replaceAll(metaspace, ' ');
    out = out.trimLeft();
    return out;
  }

  String? _pieceForId(int id) {
    _idToPiece ??= List<String?>.filled(_pieces.length, null);
    final cache = _idToPiece!;
    if (id >= 0 && id < cache.length && cache[id] != null) return cache[id];
    for (final e in _pieces.entries) {
      final pid = e.value.id;
      if (pid >= 0 && pid < cache.length && cache[pid] == null) {
        cache[pid] = e.key;
      }
    }
    if (id >= 0 && id < cache.length) return cache[id];
    return null;
  }

  List<String?>? _idToPiece;

  String _basicNormalize(String s) {
    return s.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
  }

  String _applyMetaspace(String s) {
    final collapsed = s.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (collapsed.isEmpty) return metaspace;
    final replaced = collapsed.replaceAll(' ', metaspace);
    return replaced.startsWith(metaspace) ? replaced : '$metaspace$replaced';
  }

  List<int> _viterbiEncode(String s) {
    final n = s.length;
    final bestScore = List<double>.filled(n + 1, double.negativeInfinity);
    final bestPrev = List<int>.filled(n + 1, -1);
    final bestPiece = List<String?>.filled(n + 1, null);

    bestScore[0] = 0.0;

    for (var i = 0; i < n; i++) {
      if (bestScore[i].isInfinite) continue;

      var node = _trie;
      for (var j = i; j < min(n, i + _maxPieceLen); j++) {
        final ch = s[j];
        node = node.children[ch] ?? _TrieNode.empty;
        if (identical(node, _TrieNode.empty)) break;

        if (node.isTerminal) {
          final piece = s.substring(i, j + 1);
          final p = _pieces[piece];
          if (p != null) {
            final score = bestScore[i] + p.score;
            final k = j + 1;
            if (score > bestScore[k]) {
              bestScore[k] = score;
              bestPrev[k] = i;
              bestPiece[k] = piece;
            }
          }
        }
      }

      final k = i + 1;
      final score = bestScore[i] - 100.0;
      if (score > bestScore[k]) {
        bestScore[k] = score;
        bestPrev[k] = i;
        bestPiece[k] = null;
      }
    }

    final out = <int>[];
    var idx = n;
    while (idx > 0) {
      final prev = bestPrev[idx];
      if (prev < 0) break;
      final piece = bestPiece[idx];
      out.add(piece == null ? unkId : _pieces[piece]!.id);
      idx = prev;
    }
    return out.reversed.toList(growable: false);
  }
}

class _Piece {
  final int id;
  final double score;

  const _Piece({required this.id, required this.score});
}

class _TrieNode {
  final Map<String, _TrieNode> children;
  final bool isTerminal;

  const _TrieNode({required this.children, required this.isTerminal});

  static const empty = _TrieNode(children: {}, isTerminal: false);

  static _TrieNode build(Iterable<String> pieces) {
    final root = _MutableTrieNode();
    for (final p in pieces) {
      var cur = root;
      for (var i = 0; i < p.length; i++) {
        cur = cur.children.putIfAbsent(p[i], () => _MutableTrieNode());
      }
      cur.isTerminal = true;
    }
    return root.freeze();
  }
}

class _MutableTrieNode {
  final Map<String, _MutableTrieNode> children = {};
  bool isTerminal = false;

  _TrieNode freeze() {
    return _TrieNode(
      children: children.map((k, v) => MapEntry(k, v.freeze())),
      isTerminal: isTerminal,
    );
  }
}
