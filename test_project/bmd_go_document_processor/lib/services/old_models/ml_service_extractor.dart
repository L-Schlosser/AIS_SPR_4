import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
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
  /// Delimiters so the model can separate boilerplate from data (reduces instruction echo).
  static const ocrTextBegin = '--- OCR TEXT ---';
  static const extractionBegin = '--- START EXTRACTION NOW ---';

  /// One-line JSON template keyed to classifier label — avoids receipt-shaped nulls on sick notes.
  static String shapeExampleJson(String documentType) {
    final t = documentType.trim().toLowerCase();
    if (t.isEmpty || t == 'unknown') {
      return '{"subject":null,"primary_date":null,"amount":null,"party":null}';
    }
    if (t.contains('krank') ||
        t.contains('arbeitsun') ||
        t.contains('attest') ||
        t.contains('eintritt') ||
        t.contains('pflege') ||
        t.contains('befrei') ||
        t.contains('sick') ||
        t.contains('medical') ||
        t.contains('doctor')) {
      return '{"patient_name":null,"diagnosis_or_reason":null,"unable_from":null,"unable_until":null,"issue_date":null,"doctor_or_issuer":null,"address":null}';
    }
    if (t.contains('rechn') ||
        t.contains('beleg') ||
        t.contains('kass') ||
        t.contains('quitt') ||
        t.contains('invoice') ||
        t.contains('lieferschein')) {
      return '{"merchant":null,"total":null,"currency":null,"date":null,"invoice_id":null}';
    }
    if (t.contains('vertrag') || t.contains('miet') || t.contains('künd')) {
      return '{"parties":null,"start_date":null,"end_date":null,"subject":null}';
    }
    if (t.contains('konto') || t.contains('bank') || t.contains('kontoausz')) {
      return '{"account_holder":null,"iban":null,"booking_date":null,"amount":null,"reference":null}';
    }
    return '{"subject":null,"primary_date":null,"amount":null,"reference":null}';
  }

  /// OCR first, then a short tail task (better for seq2seq than instructions before OCR).
  static String getRouterPrompt(String documentType, String ocrText) {
    final type = documentType.trim().isEmpty ? 'unknown' : documentType.trim();
    final shape = shapeExampleJson(documentType);
    return '''
$ocrTextBegin
$ocrText
$extractionBegin
Doc: $type
One JSON object only. No prose, no markdown, no bullet lists. Use the keys below; copy text from the OCR above; use null if missing.
$shape
''';
  }

  /// Numbered fallback: minimal rules, copy-from-OCR example.
  static String getNumberedFallbackPrompt(String documentType, String ocrText) {
    final type = documentType.trim().isEmpty ? 'unknown' : documentType.trim();
    return '''
$ocrTextBegin
$ocrText
$extractionBegin
Doc: $type
List facts from the OCR only. Numbered lines only (example). No titles, no rules, no OCR repeat.

Example:
1. REWE City
2. 3.45 EUR

''';
  }

  @Deprecated('Use getNumberedFallbackPrompt for retry; kept for experiments.')
  static String getKeyValueFallbackPrompt(String documentType, String ocrText) {
    final type = documentType.trim().isEmpty ? 'unknown' : documentType.trim();
    return '''
Facts from OCR only. Lines: key = value. No bullets, no OCR repeat.
Type: $type

$ocrTextBegin
$ocrText
$extractionBegin
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

      final promptBudget = max(96, maxInputTokens ~/ 2);
      var prompt1 = ExtractionSchemas.getRouterPrompt(documentType, cleanedOcr);
      prompt1 = _clipPromptToTokenBudget(prompt1, cleanedOcr, promptBudget);
      _logPromptTokenSplit(prompt1, cleanedOcr);

      var out1 = await _runInference(
        prompt: prompt1,
        maxInputTokens: maxInputTokens,
        maxNewTokens: maxNewTokens,
      );
      if (kDebugMode) {
        debugPrint('MlServiceExtractor raw output (first 400 chars): ${out1.length > 400 ? out1.substring(0, 400) : out1}');
      }

      final san1 = _cleanInstructionEchoOutput(out1);
      Map<String, dynamic> parsed = MlServiceExtractor._parseJsonFromOutput(san1) ??
          _parseNumberedExtractionLines(san1) ??
          const <String, dynamic>{};

      if (_mapHasOnlyNullOrEmptyValues(parsed)) {
        parsed = const <String, dynamic>{};
      }

      final echo1 = _looksLikeInstructionEcho(san1);
      final mapBad1 = parsed.isNotEmpty && _mapLooksLikeInstructions(parsed);

      if (parsed.isEmpty || echo1 || mapBad1) {
        if (echo1 || mapBad1) {
          parsed = const <String, dynamic>{};
        }
        final prompt2 = ExtractionSchemas.getNumberedFallbackPrompt(documentType, cleanedOcr);
        final prompt2clipped = _clipPromptToTokenBudget(prompt2, cleanedOcr, promptBudget);
        _logPromptTokenSplit(prompt2clipped, cleanedOcr);

        final out2 = await _runInference(
          prompt: prompt2clipped,
          maxInputTokens: maxInputTokens,
          maxNewTokens: maxNewTokens,
        );
        if (kDebugMode) {
          debugPrint('MlServiceExtractor numbered retry (first 400 chars): ${out2.length > 400 ? out2.substring(0, 400) : out2}');
        }
        final san2 = _cleanInstructionEchoOutput(out2);
        parsed = MlServiceExtractor._parseJsonFromOutput(san2) ??
            _parseNumberedExtractionLines(san2) ??
            _parseKeyValueLines(san2);
      }

      if (_mapHasOnlyNullOrEmptyValues(parsed)) {
        parsed = const <String, dynamic>{};
      }

      // Final guard: never return instruction/rule text as extracted data.
      if (parsed.isNotEmpty && _mapLooksLikeInstructions(parsed)) {
        parsed = const <String, dynamic>{};
      }

      if (parsed.isEmpty) {
        final fallback = _genericFeatureExtract(cleanedOcr, documentType: documentType);
        return ExtractedDocumentData(
          documentType: documentType,
          data: fallback,
          extractionErrors: const [
            'Model output contained no structured fields; used generic OCR feature extraction fallback.'
          ],
        );
      }

      final merged = _mergeOcrSignalFallback(parsed, cleanedOcr, documentType);
      return ExtractedDocumentData(
        documentType: documentType,
        data: merged,
      );
    } catch (e) {
      return ExtractedDocumentData(documentType: documentType, data: <String, dynamic>{}, extractionErrors: ['Extraction error: $e']);
    }
  }

  int _countTokens(String text) => _tokenizer!.encode(text, maxLength: 16384).length;

  /// Keeps prefix + suffix markers; shrinks OCR middle only so instructions stay intact.
  String _clipPromptToTokenBudget(String prompt, String ocrBody, int maxTokens) {
    if (_countTokens(prompt) <= maxTokens) return prompt;

    const begin = ExtractionSchemas.ocrTextBegin;
    const end = ExtractionSchemas.extractionBegin;
    final i0 = prompt.indexOf(begin);
    final i1 = prompt.indexOf(end);
    if (i0 < 0 || i1 < i0 + begin.length) {
      return _tailTruncatePrompt(prompt, maxTokens);
    }

    final prefix = prompt.substring(0, i0 + begin.length);
    var ocr = prompt.substring(i0 + begin.length, i1);
    final suffix = prompt.substring(i1);
    var guard = 0;
    while (_countTokens(prefix + ocr + suffix) > maxTokens && ocr.length > 200 && guard < 40) {
      ocr = _compactOcrText(ocr, maxChars: (ocr.length * 0.82).floor().clamp(200, ocr.length));
      guard++;
    }
    if (_countTokens(prefix + ocr + suffix) > maxTokens) {
      ocr = _compactOcrText(ocrBody, maxChars: min(ocrBody.length, 1200));
    }
    return prefix + ocr + suffix;
  }

  String _tailTruncatePrompt(String prompt, int maxTokens) {
    var p = prompt;
    var guard = 0;
    while (_countTokens(p) > maxTokens && p.length > 120 && guard < 50) {
      p = p.substring(0, (p.length * 0.88).floor());
      guard++;
    }
    return p;
  }

void _logPromptTokenSplit(String fullPrompt, String ocrBlock) {
    if (!kDebugMode) return;
    final begin = fullPrompt.indexOf(ExtractionSchemas.ocrTextBegin);
    final prefix = begin >= 0 ? fullPrompt.substring(0, begin) : fullPrompt;
    final tokPrefix = _tokenizer!.encode(prefix, maxLength: 16384).length;
    final tokOcr = _tokenizer!.encode(ocrBlock, maxLength: 16384).length;
    final tokTotal = _tokenizer!.encode(fullPrompt, maxLength: 16384).length;
    debugPrint(
      'MlServiceExtractor prompt tokens: prefix≈$tokPrefix ocr(block)≈$tokOcr full≈$tokTotal',
    );
    if (tokPrefix > tokOcr && tokOcr > 0) {
      debugPrint(
        'MlServiceExtractor warning: instruction token estimate exceeds OCR block; prompts may be top-heavy.',
      );
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
    var trimmed = _stripEchoedOcrFromDecoderOutput(output).trim();

    // Strip common markdown fences (even though we ask for none).
    if (trimmed.startsWith('```')) {
      trimmed = trimmed
          .replaceFirst(RegExp(r'^```(?:json)?\s*', caseSensitive: false), '')
          .replaceFirst(RegExp(r'\s*```$'), '')
          .trim();
    }

    if (!trimmed.contains('{')) {
      final wrapped = _tryBraceWrapJsonKeyFragment(trimmed);
      if (wrapped != null) trimmed = wrapped;
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

/// Cuts off anything from our OCR delimiter onward (models often paste the prompt/OCR back).
String _stripEchoedOcrFromDecoderOutput(String output) {
  var s = output.trim();
  final lo = s.toLowerCase();
  final needles = <String>[
    ExtractionSchemas.ocrTextBegin.toLowerCase(),
    '--- ocr text ---',
    '- ocr text ---',
    'ocr text ---',
  ];
  var cutAt = -1;
  for (final n in needles) {
    final at = lo.indexOf(n);
    if (at >= 0) {
      cutAt = cutAt < 0 ? at : min(cutAt, at);
    }
  }
  if (cutAt >= 0) {
    s = s.substring(0, cutAt).trim();
  }
  return s;
}

/// Wraps a bare `"k":v,...` fragment so [jsonDecode] can parse it.
String? _tryBraceWrapJsonKeyFragment(String s) {
  var t = s.trim();
  t = t.replaceFirst(RegExp(r'^[\s\-–—•]+'), '');
  if (t.startsWith('{')) return null;
  final q = t.indexOf('"');
  if (q < 0) return null;
  t = t.substring(q);
  if (!RegExp(r'^"[^"]+"\s*:').hasMatch(t)) return null;
  t = t.replaceFirst(RegExp(r',\s*$'), '');
  if (!t.endsWith('}')) {
    return '{$t}';
  }
  if (!t.startsWith('{')) {
    return '{$t';
  }
  return t;
}

bool _mapHasOnlyNullOrEmptyValues(Map<String, dynamic> m) {
  if (m.isEmpty) return false;
  for (final v in m.values) {
    if (v == null) continue;
    if (v is String && v.trim().isEmpty) continue;
    return false;
  }
  return true;
}

/// Fills gaps with deterministic OCR regexes; model fields win on key clashes.
Map<String, dynamic> _mergeOcrSignalFallback(
  Map<String, dynamic> model,
  String ocrText,
  String documentType,
) {
  final fb = _genericFeatureExtract(ocrText, documentType: documentType);
  if (fb.isEmpty) return model;
  final out = Map<String, dynamic>.from(fb);
  for (final e in model.entries) {
    if (_isMissingMergeValue(e.value)) continue;
    out[e.key] = e.value;
  }
  return out;
}

bool _isMissingMergeValue(dynamic v) {
  if (v == null) return true;
  if (v is String && v.trim().isEmpty) return true;
  return false;
}

/// Dedupes repeated rule lines, strips obvious garbage tokens, drops instruction-looking lines.
String _cleanInstructionEchoOutput(String raw) {
  var s = _stripEchoedOcrFromDecoderOutput(raw).trim();

  s = s.replaceAll(RegExp(r'\b([KXkx#])\1{3,}\b'), ' ');
  s = s.replaceAll(RegExp(r'\bx{5,}\b', caseSensitive: false), ' ');
  s = s.replaceAll(RegExp(r'\bk{5,}\b', caseSensitive: false), ' ');
  s = s.replaceAll(RegExp(r'[#]{6,}'), ' ');

  final rawLines = s.split('\n');
  final kept = <String>[];
  String? prevNorm;
  final seenNorm = <String>{};

  for (final rawLine in rawLines) {
    var line = rawLine.trim();
    if (line.isEmpty) continue;
    if (_lineLooksLikeInstruction(line)) continue;

    final norm = line.toLowerCase().replaceAll(RegExp(r'\s+'), ' ');
    if (prevNorm == norm) continue;
    prevNorm = norm;

    if (line.startsWith('- ') || line.startsWith('• ')) {
      final rest = line.substring(2).trim();
      if (_startsWithInstructionKeyword(rest)) continue;
    }

    if (_looksIncompleteTruncatedLine(line)) continue;

    if (seenNorm.contains(norm) && _isLikelyRuleLine(norm)) continue;
    seenNorm.add(norm);

    kept.add(line);
  }

  return kept.join('\n').trim();
}

bool _startsWithInstructionKeyword(String s) {
  final t = s.toLowerCase();
  const keys = [
    'nur ',
    'keine ',
    'wiederhole',
    'zahlen ',
    'arrays ',
    'datum ',
    'vorgaben',
    'json',
    'ocr',
    'listen ',
    'null ',
    'output ',
    'no ',
    'do not',
    'must ',
    'rules',
  ];
  for (final k in keys) {
    if (t.startsWith(k)) return true;
  }
  return false;
}

bool _isLikelyRuleLine(String norm) {
  return _instructionKeywordHitCount(norm) >= 1;
}

int _instructionKeywordHitCount(String t) {
  var hits = 0;
  const keys = [
    'nur echte',
    'keine metadaten',
    'keine erkl',
    'json-objekt',
    'yyyy-mm-dd',
    'vorgaben',
    'arrays bleiben',
    'null nur',
    'ocr-text',
    'schluessel',
    'wiederhole',
    'output only',
    'no explanation',
    'no markdown',
  ];
  for (final k in keys) {
    if (t.contains(k)) hits++;
  }
  return hits;
}

bool _looksIncompleteTruncatedLine(String line) {
  if (line.length < 24) return false;
  final t = line.trimRight();
  if (t.endsWith('-') || t.endsWith('–') || t.endsWith('—')) return true;
  final parts = t.split(RegExp(r'\s+'));
  if (parts.length < 2) return false;
  final last = parts.last;
  if (last.length <= 2 && RegExp(r'^[A-Za-zÄÖÜäöü]{1,2}$').hasMatch(last)) {
    return true;
  }
  return false;
}

/// Parses `1. fact` / `2) fact` lines into a flat map (fact_1, fact_2, ...).
Map<String, dynamic>? _parseNumberedExtractionLines(String s) {
  final cleaned = _cleanInstructionEchoOutput(s);
  final out = <String, dynamic>{};
  final re = RegExp(r'^\s*(\d+)\s*[\).:]\s*(.+)$');
  for (final raw in cleaned.split('\n')) {
    final line = raw.trim();
    if (line.isEmpty) continue;
    final m = re.firstMatch(line);
    if (m == null) continue;
    final idx = m.group(1)!;
    var value = m.group(2)!.trim();
    if (value.isEmpty) continue;
    if (_lineLooksLikeInstruction(value)) continue;
    value = value.replaceAll(RegExp(r'[;,.]+$'), '').trim();
    out['fact_$idx'] = _coerceScalar(value);
  }
  return out.isEmpty ? null : out;
}

bool _looksLikeInstructionEcho(String s) {
  final trimmed = s.trim();
  final head = _stripEchoedOcrFromDecoderOutput(s);
  if (trimmed.isNotEmpty && head.length < trimmed.length - 15) {
    return true;
  }
  final t = head.toLowerCase();
  final badPhrases = <String>[
    'ausgabe',
    'muss',
    'genau',
    'json-objekt',
    'regeln',
    'keine erklärungen',
    'keine erkl',
    'null nur',
    'arrays bleiben arrays',
    'arrays bleiben',
    'yyyy-mm-dd',
    'vorgaben',
    'nur echte informationen',
    'keine metadaten',
    'ocr-text',
    'wiederhole nicht',
    'keine aufzählung',
    'schluessel = wert',
    'output only',
    'no explanation',
    'no markdown',
    'do not repeat',
    'must be valid json',
    'extract structured data',
    'one json object only',
    'no prose, no markdown',
  ];
  var hits = 0;
  for (final p in badPhrases) {
    if (t.contains(p)) hits++;
  }

  if (RegExp(r'([a-zäöüß]{4,})\s*:\s*\1\b').hasMatch(t)) {
    hits += 2;
  }

  var bulletInstructionLines = 0;
  var duplicateRuleLines = 0;
  final lineNorms = <String, int>{};
  for (final raw in head.split('\n')) {
    final line = raw.trim();
    if (line.isEmpty) continue;
    if (line.startsWith('- ') || line.startsWith('• ')) {
      final rest = line.substring(2).trim();
      if (_startsWithInstructionKeyword(rest)) bulletInstructionLines++;
    }
    final norm = line.toLowerCase().replaceAll(RegExp(r'\s+'), ' ');
    if (_instructionKeywordHitCount(norm) >= 2) {
      lineNorms[norm] = (lineNorms[norm] ?? 0) + 1;
    }
  }
  for (final c in lineNorms.values) {
    if (c >= 2) duplicateRuleLines++;
  }
  if (bulletInstructionLines >= 2) hits += 2;
  if (duplicateRuleLines >= 1) hits += 2;

  if (RegExp(r'\b[xk#]{5,}\b', caseSensitive: false).hasMatch(head)) hits++;

  final hasDocSignal = RegExp(
    r'(\d{2,})|(\biban\b)|(\beur\b)|[€$]|(\bAT\d{2}\b)|(\d+[.,]\d{2})',
  ).hasMatch(head);

  if (hits >= 3 && !hasDocSignal) return true;

  final hasJsonBraces = head.contains('{') && head.contains('}');
  if (hits >= 2 && !hasJsonBraces && !hasDocSignal) return true;

  if (bulletInstructionLines >= 3) return true;

  return false;
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
  const needles = <String>[
    'nur echte informationen',
    'nichts erfinden',
    'keine metadaten',
    'keine erklärungen',
    'keine erkl',
    'arrays bleiben arrays',
    'arrays bleiben',
    'null nur',
    'yyyy-mm-dd',
    'vorgaben',
    'ausgabe',
    'json-objekt',
    'ocr-text',
    'wiederhole nicht',
    'keine aufzählungszeichen',
    'schluessel = wert',
    'output only',
    'no explanation',
    'no markdown',
    'do not repeat',
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
  final isMedical = dt.contains('krank') ||
      dt.contains('pflege') ||
      dt.contains('arzt') ||
      dt.contains('arbeitsun') ||
      dt.contains('attest') ||
      dt.contains('sick') ||
      dt.contains('medical');

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

  final auFrom = RegExp(
    r'\bArbeitsunfähig(?:keit)?\s+von\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})',
    caseSensitive: false,
  ).firstMatch(text);
  if (auFrom != null && periodStart == null) {
    periodStart = _normalizeDate(auFrom.group(1));
  }

  final auBis = RegExp(
    r'\bArbeitsunfähig(?:keit)?\s+bis\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})',
    caseSensitive: false,
  ).firstMatch(text);
  if (auBis != null && periodEnd == null) {
    periodEnd = _normalizeDate(auBis.group(1));
  }

  // Issue date near "Datum"
  String? issueDate;
  final aus = RegExp(
    r'\bAusstellungsdatum\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})',
    caseSensitive: false,
  ).firstMatch(text);
  if (aus != null) {
    issueDate = _normalizeDate(aus.group(1));
  }
  final issue = RegExp(
    r'\bdatum\b[^0-9]{0,10}(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})',
    caseSensitive: false,
  ).firstMatch(text);
  if (issue != null && issueDate == null) {
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
  if (patient == null) {
    final fam = RegExp(
      r'\b(?:Familiar\s+name|Familienname|Name)\s*,?\s*Vorname\s*,?\s*([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\s\-]{1,80}?)(?=\s{2,}|\s*Krank|\s*Arbeits|\s*Ausstell|$)',
      caseSensitive: false,
    ).firstMatch(text);
    if (fam != null) patient = fam.group(1)?.trim();
  }
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

  String? diagnosisOrReason;
  final gr = RegExp(
    r'\bGrund\s+der\s+Arbeitsunfähigkeit\s*:?\s*([^\n]{1,160})',
    caseSensitive: false,
  ).firstMatch(text);
  if (gr != null) diagnosisOrReason = gr.group(1)?.trim();

  return <String, dynamic>{
    if (periodStart != null) 'period_start': periodStart,
    if (periodEnd != null) 'period_end': periodEnd,
    if (issueDate != null) 'issue_date': issueDate,
    if (dates.isNotEmpty) 'dates': dates.toList()..sort(),
    if (doctor != null) 'doctor': doctor,
    if (patient != null) 'patient': patient,
    if (diagnosisOrReason != null) 'diagnosis_or_reason': diagnosisOrReason,
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
