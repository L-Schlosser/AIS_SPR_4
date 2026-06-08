import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class ExtractedEntity {
  final String label;
  final String text;
  final int start;
  final int end;
  final double score;

  const ExtractedEntity({
    required this.label,
    required this.text,
    required this.start,
    required this.end,
    required this.score,
  });
}

class GLiNERService {
  GLiNERService({
    this.modelAssetPath = 'assets/models/GLiNER/model_int8.onnx',
    this.tokenizerAssetPath = 'assets/models/GLiNER/tokenizer.json',
    this.tokenizerConfigAssetPath = 'assets/models/GLiNER/tokenizer_config.json',
    this.configAssetPath = 'assets/models/GLiNER/gliner_config.json',
    this.labelsAssetPath = 'assets/models/GLiNER/class_labels.json',
    this.threshold = 0.5,
    this.fallbackThresholds = const [0.35, 0.25, 0.15, 0.1, 0.05],
    this.debugModelOutput = false,
    this.flatNer = true,
    this.multiLabel = true,
  });

  final String modelAssetPath;
  final String tokenizerAssetPath;
  final String tokenizerConfigAssetPath;
  final String configAssetPath;
  final String labelsAssetPath;
  final double threshold;
  final List<double> fallbackThresholds;
  final bool debugModelOutput;
  final bool flatNer;
  final bool multiLabel;

  late final OrtSession _session;
  late final _GlinerTokenizer _tokenizer;
  late final Map<String, List<String>> _labelsByDocumentType;
  late final int _maxLength;
  late final int _maxSpanLength;
  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) {
      return;
    }

    final configMap = jsonDecode(await rootBundle.loadString(configAssetPath)) as Map<String, dynamic>;
    _maxLength = (configMap['max_length'] as num?)?.toInt() ?? 384;
    _maxSpanLength = (configMap['max_span_length'] as num?)?.toInt() ?? 12;

    _labelsByDocumentType = _loadLabelsByDocumentType(
      jsonDecode(await rootBundle.loadString(labelsAssetPath)),
    );

    _tokenizer = _GlinerTokenizer.fromJson(
      jsonDecode(await rootBundle.loadString(tokenizerAssetPath)) as Map<String, dynamic>,
      configJson: jsonDecode(await rootBundle.loadString(tokenizerConfigAssetPath)) as Map<String, dynamic>,
      maxLength: _maxLength,
    );

    final modelData = await rootBundle.load(modelAssetPath);
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(modelData.buffer.asUint8List(), sessionOptions);

    _initialized = true;
  }

  Future<List<ExtractedEntity>> extract(String text, {String? documentType}) async {
    if (!_initialized) {
      throw StateError('GLiNERService is not initialized. Call initialize() first.');
    }

    if (text.trim().isEmpty) {
      return const <ExtractedEntity>[];
    }

    final labels = _resolveLabels(documentType);
    if (labels.isEmpty) {
      return const <ExtractedEntity>[];
    }

    final labelCount = labels.length;
    final encoded = _tokenizer.encode(text, labels, _maxSpanLength);
    if (encoded.wordSpans.isEmpty || encoded.inputIds.length < 2) {
      return const <ExtractedEntity>[];
    }

    if (debugModelOutput) {
      print('[GLiNER] encoded inputIds=${encoded.inputIds.length} attentionMask=${encoded.attentionMask.length} wordsMask=${encoded.wordsMask.length} words=${encoded.wordSpans.length} spans=${encoded.spanCount}');
      print('[GLiNER] spanMask valid=${encoded.spanMask.where((e) => e != 0).length}/${encoded.spanMask.length}');
      print('[GLiNER] first inputIds=${encoded.inputIds.take(20).toList()}');
      print('[GLiNER] first wordsMask=${encoded.wordsMask.take(40).toList()}');
    }

    final inputNameMap = _buildInputNameMap(_session.inputNames);

    Map<String, OrtValue> buildInputs() {
      final m = <String, OrtValue>{};
      m[inputNameMap['input_ids']!] = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(encoded.inputIds),
        [1, encoded.inputIds.length],
      );
      m[inputNameMap['attention_mask']!] = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(encoded.attentionMask),
        [1, encoded.attentionMask.length],
      );
      m[inputNameMap['words_mask']!] = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(encoded.wordsMask),
        [1, encoded.wordsMask.length],
      );
      m[inputNameMap['span_mask']!] = OrtValueTensor.createTensorWithDataList(
        encoded.spanMask.map((e) => e != 0).toList(growable: false),
        [1, encoded.spanCount],
      );
      m[inputNameMap['text_lengths']!] = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList([encoded.wordSpans.length]),
        [1, 1],
      );
      m[inputNameMap['span_idx']!] = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(encoded.spanIdxFlat),
        [1, encoded.spanCount, 2],
      );
      return m;
    }
    final outputs = await _session.runAsync(OrtRunOptions(), buildInputs());
    if (outputs == null || outputs.isEmpty) {
      throw StateError('GLiNER returned no outputs.');
    }
    if (debugModelOutput) {
      print('[GLiNER] output names: ${_session.outputNames}');
      for (var i = 0; i < outputs.length; i++) {
        final name = i < _session.outputNames.length ? _session.outputNames[i] : 'output_$i';
        final value = outputs[i]?.value;
        print('[GLiNER] $name type=${value.runtimeType} value=${_previewValue(value)}');
      }
    }


    final logits = _pickLogits(outputs, _session.outputNames);
    final sampleLogits = _selectSingleSample(logits);

    if (debugModelOutput) {
      print('[GLiNER] sampleLogits runtimeType=${sampleLogits.runtimeType}');
      print('[GLiNER] sampleLogits preview=${_previewValue(sampleLogits)}');
      final maxRaw = _findMaxNumeric(sampleLogits);
      if (maxRaw != null) {
        print('[GLiNER] max raw logit=$maxRaw sigmoid=${_sigmoid(maxRaw)}');
      }
    }

    final candidates = <_CandidateEntity>[];
    if (sampleLogits.isNotEmpty && sampleLogits.first is List) {
      final firstRow = sampleLogits.first as List;
      if (firstRow.isNotEmpty && firstRow.first is List) {
        _decodeGridLogits(
          sampleLogits: sampleLogits,
          wordSpans: encoded.wordSpans,
          candidates: candidates,
          labelNames: labels,
          labelCount: labelCount,
        );
      } else {
        _decodeSpanLogits(
          sampleLogits: sampleLogits,
          spanIdx: encoded.spanIdx,
          spanMask: encoded.spanMask,
          wordSpans: encoded.wordSpans,
          candidates: candidates,
          labelNames: labels,
          labelCount: labelCount,
        );
      }
    }

    if (debugModelOutput) {
      print('[GLiNER] candidate count before filtering: ${candidates.length}');
      for (final candidate in candidates.take(10)) {
        print('[GLiNER] candidate ${candidate.label} "${candidate.start}-${candidate.end}" score=${candidate.score} text="${encoded.normalizedText.substring(candidate.start, candidate.end)}"');
      }
    }

    final primary = _greedySearch(candidates, flatNer: flatNer, multiLabel: multiLabel);
    if (primary.isNotEmpty) {
      return _toEntities(primary, encoded.normalizedText);
    }

    for (final fallbackThreshold in fallbackThresholds) {
      final fallbackCandidates = <_CandidateEntity>[];
      if (sampleLogits.isNotEmpty && sampleLogits.first is List) {
        final firstRow = sampleLogits.first as List;
        if (firstRow.isNotEmpty && firstRow.first is List) {
          _decodeGridLogits(
            sampleLogits: sampleLogits,
            wordSpans: encoded.wordSpans,
            candidates: fallbackCandidates,
            labelNames: labels,
            labelCount: labelCount,
            thresholdOverride: fallbackThreshold,
          );
        } else {
          _decodeSpanLogits(
            sampleLogits: sampleLogits,
            spanIdx: encoded.spanIdx,
            spanMask: encoded.spanMask,
            wordSpans: encoded.wordSpans,
            candidates: fallbackCandidates,
            labelNames: labels,
            labelCount: labelCount,
            thresholdOverride: fallbackThreshold,
          );
        }
      }

      final fallback = _greedySearch(
        fallbackCandidates,
        flatNer: flatNer,
        multiLabel: multiLabel,
      );
      if (fallback.isNotEmpty) {
        return _toEntities(fallback, encoded.normalizedText);
      }
    }

    return const <ExtractedEntity>[];
  }

  String _previewValue(dynamic value) {
    final text = value.toString();
    if (text.length <= 500) {
      return text;
    }
    return '${text.substring(0, 500)}...';
  }

  Map<String, List<String>> _loadLabelsByDocumentType(dynamic jsonValue) {
    if (jsonValue is List) {
      return <String, List<String>>{
        '__default__': jsonValue.map((value) => value.toString()).toList(growable: false),
      };
    }

    if (jsonValue is Map) {
      final labelsByType = <String, List<String>>{};
      for (final entry in jsonValue.entries) {
        final key = entry.key.toString();
        final value = entry.value;
        if (value is List) {
          labelsByType[key] = value.map((item) => item.toString()).toList(growable: false);
        }
      }
      return labelsByType;
    }

    return const <String, List<String>>{};
  }

  List<String> _resolveLabels(String? documentType) {
    final normalizedDocumentType = _normalizeDocumentType(documentType);
    if (normalizedDocumentType != null) {
      final typedLabels = _labelsByDocumentType[normalizedDocumentType];
      if (typedLabels != null && typedLabels.isNotEmpty) {
        return typedLabels;
      }
    }

    final defaultLabels = _labelsByDocumentType['__default__'];
    if (defaultLabels != null && defaultLabels.isNotEmpty) {
      return defaultLabels;
    }

    final flattened = <String>[];
    final seen = <String>{};
    for (final labels in _labelsByDocumentType.values) {
      for (final label in labels) {
        if (seen.add(label)) {
          flattened.add(label);
        }
      }
    }
    return flattened;
  }

  String? _normalizeDocumentType(String? documentType) {
    if (documentType == null) {
      return null;
    }

    final normalized = documentType.trim().toLowerCase();
    if (normalized.isEmpty || normalized == 'unknown') {
      return null;
    }

    const aliases = <String, String>{
      'invoice': 'Rechnung',
      'rechnung': 'Rechnung',
      'receipt': 'Kassenbeleg',
      'kassenbeleg': 'Kassenbeleg',
      'doctor_note': 'Arztbestaetigung',
      'doctor note': 'Arztbestaetigung',
      'doctor_note_': 'Arztbestaetigung',
      'arztbestaetigung': 'Arztbestaetigung',
      'care_leave': 'Pflegefreistellung',
      'care leave': 'Pflegefreistellung',
      'pflegefreistellung': 'Pflegefreistellung',
      'delivery_note': 'Lieferschein',
      'delivery note': 'Lieferschein',
      'lieferschein': 'Lieferschein',
      'master_data': 'master_data',
      'master data': 'master_data',
    };

    return aliases[normalized] ?? normalized;
  }

  double? _findMaxNumeric(dynamic value) {
    double? maxValue;

    void visit(dynamic node) {
      if (node is num) {
        final candidate = node.toDouble();
        maxValue = maxValue == null || candidate > maxValue! ? candidate : maxValue;
        return;
      }
      if (node is List) {
        for (final item in node) {
          visit(item);
        }
      }
    }

    visit(value);
    return maxValue;
  }

  List<ExtractedEntity> _toEntities(List<_CandidateEntity> candidates, String text) {
    return candidates
        .map(
          (candidate) => ExtractedEntity(
            label: candidate.label,
            text: text.substring(candidate.start, candidate.end),
            start: candidate.start,
            end: candidate.end,
            score: candidate.score,
          ),
        )
        .toList(growable: false);
  }

  void dispose() {
    if (_initialized) {
      _session.release();
      _initialized = false;
    }
  }

  Map<String, String> _buildInputNameMap(List<String> inputNames) {
    final resolved = <String, String>{};
    String pick(List<String> candidates) {
      for (final candidate in candidates) {
        if (inputNames.contains(candidate)) {
          return candidate;
        }
      }
      return inputNames.first;
    }

    resolved['input_ids'] = pick(const ['input_ids', 'inputIds', 'ids']);
    resolved['attention_mask'] = pick(const ['attention_mask', 'attentionMask', 'mask']);
    resolved['words_mask'] = pick(const ['words_mask', 'wordsMask']);
    resolved['text_lengths'] = pick(const ['text_lengths', 'textLengths']);
    resolved['span_idx'] = pick(const ['span_idx', 'spanIdx']);
    resolved['span_mask'] = pick(const ['span_mask', 'spanMask']);
    return resolved;
  }

  dynamic _pickLogits(List<OrtValue?> outputs, List<String> outputNames) {
    for (var i = 0; i < min(outputs.length, outputNames.length); i++) {
      if (outputNames[i] == 'logits') {
        return outputs[i]!.value;
      }
    }
    return outputs.first!.value;
  }

  List<dynamic> _selectSingleSample(dynamic logits) {
    if (logits is List && logits.isNotEmpty && logits.first is List) {
      final first = logits.first as List;
      if (first.isNotEmpty && first.first is List) {
        return first;
      }
    }
    return logits as List<dynamic>;
  }

  void _decodeGridLogits({
    required List<dynamic> sampleLogits,
    required List<_WordSpan> wordSpans,
    required List<_CandidateEntity> candidates,
    required List<String> labelNames,
    required int labelCount,
    double thresholdOverride = -1,
  }) {
    final wordCount = wordSpans.length;
    final minThreshold = thresholdOverride >= 0 ? thresholdOverride : threshold;
    for (var start = 0; start < sampleLogits.length; start++) {
      final widthRows = sampleLogits[start] as List<dynamic>;
      for (var width = 0; width < widthRows.length; width++) {
        if (start + width >= wordCount) {
          continue;
        }

        final classScores = widthRows[width] as List<dynamic>;
        for (var classIndex = 0; classIndex < min(classScores.length, labelCount); classIndex++) {
          final score = _sigmoid((classScores[classIndex] as num).toDouble());
          if (score <= minThreshold) {
            continue;
          }
          candidates.add(
            _CandidateEntity(
              start: wordSpans[start].start,
              end: wordSpans[start + width].end,
              label: labelNames[classIndex],
              score: score,
            ),
          );
        }
      }
    }
  }

  void _decodeSpanLogits({
    required List<dynamic> sampleLogits,
    required List<List<int>> spanIdx,
    required List<int> spanMask,
    required List<_WordSpan> wordSpans,
    required List<_CandidateEntity> candidates,
    required List<String> labelNames,
    required int labelCount,
    double thresholdOverride = -1,
  }) {
    final spanCount = min(spanIdx.length, sampleLogits.length);
    final minThreshold = thresholdOverride >= 0 ? thresholdOverride : threshold;
    for (var i = 0; i < spanCount; i++) {
      if (i >= spanMask.length || spanMask[i] == 0) {
        continue;
      }

      final span = spanIdx[i];
      if (span.length < 2) {
        continue;
      }

      final startWord = span[0];
      final endWord = span[1];
      if (startWord < 0 || endWord < startWord || endWord >= wordSpans.length) {
        continue;
      }

      final classScores = sampleLogits[i] as List<dynamic>;
      for (var classIndex = 0; classIndex < min(classScores.length, labelCount); classIndex++) {
        final score = _sigmoid((classScores[classIndex] as num).toDouble());
        if (score <= minThreshold) {
          continue;
        }
        candidates.add(
          _CandidateEntity(
            start: wordSpans[startWord].start,
            end: wordSpans[endWord].end,
            label: labelNames[classIndex],
            score: score,
          ),
        );
      }
    }
  }

  List<_CandidateEntity> _greedySearch(
    List<_CandidateEntity> candidates, {
    required bool flatNer,
    required bool multiLabel,
  }) {
    if (candidates.isEmpty) {
      return const <_CandidateEntity>[];
    }

    final sorted = [...candidates]..sort((left, right) => right.score.compareTo(left.score));
    final kept = <_CandidateEntity>[];

    for (final candidate in sorted) {
      var overlaps = false;
      for (final existing in kept) {
        if (_hasOverlap(candidate, existing, flatNer: flatNer, multiLabel: multiLabel)) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) {
        kept.add(candidate);
      }
    }

    kept.sort((left, right) {
      final startComparison = left.start.compareTo(right.start);
      if (startComparison != 0) {
        return startComparison;
      }
      final endComparison = left.end.compareTo(right.end);
      if (endComparison != 0) {
        return endComparison;
      }
      final labelComparison = left.label.compareTo(right.label);
      if (labelComparison != 0) {
        return labelComparison;
      }
      return right.score.compareTo(left.score);
    });

    return kept;
  }

  bool _hasOverlap(
    _CandidateEntity left,
    _CandidateEntity right, {
    required bool flatNer,
    required bool multiLabel,
  }) {
    if (left.start == right.start && left.end == right.end) {
      return !multiLabel;
    }

    final intersects = left.start <= right.end && right.start <= left.end;
    if (!intersects) {
      return false;
    }

    if (!flatNer) {
      final leftNestedInRight = left.start >= right.start && left.end <= right.end;
      final rightNestedInLeft = right.start >= left.start && right.end <= left.end;
      return !(leftNestedInRight || rightNestedInLeft);
    }

    return true;
  }

  double _sigmoid(double value) {
    return 1.0 / (1.0 + exp(-value));
  }
}

class _CandidateEntity {
  const _CandidateEntity({
    required this.start,
    required this.end,
    required this.label,
    required this.score,
  });

  final int start;
  final int end;
  final String label;
  final double score;
}

class _WordSpan {
  const _WordSpan(this.start, this.end);

  final int start;
  final int end;
}

class _EncodedInput {
  const _EncodedInput({
    required this.inputIds,
    required this.attentionMask,
    required this.wordsMask,
    required this.spanIdx,
    required this.spanMask,
    required this.wordSpans,
    required this.normalizedText,
  });

  final List<int> inputIds;
  final List<int> attentionMask;
  final List<int> wordsMask;
  final List<List<int>> spanIdx;
  final List<int> spanMask;
  final List<_WordSpan> wordSpans;
  final String normalizedText;

  int get spanCount => spanIdx.length;

  List<int> get spanIdxFlat {
    final out = <int>[];
    for (final pair in spanIdx) {
      if (pair.length >= 2) {
        out.add(pair[0]);
        out.add(pair[1]);
      } else if (pair.isNotEmpty) {
        out.add(pair[0]);
        out.add(pair[0]);
      } else {
        out.add(0);
        out.add(0);
      }
    }
    return out;
  }
}

class _GlinerTokenizer {
  _GlinerTokenizer._(
    this._tokenToId,
    this._tokenScore,
    this._trieRoot,
    this.maxLength,
    this.padId,
    this.clsId,
    this.sepId,
    this.entId,
    this.promptSepId,
  );

  factory _GlinerTokenizer.fromJson(
    Map<String, dynamic> json, {
    Map<String, dynamic>? configJson,
    required int maxLength,
  }) {
    final tokenToId = <String, int>{};
    final tokenScore = <String, double>{};
    final specialTokenIds = _readSpecialTokenIds(configJson ?? json);

    final model = json['model'] as Map<String, dynamic>? ?? const <String, dynamic>{};
    final vocab = model['vocab'] as List<dynamic>? ?? const <dynamic>[];
    for (var i = 0; i < vocab.length; i++) {
      final entry = vocab[i];
      if (entry is List && entry.length >= 2) {
        final token = entry[0].toString();
        final score = (entry[1] as num).toDouble();
        tokenToId[token] = i;
        tokenScore[token] = score;
      }
    }

    final trieRoot = _TrieNode();
    for (final entry in tokenToId.entries) {
      final token = entry.key;
      if (_excludedFromTrie(token)) {
        continue;
      }
      final score = tokenScore[token] ?? 0.0;
      trieRoot.insert(token, entry.value, score);
    }

    return _GlinerTokenizer._(
      tokenToId,
      tokenScore,
      trieRoot,
      maxLength,
      specialTokenIds['[PAD]'] ?? 0,
      specialTokenIds['[CLS]'] ?? 1,
      specialTokenIds['[SEP]'] ?? 2,
      specialTokenIds['<<ENT>>'] ?? 128002,
      specialTokenIds['<<SEP>>'] ?? 128003,
    );
  }

  final Map<String, int> _tokenToId;
  final Map<String, double> _tokenScore;
  final _TrieNode _trieRoot;
  final int maxLength;
  final int padId;
  final int clsId;
  final int sepId;
  final int entId;
  final int promptSepId;

  _EncodedInput encode(String text, List<String> labels, int maxSpanLength) {
    final words = _splitIntoWords(text);
    final keptWords = <_WordSpan>[];
    final inputIds = <int>[clsId];
    final wordsMask = <int>[0];

    for (var labelIndex = 0; labelIndex < labels.length; labelIndex++) {
      inputIds.add(entId);
      wordsMask.add(0);

      final labelTokenIds = _tokenizeText(labels[labelIndex]);
      for (final tokenId in labelTokenIds) {
        inputIds.add(tokenId);
        wordsMask.add(0);
      }

      inputIds.add(labelIndex == labels.length - 1 ? sepId : promptSepId);
      wordsMask.add(0);
    }

    for (var wordIndex = 0; wordIndex < words.length; wordIndex++) {
      final word = words[wordIndex];
      final wordText = text.substring(word.start, word.end);
      final normalized = '▁${wordText}';
      final wordTokenIds = _tokenizePiece(normalized);
      if (wordTokenIds.isEmpty) {
        continue;
      }

      if (inputIds.length + wordTokenIds.length + 1 > maxLength) {
        break;
      }

      keptWords.add(_WordSpan(word.start, word.end));
      for (var tokenIndex = 0; tokenIndex < wordTokenIds.length; tokenIndex++) {
        inputIds.add(wordTokenIds[tokenIndex]);
        wordsMask.add(tokenIndex == 0 ? keptWords.length : 0);
      }
    }

    inputIds.add(sepId);
    wordsMask.add(0);

    final attentionMask = List<int>.filled(inputIds.length, 1);

    final spanIdx = <List<int>>[];
    final spanMask = <int>[];
    for (var start = 0; start < keptWords.length; start++) {
      for (var width = 0; width < maxSpanLength; width++) {
        final end = start + width;
        spanIdx.add([start, end]);
        spanMask.add(end < keptWords.length ? 1 : 0);
      }
    }

    return _EncodedInput(
      inputIds: inputIds,
      attentionMask: attentionMask,
      wordsMask: wordsMask,
      spanIdx: spanIdx,
      spanMask: spanMask,
      wordSpans: keptWords,
      normalizedText: text,
    );
  }

  List<int> _tokenizePiece(String piece) {
    final runeChars = _runes(piece);
    final matches = <int>[];

    final dp = List<double>.filled(runeChars.length + 1, double.negativeInfinity);
    final prev = List<int>.filled(runeChars.length + 1, -1);
    final prevTokenId = List<int>.filled(runeChars.length + 1, -1);
    dp[0] = 0.0;

    for (var position = 0; position < runeChars.length; position++) {
      if (!dp[position].isFinite) {
        continue;
      }

      _TrieNode? node = _trieRoot;
      for (var end = position; end < runeChars.length; end++) {
        node = node?.children[runeChars[end]];
        if (node == null) {
          break;
        }
        if (node.tokenId != null) {
          final candidateScore = dp[position] + node.score;
          if (candidateScore > dp[end + 1]) {
            dp[end + 1] = candidateScore;
            prev[end + 1] = position;
            prevTokenId[end + 1] = node.tokenId!;
          }
        }
      }
    }

    if (dp[runeChars.length].isFinite) {
      var cursor = runeChars.length;
      while (cursor > 0 && prev[cursor] >= 0) {
        matches.add(prevTokenId[cursor]);
        cursor = prev[cursor];
      }
      return matches.reversed.toList(growable: false);
    }

    return _byteFallback(piece);
  }

  List<int> _tokenizeText(String text) {
    final tokenIds = <int>[];
    for (final word in _splitIntoWords(text)) {
      final piece = text.substring(word.start, word.end);
      tokenIds.addAll(_tokenizePiece('▁$piece'));
    }
    return tokenIds;
  }

  List<int> _byteFallback(String piece) {
    final bytes = utf8.encode(piece);
    final ids = <int>[];
    for (final byte in bytes) {
      final token = '<0x${byte.toRadixString(16).padLeft(2, '0').toUpperCase()}>';
      ids.add(_tokenToId[token] ?? _tokenToId['[UNK]'] ?? 3);
    }
    return ids;
  }

  List<_WordSpan> _splitIntoWords(String text) {
    final words = <_WordSpan>[];
    int? wordStart;

    bool isWhitespace(String ch) {
      return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\u000B' || ch == '\u000C';
    }

    bool isPunctuation(String ch) {
      const punctuation = r'.,;:!?()[]{}<>/\\|@#$%^&*_+=~`"\';
      return punctuation.contains(ch);
    }

    void flush(int end) {
      if (wordStart != null && wordStart! < end) {
        words.add(_WordSpan(wordStart!, end));
      }
      wordStart = null;
    }

    for (var index = 0; index < text.length; index++) {
      final ch = text[index];
      if (isWhitespace(ch)) {
        flush(index);
        continue;
      }
      if (isPunctuation(ch)) {
        flush(index);
        words.add(_WordSpan(index, index + 1));
        continue;
      }
      wordStart ??= index;
    }

    flush(text.length);
    return words;
  }

  List<String> _runes(String text) {
    return text.runes.map((rune) => String.fromCharCode(rune)).toList(growable: false);
  }

  static bool _excludedFromTrie(String token) {
    return token == '[PAD]' || token == '[CLS]' || token == '[SEP]' || token == '[UNK]' || token == '[MASK]';
  }

  static Map<String, int> _readSpecialTokenIds(Map<String, dynamic> json) {
    final specialTokenIds = <String, int>{};

    final addedTokens = json['added_tokens'] as List<dynamic>? ?? const <dynamic>[];
    for (final entry in addedTokens) {
      if (entry is Map<String, dynamic>) {
        final content = entry['content']?.toString();
        final id = (entry['id'] as num?)?.toInt();
        if (content != null && id != null) {
          specialTokenIds[content] = id;
        }
      }
    }

    final addedTokensDecoder = json['added_tokens_decoder'];
    if (addedTokensDecoder is Map) {
      for (final entry in addedTokensDecoder.entries) {
        final id = int.tryParse(entry.key.toString());
        final value = entry.value;
        if (id == null || value is! Map) {
          continue;
        }

        final content = value['content']?.toString();
        if (content != null) {
          specialTokenIds[content] = id;
        }
      }
    }

    return specialTokenIds;
  }
}

class _TrieNode {
  final Map<String, _TrieNode> children = <String, _TrieNode>{};
  int? tokenId;
  double score = 0.0;

  void insert(String token, int tokenId, double score) {
    var node = this;
    for (final rune in token.runes.map((value) => String.fromCharCode(value))) {
      node = node.children.putIfAbsent(rune, () => _TrieNode());
    }
    node.tokenId = tokenId;
    node.score = score;
  }
}