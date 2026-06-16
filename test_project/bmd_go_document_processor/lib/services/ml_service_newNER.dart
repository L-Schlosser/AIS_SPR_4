import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

// -----------------------------------------------------------------------------
// Data classes (same shape as your existing MLServiceNER)
// -----------------------------------------------------------------------------

class NEREntitySpan {
  final String type; // e.g. "company", "total"
  final String text;
  final int start;
  final int end;

  NEREntitySpan({
    required this.type,
    required this.text,
    required this.start,
    required this.end,
  });

  Map<String, dynamic> toJson() =>
      {'type': type, 'text': text, 'start': start, 'end': end};
}

class NERResult {
  /// German display keys -> extracted text, e.g. {"firma": "Billa AG"}
  final Map<String, String> felder;
  final List<NEREntitySpan> spans;
  final String rawText;
  final String documentType;

  NERResult({
    required this.felder,
    required this.spans,
    required this.rawText,
    required this.documentType,
  });
}

// -----------------------------------------------------------------------------
// Service
// -----------------------------------------------------------------------------

class MLServiceNewNER {
  late OrtSession _session;
  late Map<String, int> _vocab;
  late Map<int, String> _idToLabel;
  late Map<String, List<String>> _fieldsByType;
  late Map<String, String> _fieldDisplayDe;
  bool _tokenizerCased = true;
  bool _isInitialized = false;

  // BERT special token ids (from vocab.txt line index)
  static const int _padId = 0;
  static const int _unkId = 101;
  static const int _clsId = 102;
  static const int _sepId = 103;
  static const int _maxLength = 512;
  static const int _maxWordChars = 100;
  static const double _minEntityScore = 0.35;

  static const String _modelAsset = 'assets/models/NER_Model/model.onnx';
  static const String _vocabAsset = 'assets/models/NER_Model/tokenizer/vocab.txt';
  static const String _metaAsset = 'assets/models/NER_Model/model_meta.json';

  Future<void> initialize() async {
    if (_isInitialized) return;

    final vocabText = await rootBundle.loadString(_vocabAsset);
    _vocab = <String, int>{};
    var idx = 0;
    for (final line in const LineSplitter().convert(vocabText)) {
      _vocab[line] = idx++;
    }

    final metaJson =
        jsonDecode(await rootBundle.loadString(_metaAsset)) as Map<String, dynamic>;
    _idToLabel = {};
    (metaJson['id2label'] as Map<String, dynamic>).forEach((k, v) {
      _idToLabel[int.parse(k)] = v as String;
    });
    _fieldsByType = {};
    (metaJson['fields_by_type'] as Map<String, dynamic>).forEach((k, v) {
      _fieldsByType[k] = List<String>.from(v as List);
    });
    _tokenizerCased = metaJson['tokenizer_cased'] as bool? ?? true;
    _fieldDisplayDe = {};
    if (metaJson['field_display_de'] != null) {
      (metaJson['field_display_de'] as Map<String, dynamic>).forEach((k, v) {
        _fieldDisplayDe[k] = v as String;
      });
    } else {
      _fieldDisplayDe.addAll(_defaultFieldDisplayDe);
    }

    final modelData = await rootBundle.load(_modelAsset);
    final sessionOptions = OrtSessionOptions();
    sessionOptions.setIntraOpNumThreads(2);
    _session =
        OrtSession.fromBuffer(modelData.buffer.asUint8List(), sessionOptions);

    _isInitialized = true;
  }

  /// [documentType] from your classifier:
  /// invoice | receipt | doctor_note | care_leave | delivery_note | master_data
  Future<NERResult> extract(String documentType, String text) async {
    if (!_isInitialized) {
      throw Exception('MLServiceNewNER not initialized - call initialize() first');
    }

    final allowedFields = _fieldsByType[documentType] ?? <String>[];

    // 1. Whitespace/punctuation tokenize with char offsets
    final words = _basicTokenize(text);

    // 2. Group adjacent tokens (no whitespace gap) into logical words
    final groupOfWord = _groupAdjacent(words);

    // 3. WordPiece tokenize
    final subwords = _wordPieceTokenize(words);

    // 4. [CLS] body [SEP], pad to _maxLength
    final maxBody = _maxLength - 2;
    final body =
        subwords.length > maxBody ? subwords.sublist(0, maxBody) : subwords;
    final inputIds = <int>[_clsId];
    final groupIdxPerToken = <int>[-1];
    for (final sw in body) {
      inputIds.add(sw.id);
      groupIdxPerToken.add(groupOfWord[sw.wordIndex]);
    }
    inputIds.add(_sepId);
    groupIdxPerToken.add(-1);

    final actualLen = inputIds.length;
    final attentionMask = List<int>.filled(_maxLength, 0);
    for (var i = 0; i < actualLen; i++) {
      attentionMask[i] = 1;
    }
    while (inputIds.length < _maxLength) {
      inputIds.add(_padId);
      groupIdxPerToken.add(-1);
    }

    // 5. ONNX inference
    final inputIdsTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList(inputIds),
      [1, _maxLength],
    );
    final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList(attentionMask),
      [1, _maxLength],
    );
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, <String, OrtValue>{
      'input_ids': inputIdsTensor,
      'attention_mask': attentionMaskTensor,
    });
    if (outputs == null || outputs.isEmpty) {
      throw Exception('NER produced no outputs');
    }

    // 6. Argmax + softmax confidence per token; one tag per logical group
    final raw = outputs[0]!.value as List;
    final List seq = raw.first as List;

    final groupSpans = <_GroupSpan>[];
    var curG = -1;
    for (var wi = 0; wi < words.length; wi++) {
      final g = groupOfWord[wi];
      if (g != curG) {
        groupSpans.add(_GroupSpan(words[wi].start, words[wi].end));
        curG = g;
      } else {
        groupSpans.last.end = words[wi].end;
      }
    }

    final groupTags = <int, String>{};
    final groupScores = <int, double>{};
    final seenGroups = <int>{};
    for (var t = 0; t < seq.length; t++) {
      final g = groupIdxPerToken[t];
      if (g < 0 || seenGroups.contains(g)) continue;
      seenGroups.add(g);

      final tokLogits = seq[t] as List;
      var best = 0;
      var bestVal = (tokLogits[0] as num).toDouble();
      for (var i = 1; i < tokLogits.length; i++) {
        final v = (tokLogits[i] as num).toDouble();
        if (v > bestVal) {
          bestVal = v;
          best = i;
        }
      }
      final probs = _softmax(tokLogits);
      groupTags[g] = _idToLabel[best] ?? 'O';
      groupScores[g] = probs[best];
    }

    // 7. BIO -> spans
    final spans = <NEREntitySpan>[];
    String? curType;
    int? curStart;
    int? curEnd;

    void flush() {
      if (curType != null && curStart != null && curEnd != null) {
        spans.add(NEREntitySpan(
          type: curType!,
          text: text.substring(curStart!, curEnd!),
          start: curStart!,
          end: curEnd!,
        ));
      }
      curType = null;
      curStart = null;
      curEnd = null;
    }

    for (var g = 0; g < groupSpans.length; g++) {
      final tag = groupTags[g] ?? 'O';
      final score = groupScores[g] ?? 0.0;
      if (tag == 'O' || score < _minEntityScore) {
        flush();
        continue;
      }

      final gs = groupSpans[g];
      if (tag.startsWith('B-')) {
        flush();
        curType = tag.substring(2);
        curStart = gs.start;
        curEnd = gs.end;
      } else if (tag.startsWith('I-')) {
        final et = tag.substring(2);
        if (curType == et) {
          curEnd = gs.end;
        } else {
          flush();
          curType = et;
          curStart = gs.start;
          curEnd = gs.end;
        }
      }
    }
    flush();

    // 8. Filter by document type + aggregate field map
    final felder = <String, String>{};
    for (final span in spans) {
      if (!allowedFields.contains(span.type)) continue;
      if (felder.containsKey(span.type)) {
        felder[span.type] = '${felder[span.type]!} ${span.text}';
      } else {
        felder[span.type] = span.text;
      }
    }

    // 9. Regex fallbacks for missing fields (mirrors entity_decoder.py)
    for (final field in allowedFields) {
      if (felder.containsKey(field) && felder[field]!.isNotEmpty){
        print("Found $field from model: ${felder[field]}");
        continue;
      }
      print("regexFallback!");
      final value = _regexFallback(field, text);
      print("regexFallback field: $field, value: $value");
      if (value.isNotEmpty) {
        felder[field] = value;
      }
    }

    // 10. Map internal keys -> German display keys for UI
    final felderDe = <String, String>{};
    for (final entry in felder.entries) {
      final displayKey = _fieldDisplayDe[entry.key] ?? entry.key;
      felderDe[displayKey] = entry.value;
    }

    return NERResult(
      felder: felderDe,
      spans: spans.where((s) => allowedFields.contains(s.type)).toList(),
      rawText: text,
      documentType: documentType,
    );
  }

  static const Map<String, String> _defaultFieldDisplayDe = {
    'company': 'firma',
    'address': 'adresse',
    'date': 'datum',
    'total': 'gesamtbetrag',
    'subtotal': 'zwischensumme',
    'vat': 'mwst_betrag',
    'invoice_number': 'rechnungsnummer',
    'currency': 'waehrung',
    'patient': 'patient',
    'doctor': 'arzt',
    'start_date': 'startdatum',
    'end_date': 'enddatum',
    'issue_date': 'ausstellungsdatum',
    'diagnosis': 'diagnose',
    'delivery_date': 'lieferdatum',
    'delivery_number': 'lieferscheinnummer',
    'customer': 'kunde',
    'phone': 'telefon',
    'email': 'email',
    'iban': 'iban',
    'vat_id': 'uid',
  };

  bool get isInitialized => _isInitialized;

  void dispose() => _session.release();

  // ---------------------------------------------------------------------------
  // Tokenization (same approach as your MLServiceNER)
  // ---------------------------------------------------------------------------

  List<_WordSpan> _basicTokenize(String text) {
    final spans = <_WordSpan>[];
    final buf = StringBuffer();
    var wordStart = -1;

    for (var i = 0; i < text.length; i++) {
      final ch = text.codeUnitAt(i);
      final isWS = ch == 0x20 ||
          ch == 0x09 ||
          ch == 0x0A ||
          ch == 0x0D ||
          ch == 0x00A0 ||
          ch == 0x2028 ||
          ch == 0x2029;
      final isPunct = _isPunctuation(ch);

      if (isWS) {
        if (buf.isNotEmpty) {
          spans.add(_WordSpan(buf.toString(), wordStart, i));
          buf.clear();
        }
        wordStart = -1;
      } else if (isPunct) {
        if (buf.isNotEmpty) {
          spans.add(_WordSpan(buf.toString(), wordStart, i));
          buf.clear();
        }
        spans.add(_WordSpan(text[i], i, i + 1));
        wordStart = -1;
      } else {
        if (wordStart < 0) wordStart = i;
        buf.write(text[i]);
      }
    }
    if (buf.isNotEmpty) {
      spans.add(_WordSpan(buf.toString(), wordStart, text.length));
    }
    return spans;
  }

  List<int> _groupAdjacent(List<_WordSpan> words) {
    final groups = List<int>.filled(words.length, 0);
    var g = 0;
    for (var i = 0; i < words.length; i++) {
      if (i > 0 && words[i].start != words[i - 1].end) {
        g++;
      }
      groups[i] = g;
    }
    return groups;
  }

  bool _isPunctuation(int ch) {
    return (ch >= 33 && ch <= 47) ||
        (ch >= 58 && ch <= 64) ||
        (ch >= 91 && ch <= 96) ||
        (ch >= 123 && ch <= 126);
  }

  List<_Subword> _wordPieceTokenize(List<_WordSpan> words) {
    final result = <_Subword>[];
    for (var wi = 0; wi < words.length; wi++) {
      final word = _tokenizerCased ? words[wi].text : words[wi].text.toLowerCase();
      if (word.length > _maxWordChars) {
        result.add(_Subword(_unkId, wi));
        continue;
      }
      var bad = false;
      var start = 0;
      final subIds = <int>[];
      while (start < word.length) {
        var end = word.length;
        int? curId;
        while (start < end) {
          var sub = word.substring(start, end);
          if (start > 0) sub = '##$sub';
          final id = _vocab[sub];
          if (id != null) {
            curId = id;
            break;
          }
          end--;
        }
        if (curId == null) {
          bad = true;
          break;
        }
        subIds.add(curId);
        start = end;
      }
      if (bad || subIds.isEmpty) {
        result.add(_Subword(_unkId, wi));
      } else {
        for (final id in subIds) {
          result.add(_Subword(id, wi));
        }
      }
    }
    return result;
  }

  List<double> _softmax(List logits) {
    var maxVal = (logits[0] as num).toDouble();
    for (var i = 1; i < logits.length; i++) {
      final v = (logits[i] as num).toDouble();
      if (v > maxVal) maxVal = v;
    }
    final exps = List<double>.generate(
      logits.length,
      (i) => exp((logits[i] as num).toDouble() - maxVal),
    );
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  // ---------------------------------------------------------------------------
  // Regex fallbacks (optional safety net, same idea as Python entity_decoder)
  // ---------------------------------------------------------------------------

  String _regexFallback(String field, String text) {
    const datePat =
        r'(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{4}[./\-]\d{1,2}[./\-]\d{1,2})';
    const amountPat = r'([\d]{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2})';

    RegExp? pattern;
    switch (field) {
      case 'date':
      case 'issue_date':
        pattern = RegExp(
          '(?:datum|rechnungsdatum|ausstellungsdatum)\\s*:?\\s*$datePat',
          caseSensitive: false,
        );
        final m = pattern.firstMatch(text);
        if (m != null) return m.group(1) ?? '';
        pattern = RegExp('\\b$datePat\\b');
        break;
      case 'start_date':
        pattern = RegExp('(?:from|ab|vom)\\s+$datePat', caseSensitive: false);
        break;
      case 'end_date':
        pattern = RegExp('(?:until|to|bis)\\s+$datePat', caseSensitive: false);
        break;
      case 'delivery_date':
        pattern = RegExp('(?:lieferdatum)\\s*:?\\s*$datePat', caseSensitive: false);
        break;
      case 'total':
        pattern = RegExp('Summe\\s+EUR\\s+$amountPat', caseSensitive: false);
        final m = pattern.firstMatch(text);
        if (m != null) return (m.group(1) ?? '').replaceAll(',', '.');
        pattern = RegExp(
          '(?:summe|gesamt|gesamtbetrag|bezahlt)\\s*(?:EUR\\s*)?$amountPat',
          caseSensitive: false,
          dotAll: true,
        );
        break;
      case 'subtotal':
        pattern = RegExp(
          '(?:zwischensumme|nettobetrag|netto)\\s*:?\\s*$amountPat',
          caseSensitive: false,
        );
        break;
      case 'vat':
        pattern = RegExp(
          '(?:mwst|ust|vat)\\s*(?:betrag)?\\s*(?:von\\s+$amountPat\\s*=\\s*)?($amountPat)',
          caseSensitive: false,
        );
        break;
      case 'invoice_number':
        pattern = RegExp(
          '(?:beleg\\s*nr\\.?|re-nr\\.?|rechnungsnummer|bon-nr)\\s*:?\\s*([\\w./-]{3,30})',
          caseSensitive: false,
        );
        break;
      case 'delivery_number':
        pattern = RegExp(
          '(?:lieferschein|ls[\\s.#-]*)\\s*:?\\s*([A-Z0-9][\\w./-]{2,20})',
          caseSensitive: false,
        );
        break;
      case 'currency':
        pattern = RegExp(r'\b(EUR|USD|GBP|CHF)\b');
        break;
      case 'email':
        pattern = RegExp(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b');
        break;
      case 'phone':
        pattern = RegExp(
          r'(?:TEL|Tel|Telefon)\s*:?\s*([+\d][\d\s./-]{6,20})',
          caseSensitive: false,
        );
        break;
      case 'iban':
        pattern = RegExp(r'\b(AT\d{18,20}|DE\d{20,22}|[A-Z]{2}\d{2}[A-Z0-9]{11,30})\b');
        break;
      case 'vat_id':
        pattern = RegExp(
          r'\b((?:ATU|UID|UST[\s.-]?ID)\s*[\dA-Z]{6,14})\b',
          caseSensitive: false,
        );
        break;
      case 'company':
        pattern = RegExp(
          r'\b([A-ZÄÖÜ][\wÄÖÜäöüß.&-]+(?:\s+(?:AG|GmbH|KG|OG|e\.U\.|GesmbH))+)\b',
        );
        break;
      case 'address':
        pattern = RegExp(
          r'\b(\d{4}\s+[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß\s.-]+(?:\d{1,4})?)\b',
        );
        break;
    }
    if (pattern == null) return '';
    final m = pattern.firstMatch(text);
    if (m == null) return '';
    final val = (m.groupCount >= 1 ? m.group(1) : m.group(0)) ?? '';
    return field == 'total' ? val.replaceAll(',', '.') : val;
  }
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------

class _WordSpan {
  final String text;
  final int start;
  final int end;
  _WordSpan(this.text, this.start, this.end);
}

class _Subword {
  final int id;
  final int wordIndex;
  _Subword(this.id, this.wordIndex);
}

class _GroupSpan {
  int start;
  int end;
  _GroupSpan(this.start, this.end);
}

// -----------------------------------------------------------------------------
// USAGE EXAMPLE — copy into your scan/classification flow:
//
//   print('Extract features');
//   final nerExtractor = MLServiceNewNER();
//   await nerExtractor.initialize();
//
//   // documentType comes from your existing classifier, e.g. "invoice"
//   final extractedInfos = await nerExtractor.extract(
//     classificationResult.documentType,  // "invoice" | "receipt" | ...
//     extractedText,
//   );
//
//   classificationResult.infos.addAll(extractedInfos.felder);
//
//   // Example felder for receipt (German UI keys):
//   // {
//   //   "firma": "Billa AG",
//   //   "adresse": "1010 WIEN FRANZ JOSEFS KAI 29",
//   //   "datum": "28.09.2022",
//   //   "gesamtbetrag": "8.84",
//   //   "waehrung": "EUR"
//   // }
//
//   nerExtractor.dispose();
//
// pubspec.yaml assets:
//   assets:
//     - assets/models/NER_Model/model.onnx
//     - assets/models/NER_Model/model_meta.json
//     - assets/models/NER_Model/tokenizer/vocab.txt
// -----------------------------------------------------------------------------
