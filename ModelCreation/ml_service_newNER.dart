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

  // BERT special token ids — looked up from vocab.txt in initialize().
  // (Hardcoding these is a bug: different BERT vocabs use different ids.)
  int _padId = 0;
  int _unkId = 100;
  int _clsId = 101;
  int _sepId = 102;
  static const int _maxLength = 384;
  static const int _maxWordChars = 100;
  static const double _minEntityScore = 0.35;

  // Fields trusted from regex over the model (mirror of entity_decoder.py).
  static const Set<String> _regexPrimary = {
    'date', 'issue_date', 'start_date', 'end_date', 'delivery_date', 'due_date',
    'birth_date', 'time', 'total', 'subtotal', 'vat', 'vat_rate',
    'invoice_number', 'receipt_number', 'order_number', 'customer_number',
    'delivery_number', 'currency', 'email', 'phone', 'fax', 'website',
    'iban', 'bic', 'vat_id', 'company_register_number', 'insurance_number',
    'payment_method', 'insurer', 'bank_name',
  };

  // Address-style fields: choose the longer of NER vs regex (most complete).
  static const Set<String> _addressFields = {
    'address', 'customer_address', 'doctor_address',
  };

  // Fields whose captured value is a date (spaces normalised out).
  static const Set<String> _dateFields = {
    'date', 'issue_date', 'start_date', 'end_date', 'delivery_date',
    'due_date', 'birth_date',
  };

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

    // Resolve special-token ids from the actual vocab (robust across models).
    _padId = _vocab['[PAD]'] ?? 0;
    _unkId = _vocab['[UNK]'] ?? 100;
    _clsId = _vocab['[CLS]'] ?? 101;
    _sepId = _vocab['[SEP]'] ?? 102;

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

    // 7. BIO -> spans (tracking the averaged confidence per span)
    final spans = <NEREntitySpan>[];
    final spanScores = <double>[];
    String? curType;
    int? curStart;
    int? curEnd;
    double curScoreSum = 0.0;
    int curScoreCnt = 0;

    void flush() {
      if (curType != null && curStart != null && curEnd != null) {
        spans.add(NEREntitySpan(
          type: curType!,
          text: text.substring(curStart!, curEnd!),
          start: curStart!,
          end: curEnd!,
        ));
        spanScores.add(curScoreCnt > 0 ? curScoreSum / curScoreCnt : 0.0);
      }
      curType = null;
      curStart = null;
      curEnd = null;
      curScoreSum = 0.0;
      curScoreCnt = 0;
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
        curScoreSum = score;
        curScoreCnt = 1;
      } else if (tag.startsWith('I-')) {
        final et = tag.substring(2);
        if (curType == et) {
          curEnd = gs.end;
          curScoreSum += score;
          curScoreCnt++;
        } else {
          flush();
          curType = et;
          curStart = gs.start;
          curEnd = gs.end;
          curScoreSum = score;
          curScoreCnt = 1;
        }
      }
    }
    flush();

    // 8. Best NER span per field (highest score, then longest) — never concat.
    final bestNer = <String, String>{};
    final bestNerScore = <String, double>{};
    for (var i = 0; i < spans.length; i++) {
      final s = spans[i];
      if (!allowedFields.contains(s.type)) continue;
      final sc = spanScores[i];
      final prev = bestNerScore[s.type];
      if (prev == null ||
          sc > prev ||
          (sc == prev && s.text.length > (bestNer[s.type]?.length ?? 0))) {
        bestNer[s.type] = _clean(s.text);
        bestNerScore[s.type] = sc;
      }
    }

    // 9. Merge NER + regex per field (regex-primary for deterministic fields).
    final felder = <String, String>{};
    for (final field in allowedFields) {
      final regexVal = _regexFallback(field, text);
      final nerVal = bestNer[field] ?? '';
      String chosen;
      if (_addressFields.contains(field)) {
        // Prefer the more complete address (NER may stop early on line breaks).
        chosen = regexVal.length > nerVal.length ? regexVal : nerVal;
      } else if (_regexPrimary.contains(field)) {
        chosen = regexVal.isNotEmpty ? regexVal : nerVal;
      } else {
        chosen = nerVal.isNotEmpty ? nerVal : regexVal;
      }
      if (chosen.isEmpty) continue;
      if (field == 'total' || field == 'subtotal' || field == 'vat') {
        chosen = _normalizeAmount(chosen);
      } else if (_dateFields.contains(field)) {
        chosen = _normalizeDate(chosen);
      } else if (field == 'currency' && chosen == '€') {
        chosen = 'EUR';
      }
      felder[field] = chosen;
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

  String _clean(String value) {
    var v = value.replaceAll(RegExp(r'\s+'), ' ').trim();
    v = v.replaceAll(RegExp(r'^[.,:;\-]+|[.,:;\-]+$'), '').trim();
    return v;
  }

  String _normalizeAmount(String value) {
    var v = value.replaceAll(' ', '');
    if (v.contains(',')) {
      v = v.replaceAll('.', '').replaceAll(',', '.');
    }
    return v;
  }

  String _normalizeDate(String value) {
    // 28. 12. 2026 -> 28.12.2026 (strip spaces around separators).
    return value
        .replaceAllMapped(RegExp(r'\s*([./\-])\s*'), (m) => m.group(1)!)
        .trim();
  }

  static const Map<String, String> _defaultFieldDisplayDe = {
    'company': 'firma',
    'customer': 'kunde',
    'address': 'adresse',
    'customer_address': 'kundenadresse',
    'date': 'datum',
    'time': 'uhrzeit',
    'due_date': 'faelligkeitsdatum',
    'delivery_date': 'lieferdatum',
    'total': 'gesamtbetrag',
    'subtotal': 'nettobetrag',
    'vat': 'mwst_betrag',
    'vat_rate': 'mwst_satz',
    'currency': 'waehrung',
    'invoice_number': 'rechnungsnummer',
    'receipt_number': 'belegnummer',
    'order_number': 'bestellnummer',
    'customer_number': 'kundennummer',
    'delivery_number': 'lieferscheinnummer',
    'payment_method': 'zahlungsart',
    'iban': 'iban',
    'bic': 'bic',
    'bank_name': 'bank',
    'vat_id': 'uid',
    'company_register_number': 'firmenbuchnummer',
    'phone': 'telefon',
    'fax': 'fax',
    'email': 'email',
    'website': 'webseite',
    'contact_person': 'ansprechpartner',
    'patient': 'patient',
    'birth_date': 'geburtsdatum',
    'insurer': 'versicherungstraeger',
    'insurance_number': 'versicherungsnummer',
    'doctor': 'arzt',
    'doctor_address': 'arztadresse',
    'start_date': 'startdatum',
    'end_date': 'enddatum',
    'issue_date': 'ausstellungsdatum',
    'diagnosis': 'diagnose',
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

  // Dates allow optional spaces around separators (OCR: "28. 12. 2026").
  static const String _datePat =
      r'(\d{1,2}\s*[./\-]\s*\d{1,2}\s*[./\-]\s*\d{2,4})';
  static const String _amountPat =
      r'(\d{1,3}(?:[.\s]\d{3})*[,.]\d{2}|\d+[,.]\d{2})';
  static const String _ratePat = r'(\d{1,2}\s*%)';

  // Reusable umlaut character-class fragments.
  static const String _u = '\u00c4\u00d6\u00dc'; // ÄÖÜ
  static const String _l = '\u00e4\u00f6\u00fc\u00df'; // äöüß

  /// Ordered German/Austrian patterns per field — first match wins.
  /// Mirrors entity_decoder.py `_PATTERNS`.
  List<RegExp> _patternsFor(String field) {
    RegExp r(String p) => RegExp(p, caseSensitive: false, dotAll: true);
    switch (field) {
      case 'date':
        return [r('(?:rechnungsdatum|belegdatum|datum)\\s*[:/]?\\s*$_datePat')];
      case 'issue_date':
        return [
          r('ausstellungsdatum\\s*[:/]?\\s*$_datePat'),
          r('ausgestellt\\s*am\\s*[:/]?\\s*$_datePat'),
        ];
      case 'start_date':
        return [
          r('arbeitsunf[${_l}a]hig\\s*von\\s*[:/]?\\s*$_datePat'),
          r('pflege(?:freistellung)?\\s*(?:notwendig\\s*)?von\\s*[:/]?\\s*$_datePat'),
          r('\\bab\\s+$_datePat'),
        ];
      case 'end_date':
        return [
          r('letzter\\s*tag\\s*der\\s*.{0,40}?arbeitsunf[${_l}a]higkeit\\s*[:/]?\\s*$_datePat'),
          r('\\bbis\\s+$_datePat'),
          r('voraussichtliches?\\s*ende\\s*.{0,40}?$_datePat'),
        ];
      case 'delivery_date':
        return [r('lieferdatum\\s*(?:/\\s*date)?\\s*[:/]?\\s*$_datePat')];
      case 'due_date':
        return [
          r('(?:zahlbar\\s*bis|f[${_l}a]llig(?:keitsdatum)?(?:\\s*am)?|zahlungsziel)\\s*[:/]?\\s*$_datePat'),
        ];
      case 'birth_date':
        return [
          r('(?:geburtsdatum|geboren\\s*am|geb\\.?)\\s*[:/]?\\s*$_datePat'),
        ];
      case 'time':
        return [r('(?:uhrzeit|zeit)\\s*[:.]?\\s*(\\d{1,2}[:.]\\d{2})')];
      case 'total':
        return [
          r('(?:gesamtbetrag|rechnungsbetrag|zu\\s*zahlen)\\s*[:.]?\\s*(?:EUR\\s*)?$_amountPat'),
          r('summe\\s+EUR\\s+$_amountPat'),
          r('(?:gesamt|summe|total|bezahlt|gegeben)\\s*[:.]?\\s*(?:EUR\\s*)?$_amountPat'),
        ];
      case 'subtotal':
        return [
          r('(?:zwischensumme|nettobetrag|netto|entgelt)\\s*[:.]?\\s*(?:EUR\\s*)?$_amountPat'),
        ];
      case 'vat':
        return [
          r('(?:mwst|ust|umsatzsteuer)\\s*(?:\\d{1,2}\\s*%)?\\s*von\\s+$_amountPat\\s*=\\s*$_amountPat'),
          r('(?:mwst|ust|umsatzsteuer|vat)(?:-?\\s*betrag)?\\s*(?:\\d{1,2}\\s*%)?\\s*[:.=]?\\s*(?:EUR\\s*)?$_amountPat'),
        ];
      case 'vat_rate':
        return [
          r('(?:steuersatz|mwst-?satz|ust-?satz)\\s*[:.]?\\s*$_ratePat'),
          r('(?:mwst|ust|umsatzsteuer)\\s*[:.]?\\s*$_ratePat'),
          r('$_ratePat\\s*(?:mwst|ust)'),
        ];
      case 'invoice_number':
        return [
          r('(?:rechnungs-?\\s*nr|rechnungsnummer|re-?nr)\\.?\\s*[:.]?\\s*([A-Za-z0-9][\\w./\\-]{2,28})'),
        ];
      case 'receipt_number':
        return [
          r('(?:beleg-?\\s*nr|beleg\\s*nr|bon-?nr|kassenbon\\s*nr)\\.?\\s*[:.]?\\s*([A-Za-z0-9][\\w./\\-]{2,28})'),
        ];
      case 'order_number':
        return [
          r('(?:bestellnummer|bestell-?nr|auftragsnummer|auftrags-?nr)\\.?\\s*[:.]?\\s*([A-Za-z0-9][\\w./\\-]{2,28})'),
        ];
      case 'customer_number':
        return [
          r('(?:kundennummer|kunden-?nr|kd-?nr)\\.?\\s*[:.]?\\s*([A-Za-z0-9][\\w./\\-]{2,28})'),
        ];
      case 'delivery_number':
        return [
          r('(?:lieferschein-?\\s*nr|lieferschein\\s*nr|lieferschein|ls)\\.?\\s*[:.#]?\\s*([A-Za-z0-9][\\w./\\-]{2,24})'),
        ];
      case 'currency':
        return [r('\\b(EUR|USD|GBP|CHF|\u20ac)\\b')];
      case 'payment_method':
        return [
          r('\\b(MasterCard|Maestro|Bankomat|Kreditkarte|EC-Karte|\u00dcberweisung|PayLife|Visa|Bar)\\b'),
        ];
      case 'email':
        return [r('([A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,})')];
      case 'phone':
        return [
          r('(?:tel\\.?|telefon|fon)\\s*[:.]?\\s*((?:\\+?\\d[\\d\\s./\\-]{6,20}\\d))'),
        ];
      case 'fax':
        return [
          r('(?:fax|telefax)\\s*[:.]?\\s*((?:\\+?\\d[\\d\\s./\\-]{6,20}\\d))'),
        ];
      case 'website':
        return [r('\\b((?:https?://)?www\\.[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,})\\b')];
      case 'iban':
        return [r('\\b(AT\\d{2}(?:\\s?\\d{4}){4}|AT\\d{18}|DE\\d{20})\\b')];
      case 'bic':
        return [
          r('(?:bic|swift)\\s*[:.]?\\s*([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)'),
          r('\\b([A-Z]{4}AT[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\\b'),
        ];
      case 'bank_name':
        return [
          r('\\b(Raiffeisenlandesbank|Raiffeisenbank|Erste\\s*Bank|BAWAG\\s*P\\.?S\\.?K\\.?|Bank\\s*Austria|Sparkasse|Volksbank|Oberbank|Hypo\\s*Tirol\\s*Bank)\\b'),
          r('(?:bank|kreditinstitut)\\s*[:.]?\\s*([A-Z$_u][\\w$_u$_l.\\- ]{2,28}?)(?:\\n|\$)'),
        ];
      case 'vat_id':
        return [
          r('\\b(ATU\\s?\\d{8})\\b'),
          r('(?:uid(?:-?nr)?|ust-?id)\\.?\\s*[:.]?\\s*(ATU?\\s?\\d{6,9})'),
        ];
      case 'company_register_number':
        return [
          r('(?:firmenbuchnummer|firmenbuch-?nr)\\.?\\s*[:.]?\\s*(FN\\s?\\d{4,7}\\s?[a-z]?)'),
          r('\\b(FN\\s?\\d{4,7}\\s?[a-z]?)\\b'),
        ];
      case 'insurance_number':
        return [
          r('(?:versicherungsnummer|sv-?nr|svnr)\\s*[:.]?\\s*\\n*\\s*(\\d{4}\\s*\\d{2}\\s*\\d{2}\\s*\\d{2})'),
          r('\\b(\\d{4}\\s\\d{2}\\s\\d{2}\\s\\d{2})\\b'),
        ];
      case 'insurer':
        const region =
            r'(?:\s+(?:Wien|Nieder[\u00f6o]sterreich|Ober[\u00f6o]sterreich|Steiermark|Tirol|K[\u00e4a]rnten|Salzburg|Vorarlberg|Burgenland))?';
        return [
          r('(?:versicherungstr[${_l}a]ger|kasse)\\s*[:.]?\\s*((?:\u00d6GK|SVS|BVAEB|KFA)$region)'),
          r('\\b((?:\u00d6GK|SVS|BVAEB|KFA)$region)\\b'),
        ];
      case 'patient':
        return [
          r('familienname,?\\s*vorname\\(?n?\\)?\\s*[:]?\\s*\\n*\\s*([A-Z$_u][\\w$_l]+\\s+[A-Z$_u][\\w$_l]+)'),
          r('(?:patient(?:/in|in)?|name\\s*der/des\\s*erkrankten|erkrankte\\s*person)\\s*[:]?\\s*([A-Z$_u][\\w$_l]+\\s+[A-Z$_u][\\w$_l]+)'),
        ];
      case 'doctor':
        return [
          r('((?:Univ\\.-Prof\\.\\s*)?Dr\\.?(?:\\s*med\\.?)?\\s+[A-Z$_u][\\w$_l]+(?:\\s*&\\s*Dr\\.?\\s+[A-Z$_u][\\w$_l]+)?)'),
        ];
      case 'diagnosis':
        return [
          r('(?:diagnose|befund|grund)\\s*[:]?\\s*([A-Z$_u][\\w$_l/\\- ]{2,40}?)(?:\\n|\$)'),
          r('\\b(Krankheit/Unfall|Berufskrankheit|Krankheit|Unfall)\\b'),
        ];
      case 'company':
        return [
          r('([A-Z$_u][\\w$_u$_l.\\-]*(?:\\s+[A-Z$_u&][\\w$_u$_l.\\-]*){0,3}\\s+(?:AG|GmbH|GesmbH|KG|OG|e\\.U\\.|GmbH\\s*&\\s*Co\\s*KG))'),
        ];
      case 'customer':
        return [
          r('(?:empf[${_l}a]nger|kunde|rechnungsempf[${_l}a]nger)\\s*[:]?\\s*\\n*\\s*([A-Z$_u][\\w$_u$_l.&\\-]+(?:\\s+[A-Z$_u&][\\w$_u$_l.\\-]+){0,3})'),
        ];
      case 'contact_person':
        return [
          r('(?:ansprechpartner|kontaktperson|kontakt)\\s*[:]?\\s*([A-Z$_u][\\w$_l]+\\s+[A-Z$_u][\\w$_l]+)'),
        ];
      case 'address':
      case 'customer_address':
      case 'doctor_address':
        return [
          r('\\b(\\d{4}\\s+[A-Z$_u][A-Za-z$_u$_l.\\-]+(?:\\s+[A-Z$_u][A-Za-z$_u$_l.\\-]+)*\\s+\\d{1,4})'),
          r('\\b(\\d{4}\\s+[A-Z$_u][A-Za-z$_u$_l.\\- ]{3,40})'),
        ];
      default:
        return const [];
    }
  }

  String _regexFallback(String field, String text) {
    for (final pattern in _patternsFor(field)) {
      final m = pattern.firstMatch(text);
      if (m == null) continue;
      // Use the last captured group (mirrors Python m.group(m.lastindex)).
      String? val;
      for (var g = m.groupCount; g >= 1; g--) {
        final cand = m.group(g);
        if (cand != null && cand.isNotEmpty) {
          val = cand;
          break;
        }
      }
      val ??= m.group(0);
      final cleaned = _clean(val ?? '');
      if (cleaned.isNotEmpty) return cleaned;
    }
    return '';
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
//     - assets/models/model.onnx
//     - assets/models/model_meta.json
//     - assets/models/tokenizer/vocab.txt
// -----------------------------------------------------------------------------
