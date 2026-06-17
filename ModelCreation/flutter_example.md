# Flutter Android Integration — Document Field Extraction

This guide shows production-ready integration of the ONNX NER model using the [`onnxruntime`](https://pub.dev/packages/onnxruntime) package.

## 1. Add Assets

Copy from `models/onnx/` into your Flutter project:

```
assets/models/
  model.onnx
  tokenizer/
    vocab.txt
    tokenizer_config.json
    tokenizer.json
    special_tokens_map.json
```

`pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/model.onnx
    - assets/models/tokenizer/
```

## 2. Dependencies

```yaml
dependencies:
  onnxruntime: ^1.4.1
  flutter/services.dart  # for rootBundle
```

## 3. Load Model and Tokenizer

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class DocumentExtractorService {
  late OrtSession _session;
  late Map<String, int> _vocab;
  late Map<String, dynamic> _meta;
  static const int maxLength = 512;
  static const int numLabels = 43;

  Future<void> initialize() async {
    OrtEnv.instance.init();

    final modelBytes = await rootBundle.load('assets/models/model.onnx');
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(
      modelBytes.buffer.asUint8List(),
      sessionOptions,
    );

    final vocabStr = await rootBundle.loadString(
      'assets/models/tokenizer/vocab.txt',
    );
    _vocab = {
      for (var i = 0; i < vocabStr.split('\n').length; i++)
        vocabStr.split('\n')[i]: i,
    };

    final metaStr = await rootBundle.loadString(
      'assets/models/model_meta.json',
    );
    _meta = jsonDecode(metaStr);
  }

  void dispose() {
    _session.release();
    OrtEnv.instance.release();
  }
}
```

## 4. Word-Level Tokenization (DistilBERT)

DistilBERT uses WordPiece. For OCR text, split on whitespace first, then subword-tokenize each word:

```dart
List<int> tokenizeWords(List<String> words) {
  final inputIds = <int>[_vocab['[CLS]']!];
  final wordToTokenStart = <int>[];

  for (var w = 0; w < words.length; w++) {
    wordToTokenStart.add(inputIds.length);
    final word = words[w].toLowerCase();
    // Simple greedy WordPiece (use tokenizer.json for production parity)
    if (_vocab.containsKey(word)) {
      inputIds.add(_vocab[word]!);
    } else {
      // Fallback: character-level unk split
      inputIds.add(_vocab['[UNK]']!);
    }
    if (inputIds.length >= maxLength - 1) break;
  }
  inputIds.add(_vocab['[SEP]']!);

  while (inputIds.length < maxLength) {
    inputIds.add(0); // 
  }
  return inputIds;
}

List<int> buildAttentionMask(List<int> inputIds) {
  return inputIds.map((id) => id != 0 ? 1 : 0).toList();
}
```

> **Production tip:** Bundle `tokenizer.json` and use a small Dart WordPiece implementation, or pre-tokenize on a server during development and validate token IDs against Python `inference.py`.

## 5. Run Inference

```dart
Map<String, String> extract(String documentType, String rawText) {
  final words = rawText.split(RegExp(r'\s+'));
  final inputIds = tokenizeWords(words);
  final attentionMask = buildAttentionMask(inputIds);

  final inputIdsTensor = OrtValueTensor.createTensorWithDataList(
    Int64List.fromList(inputIds),
    [1, maxLength],
  );
  final maskTensor = OrtValueTensor.createTensorWithDataList(
    Int64List.fromList(attentionMask),
    [1, maxLength],
  );

  final runOptions = OrtRunOptions();
  final outputs = _session.run(
    runOptions,
    {'input_ids': inputIdsTensor, 'attention_mask': maskTensor},
  );

  final logits = outputs[0]!.value as List; // [1, seq, numLabels]
  inputIdsTensor.release();
  maskTensor.release();
  runOptions.release();
  outputs.forEach((_, v) => v?.release());

  return _decodeOutput(documentType, rawText, words, logits[0]);
}
```

## 6. BIO Decoding → JSON

```dart
Map<String, String> _decodeOutput(
  String documentType,
  String rawText,
  List<String> words,
  List<dynamic> tokenLogits,
) {
  final id2label = Map<String, String>.from(_meta['id2label']);
  final allowedFields = List<String>.from(
    _meta['fields_by_type'][documentType] ?? [],
  );

  // Map first subword of each word to a label
  final wordTags = <String>[];
  var wordIdx = 0;
  for (var t = 1; t < tokenLogits.length && wordIdx < words.length; t++) {
    final scores = (tokenLogits[t] as List).cast<double>();
    var maxIdx = 0;
    var maxScore = scores[0];
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIdx = i;
      }
    }
    final tag = id2label['$maxIdx'] ?? 'O';
    if (tag != 'O') {
      wordTags.add(tag);
      wordIdx++;
    }
  }

  // Aggregate BIO spans
  final entities = <String, String>{};
  String? currentLabel;
  final buffer = <String>[];

  void flush() {
    if (currentLabel != null && buffer.isNotEmpty) {
      entities[currentLabel!] = buffer.join(' ');
    }
    currentLabel = null;
    buffer.clear();
  }

  for (var i = 0; i < words.length && i < wordTags.length; i++) {
    final tag = wordTags[i];
    if (tag.startsWith('B-')) {
      flush();
      currentLabel = tag.substring(2);
      buffer.add(words[i]);
    } else if (tag.startsWith('I-') && currentLabel == tag.substring(2)) {
      buffer.add(words[i]);
    } else {
      flush();
    }
  }
  flush();

  // Regex fallbacks (mirror entity_decoder.py)
  for (final field in allowedFields) {
    entities.putIfAbsent(field, () => _regexFallback(field, rawText));
  }

  return Map.fromEntries(
    entities.entries.where(
      (e) => allowedFields.contains(e.key) && e.value.isNotEmpty,
    ),
  );
}

String _regexFallback(String field, String text) {
  final patterns = <String, RegExp>{
    'date': RegExp(r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b'),
    'total': RegExp(
      r'(?:total|amount due)\s*:?\s*([\d,]+\.?\d*)',
      caseSensitive: false,
    ),
    'email': RegExp(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b'),
    'iban': RegExp(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b'),
  };
  final m = patterns[field]?.firstMatch(text);
  return m?.group(1) ?? m?.group(0) ?? '';
}
```

## 7. Usage in Widget

```dart
class InvoiceScanPage extends StatefulWidget {
  @override
  State<InvoiceScanPage> createState() => _InvoiceScanPageState();
}

class _InvoiceScanPageState extends State<InvoiceScanPage> {
  final _extractor = DocumentExtractorService();
  Map<String, String>? _fields;

  @override
  void initState() {
    super.initState();
    _extractor.initialize();
  }

  Future<void> _onOcrComplete(String ocrText, String docType) async {
    final fields = _extractor.extract(docType, ocrText);
    setState(() => _fields = fields);
  }

  @override
  void dispose() {
    _extractor.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (_fields != null)
          ..._fields!.entries.map(
            (e) => ListTile(title: Text(e.key), subtitle: Text(e.value)),
          ),
      ],
    );
  }
}
```

## 8. Integration with Existing Pipeline

Your app already produces:

```dart
final input = {
  'document_type': classifierResult, // e.g. "invoice"
  'raw_text': ocrResult,
};
```

Pass directly to the extractor:

```dart
final json = extractor.extract(
  input['document_type'] as String,
  input['raw_text'] as String,
);
// json == {"company": "...", "date": "...", "total": "...", ...}
```

## 9. Performance Tips for Android

| Tip | Detail |
|---|---|
| Load once | Initialize `OrtSession` in a singleton / isolate, not per scan |
| Thread count | `sessionOptions.setIntraOpNumThreads(2)` on mid-range devices |
| Sequence length | Truncate OCR to first 512 tokens (config `max_length`) |
| Quantization | Ship `model.onnx` INT8 (~67 MB) not FP32 |
| Isolate | Run inference in a background isolate to avoid UI jank |

## 10. Validate Against Python

```bash
python inference.py --document-type invoice --text "YOUR OCR TEXT"
```

Compare JSON output with Flutter output for the same input before release.
