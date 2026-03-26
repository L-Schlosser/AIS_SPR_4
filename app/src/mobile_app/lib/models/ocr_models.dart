import 'dart:ui';

/// Represents one recognized word/element from OCR.
class OcrWord {
  final String text;
  final Rect boundingBox;

  const OcrWord({
    required this.text,
    required this.boundingBox,
  });

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'bounding_box': rectToJson(boundingBox),
    };
  }

  factory OcrWord.fromJson(Map<String, dynamic> json) {
    return OcrWord(
      text: json['text'] as String? ?? '',
      boundingBox: rectFromJson(json['bounding_box'] as Map<String, dynamic>?),
    );
  }
}

/// Represents one OCR line, containing multiple words/elements.
class OcrLine {
  final String text;
  final Rect boundingBox;
  final List<OcrWord> words;

  const OcrLine({
    required this.text,
    required this.boundingBox,
    required this.words,
  });

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'bounding_box': rectToJson(boundingBox),
      'words': words.map((word) => word.toJson()).toList(),
    };
  }

  factory OcrLine.fromJson(Map<String, dynamic> json) {
    final wordsJson = json['words'] as List<dynamic>? ?? const [];

    return OcrLine(
      text: json['text'] as String? ?? '',
      boundingBox: rectFromJson(json['bounding_box'] as Map<String, dynamic>?),
      words: wordsJson
          .map((wordJson) => OcrWord.fromJson(wordJson as Map<String, dynamic>))
          .toList(),
    );
  }
}

/// Represents one OCR block, containing multiple lines.
class OcrBlock {
  final String text;
  final Rect boundingBox;
  final List<OcrLine> lines;

  const OcrBlock({
    required this.text,
    required this.boundingBox,
    required this.lines,
  });

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'bounding_box': rectToJson(boundingBox),
      'lines': lines.map((line) => line.toJson()).toList(),
    };
  }

  factory OcrBlock.fromJson(Map<String, dynamic> json) {
    final linesJson = json['lines'] as List<dynamic>? ?? const [];

    return OcrBlock(
      text: json['text'] as String? ?? '',
      boundingBox: rectFromJson(json['bounding_box'] as Map<String, dynamic>?),
      lines: linesJson
          .map((lineJson) => OcrLine.fromJson(lineJson as Map<String, dynamic>))
          .toList(),
    );
  }
}

/// Full OCR result for one processed document/image.
class OcrDocument {
  final String rawText;
  final List<OcrBlock> blocks;

  const OcrDocument({
    required this.rawText,
    required this.blocks,
  });

  List<OcrLine> get allLines =>
      blocks.expand((block) => block.lines).toList(growable: false);

  List<OcrWord> get allWords => allLines
      .expand((line) => line.words)
      .toList(growable: false);

  Map<String, dynamic> toJson() {
    return {
      'raw_text': rawText,
      'blocks': blocks.map((block) => block.toJson()).toList(),
    };
  }

  factory OcrDocument.fromJson(Map<String, dynamic> json) {
    final blocksJson = json['blocks'] as List<dynamic>? ?? const [];

    return OcrDocument(
      rawText: json['raw_text'] as String? ?? '',
      blocks: blocksJson
          .map((blockJson) => OcrBlock.fromJson(blockJson as Map<String, dynamic>))
          .toList(),
    );
  }
}

/// Converts a Rect into a JSON-friendly map.
Map<String, dynamic> rectToJson(Rect rect) {
  return {
    'left': rect.left,
    'top': rect.top,
    'right': rect.right,
    'bottom': rect.bottom,
    'width': rect.width,
    'height': rect.height,
  };
}

/// Builds a Rect from JSON. Falls back to Rect.zero if data is missing.
Rect rectFromJson(Map<String, dynamic>? json) {
  if (json == null) return Rect.zero;

  final left = (json['left'] as num?)?.toDouble() ?? 0.0;
  final top = (json['top'] as num?)?.toDouble() ?? 0.0;

  double? right = (json['right'] as num?)?.toDouble();
  double? bottom = (json['bottom'] as num?)?.toDouble();

  final width = (json['width'] as num?)?.toDouble();
  final height = (json['height'] as num?)?.toDouble();

  right ??= left + (width ?? 0.0);
  bottom ??= top + (height ?? 0.0);

  return Rect.fromLTRB(left, top, right, bottom);
}
