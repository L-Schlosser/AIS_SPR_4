import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class SmolResult {
  final Map<String, dynamic> json;

  SmolResult({required this.json});
}

class MLServiceSmolLM {
  bool _isInitialized = false;
  String? _modelPath;

  Future<void> initialize() async {
    if (_isInitialized) return;

    _modelPath = await _copyModelToFile();

    _isInitialized = true;
  }

  Future<SmolResult> extract(String text) async {
    if (!_isInitialized || _modelPath == null) {
      throw Exception("SmolLM not initialized");
    }

    final llama = Llama(_modelPath!);

    llama.setPrompt(_buildPrompt(text));

    final buffer = StringBuffer();

    while (true) {
      final (token, done) = llama.getNext();
      buffer.write(token);

      if (done) break;
    }

    llama.dispose();

    final cleaned = _extractJson(buffer.toString());

    return SmolResult(
      json: jsonDecode(cleaned),
    );
  }

  // ---------------------------------------------------------------------------
  // Prompt
  // ---------------------------------------------------------------------------

  String _buildPrompt(String text) {
    return """
You are an information extraction system.

Return ONLY valid JSON.

No markdown.
No explanation.

Schema:
{
  "document_type": null,
  "patient_name": null,
  "insurance_provider": null,
  "insurance_number": null,
  "address": null,
  "start_date": null,
  "end_date": null,
  "doctor": null,
  "issue_date": null
}

OCR TEXT:
$text
""";
  }

  // ---------------------------------------------------------------------------
  // JSON extraction
  // ---------------------------------------------------------------------------

  String _extractJson(String raw) {
    final start = raw.indexOf('{');
    final end = raw.lastIndexOf('}');

    if (start == -1 || end == -1) {
      throw Exception("Invalid model output:\n$raw");
    }

    return raw.substring(start, end + 1);
  }

  // ---------------------------------------------------------------------------
  // Model file handling
  // ---------------------------------------------------------------------------

  Future<String> _copyModelToFile() async {
    final data = await rootBundle.load('assets/models/smollm.gguf');

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/smollm.gguf');

    await file.writeAsBytes(data.buffer.asUint8List(), flush: true);

    return file.path;
  }

  void dispose() {}
}