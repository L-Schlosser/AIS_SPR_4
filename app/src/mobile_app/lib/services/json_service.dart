import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../models/structured_document.dart';

/// Service responsible for:
/// - converting a [StructuredDocument] into formatted JSON
/// - saving JSON to a file
/// - optionally returning the JSON string for UI/debugging
class JsonService {
  const JsonService();

  /// Converts a [StructuredDocument] into a pretty-printed JSON string.
  String toPrettyJson(StructuredDocument document) {
    return const JsonEncoder.withIndent('  ').convert(document.toJson());
  }

  /// Converts a [StructuredDocument] into a compact JSON string.
  String toCompactJson(StructuredDocument document) {
    return jsonEncode(document.toJson());
  }

  /// Saves the structured document JSON into the app's documents directory.
  ///
  /// Example output path:
  /// /data/user/0/<package>/app_flutter/ocr_result.json
  Future<File> saveToDocumentsDirectory(
    StructuredDocument document, {
    String fileName = 'ocr_result.json',
  }) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$fileName');
    final jsonString = toPrettyJson(document);

    return file.writeAsString(jsonString, flush: true);
  }

  /// Saves the structured document JSON to any target file path.
  Future<File> saveToPath(
    StructuredDocument document,
    String filePath,
  ) async {
    final file = File(filePath);
    final jsonString = toPrettyJson(document);

    return file.writeAsString(jsonString, flush: true);
  }

  /// Reads a structured document from a JSON file.
  Future<StructuredDocument> readFromFile(String filePath) async {
    final file = File(filePath);
    final content = await file.readAsString();
    final jsonMap = jsonDecode(content) as Map<String, dynamic>;

    return StructuredDocument.fromJson(jsonMap);
  }
}
