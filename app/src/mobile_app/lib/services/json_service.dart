import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../models/structured_document.dart';

class JsonService {
  const JsonService();

  String toPrettyJson(StructuredDocument document) {
    return const JsonEncoder.withIndent('  ').convert(document.toJson());
  }

  String toCompactJson(StructuredDocument document) {
    return jsonEncode(document.toJson());
  }

  Future<File> saveToDocumentsDirectory(
    StructuredDocument document, {
    String fileName = 'ocr_result.json',
  }) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$fileName');
    final jsonString = toPrettyJson(document);

    return file.writeAsString(jsonString, flush: true);
  }

  Future<File> saveToPath(StructuredDocument document, String filePath) async {
    final file = File(filePath);
    final jsonString = toPrettyJson(document);

    return file.writeAsString(jsonString, flush: true);
  }

  Future<StructuredDocument> readFromFile(String filePath) async {
    final file = File(filePath);
    final content = await file.readAsString();
    final jsonMap = jsonDecode(content) as Map<String, dynamic>;

    return StructuredDocument.fromJson(jsonMap);
  }

  Future<File> appendDocumentsToArrayFile(
    List<StructuredDocument> documents, {
    String fileName = 'ocr_results_final.json',
  }) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$fileName');

    List<dynamic> existing = [];

    if (await file.exists()) {
      try {
        final content = await file.readAsString();

        if (content.trim().isNotEmpty) {
          final decoded = jsonDecode(content);

          if (decoded is List) {
            existing = decoded;
          } else if (decoded is Map<String, dynamic>) {
            existing = [decoded];
          }
        }
      } catch (_) {
        existing = [];
      }
    }

    existing.addAll(documents.map((doc) => doc.toJson()));

    final jsonString = const JsonEncoder.withIndent('  ').convert(existing);
    return file.writeAsString(jsonString, flush: true);
  }
}
