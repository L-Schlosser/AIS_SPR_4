import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../models/structured_document.dart';

class BackendUploadService {
  const BackendUploadService();

  // TODO: replace with your real backend endpoint
  static const String _backendUrl =
      'https://your-backend-url.example.com/upload';

  static const String _queueFileName = 'pending_upload_queue.json';

  static bool _isUploading = false;

  Future<void> enqueueAndTryUpload(List<StructuredDocument> documents) async {
    if (documents.isEmpty) return;

    final queue = await _loadQueue();

    queue.addAll(documents.map((doc) => doc.toJson()));

    await _saveQueue(queue);
    await tryUploadQueue();
  }

  Future<void> tryUploadQueue() async {
    if (_isUploading) return;

    _isUploading = true;

    try {
      final queue = await _loadQueue();
      if (queue.isEmpty) return;

      final success = await _sendToBackend(queue);

      if (success) {
        await _saveQueue([]);
      }
    } finally {
      _isUploading = false;
    }
  }

  Future<bool> _sendToBackend(List<Map<String, dynamic>> payload) async {
    final client = HttpClient();

    try {
      final request = await client.postUrl(Uri.parse(_backendUrl));
      request.headers.contentType = ContentType.json;

      // IMPORTANT:
      // backend always receives a JSON ARRAY, even for a single document
      request.write(jsonEncode(payload));

      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        return true;
      }
      return false;
    } catch (e) {
      return false;
    } finally {
      client.close(force: true);
    }
  }

  Future<List<Map<String, dynamic>>> _loadQueue() async {
    final file = await _getQueueFile();

    if (!await file.exists()) {
      return [];
    }

    try {
      final content = await file.readAsString();
      if (content.trim().isEmpty) return [];

      final decoded = jsonDecode(content);

      if (decoded is List) {
        return decoded
            .whereType<Map>()
            .map((e) => Map<String, dynamic>.from(e))
            .toList();
      }

      return [];
    } catch (e) {
      return [];
    }
  }

  Future<void> _saveQueue(List<Map<String, dynamic>> queue) async {
    final file = await _getQueueFile();
    final jsonString = const JsonEncoder.withIndent('  ').convert(queue);
    await file.writeAsString(jsonString, flush: true);
  }

  Future<File> _getQueueFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_queueFileName');
  }
}
