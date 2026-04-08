import 'dart:io';

class ClassificationResult {
  final String documentType;
  final double confidence;
  final Map<String, String> infos;

  ClassificationResult({
    required this.documentType,
    required this.confidence,
    required this.infos,
  });
}

class MLService {
  bool _isInitialized = true;

  Future<void> initialize() async {}
  
  bool get isInitialized => _isInitialized;
  
  void dispose() {}
}