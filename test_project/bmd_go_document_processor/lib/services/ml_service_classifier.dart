import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'ml_service_ocr.dart' show OCRClassificationResult;

class MLServiceClassifier {
  late OrtSession _session;
  bool _isInitialized = false;

  Future<void> initialize() async {
    print("Initializing classifier model...");

    if (_isInitialized) return;

    final modelData =
        await rootBundle.load('assets/models/classifier_v2.onnx');

    final sessionOptions = OrtSessionOptions();

    _session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      sessionOptions,
    );

    _isInitialized = true;
    print("Classifier model initialized ✔");
  }

  Future<OCRClassificationResult> classify(String extractedText) async {
    if (!_isInitialized) {
      throw Exception("Model not initialized");
    }

    final text = extractedText;

    // --- INPUT ---
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      [text],
      [1, 1],
    );

    final inputs = <String, OrtValue>{
      "text": inputTensor,
    };

    final runOptions = OrtRunOptions();

    final outputs = await _session.runAsync(runOptions, inputs);

    if (outputs == null || outputs.isEmpty) {
      throw Exception("No outputs from ONNX model");
    }

    // --- OUTPUT 1: label ---
    String documentType = "unknown";
    print("Parsing label output...");

    final labelOutput = outputs[0]?.value;
    if (labelOutput is List && labelOutput.isNotEmpty) {
      documentType = labelOutput.first.toString();
    } else if (labelOutput is String) {
      documentType = labelOutput;
    }
    print("Predicted document type: $documentType");

    // --- OUTPUT 2: probability map ---
    double? confidence;

    if (outputs.length > 1) {
      final prob = outputs[1]?.value as List?;
      if (prob != null && prob.isNotEmpty) {
        confidence = (prob.first as List)
            .map((e) => (e as num).toDouble())
            .reduce((a, b) => a > b ? a : b);
      }
    }

    return OCRClassificationResult(
      documentType: documentType,
      confidence: confidence,
      infos: {
        'RawText': extractedText,
      },
    );
  }

  bool get isInitialized => _isInitialized;

  void dispose() {
    _session.release();
  }
}