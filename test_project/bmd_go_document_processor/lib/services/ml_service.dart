// import 'package:flutter/services.dart';
// import 'dart:io';
// import 'dart:typed_data';
// import 'package:image/image.dart' as img;
// import 'package:tflite_flutter/tflite_flutter.dart';

// class ClassificationResult {
//   final String documentType;
//   final double confidence;
//   final Map<String, double> scores;

//   ClassificationResult({
//     required this.documentType,
//     required this.confidence,
//     required this.scores,
//   });
// }

// class MLService {
//   late Interpreter _interpreter;
//   bool _isInitialized = false;
//   late List<int> _inputShape;
//   late List<int> _outputShape;

//   static const List<String> documentTypes = [
//     'recipe',
//     'bill',
//     'doctor_note',
//     'delivery_note',
//     'receipt',
//     'contract',
//     'unknown',
//   ];

//   Future<void> initialize() async {
//     try {
//       print('Initializing TensorFlow Lite model...');
//       _interpreter = await Interpreter.fromAsset('assets/models/classification_model.tflite');
//       _inputShape = _interpreter.getInputTensor(0).shape;
//       _outputShape = _interpreter.getOutputTensor(0).shape;
//       print('Model loaded successfully');
//       _isInitialized = true;
//     } catch (e) {
//       print('Error initializing model: $e');
//       rethrow;
//     }
//   }

//   bool get isInitialized => _isInitialized;

//   Future<ClassificationResult> classifyDocument(File imageFile) async {
//     if (!_isInitialized) {
//       throw Exception('Model not initialized');
//     }

//     try {
//       final imageBytes = await imageFile.readAsBytes();
//       final decodedImage = img.decodeImage(imageBytes);
      
//       if (decodedImage == null) {
//         throw Exception('Failed to decode image');
//       }

//       final inputTensor = _preprocessImage(decodedImage);
//       final output = List<double>.filled(documentTypes.length, 0.0);
//       _interpreter.run(inputTensor, output);

//       return _parseResults(output);
//     } catch (e) {
//       print('Error classifying document: $e');
//       rethrow;
//     }
//   }

//   List<List<List<List<double>>>> _preprocessImage(img.Image image) {
//     final resized = img.copyResize(image, width: 224, height: 224);

//     final imageMatrix = List.generate(
//       1,
//       (batch) => List.generate(
//         224,
//         (y) => List.generate(
//           224,
//           (x) {
//             final pixel = resized.getPixel(x, y);
//             return [
//               pixel.r / 255.0, 
//               pixel.g / 255.0, 
//               pixel.b / 255.0];
//           },
//         ),
//       ),
//     );

//     return imageMatrix;
//   }

//   ClassificationResult _parseResults(List<double> output) {
//     final scores = <String, double>{};
//     for (int i = 0; i < documentTypes.length; i++) {
//       scores[documentTypes[i]] = output[i];
//     }

//     int maxIndex = 0;
//     double maxScore = output[0];
//     for (int i = 1; i < output.length; i++) {
//       if (output[i] > maxScore) {
//         maxScore = output[i];
//         maxIndex = i;
//       }
//     }

//     return ClassificationResult(
//       documentType: documentTypes[maxIndex],
//       confidence: maxScore,
//       scores: scores,
//     );
//   }

//   void dispose() {
//     _interpreter.close();
//     _isInitialized = false;
//   }
// }



import 'dart:io';

class ClassificationResult {
  final String documentType;
  final double confidence;
  final Map<String, double> scores;

  ClassificationResult({
    required this.documentType,
    required this.confidence,
    required this.scores,
  });
}

class MLService {
  bool _isInitialized = true;

  Future<void> initialize() async {}
  
  bool get isInitialized => _isInitialized;
  
  void dispose() {}
}