import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;

class MLServiceTest {
  static const List<String> documentTypes = [
    'recipe',
    'bill',
    'doctor_note',
    'delivery_note',
    'receipt',
    'contract',
  ];

  static Future<Map<String, dynamic>> classifyDocument(File imageFile) async {
    await Future.delayed(const Duration(milliseconds: 800));

    final imageBytes = await imageFile.readAsBytes();
    final decodedImage = img.decodeImage(imageBytes);

    final scores = <String, double>{};
    final random = Random(decodedImage?.width ?? 0);

    var sum = 0.0;
    final tempScores = <double>[];
    for (int i = 0; i < documentTypes.length; i++) {
      final score = random.nextDouble();
      tempScores.add(score);
      sum += score;
    }

    for (int i = 0; i < documentTypes.length; i++) {
      scores[documentTypes[i]] = tempScores[i] / sum;
    }

    int maxIndex = 0;
    double maxScore = scores.values.first;
    for (int i = 1; i < documentTypes.length; i++) {
      if (scores.values.elementAt(i) > maxScore) {
        maxScore = scores.values.elementAt(i);
        maxIndex = i;
      }
    }

    return {
      'documentType': documentTypes[maxIndex],
      'confidence': maxScore,
      'scores': scores,
    };
  }
}