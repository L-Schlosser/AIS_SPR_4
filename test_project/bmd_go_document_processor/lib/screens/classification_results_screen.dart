import 'package:flutter/material.dart';
import 'package:percent_indicator/percent_indicator.dart';
import '../services/ml_service_ocr.dart';
import '../utils/doc_type_translation.dart';

class ClassificationResultsScreen extends StatefulWidget {
  final OCRClassificationResult result;

  const ClassificationResultsScreen({
    Key? key,
    required this.result,
  }) : super(key: key);

  @override
  State<ClassificationResultsScreen> createState() =>
      _ClassificationResultsScreenState();
}

class _ClassificationResultsScreenState
    extends State<ClassificationResultsScreen> {
  @override
  Widget build(BuildContext context) {
    final result = widget.result;
    final confidence = result.confidence ?? 0.0;
    final display = docTypeDE[result.documentType.toLowerCase()] ?? result.documentType;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Klassifizierungsergebnisse'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: _getColorForDocType(result.documentType),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Document Type',
                      style: TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      display.toUpperCase(),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 30),
              if(confidence != null && confidence != 0.0) ...[
                Text(
                  'Confidence Score:',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 15),
              
                LinearPercentIndicator(
                  lineHeight: 20.0,
                  percent: confidence,
                  center: Text(
                    '${(confidence * 100).toStringAsFixed(1)}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  progressColor: Colors.green,
                  backgroundColor: Colors.grey.shade300,
                ),
                const SizedBox(height: 30),
              ] else ...[
                const Text(
                  'Confidence Score: Not available',
                  style: TextStyle(fontSize: 16, fontStyle: FontStyle.italic),
                ),
                const SizedBox(height: 30),
              ],
              Text(
                'Alle Informationen',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 20),
              ..._buildInfosList(result),
              const SizedBox(height: 30),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () => Navigator.pop(context),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.grey,
                      ),
                      child: const Text('Zurück'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Speichern-Funktion ist noch nicht implementiert.'),
                          ),
                        );
                      },
                      child: const Text('Speichern'),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  List<Widget> _buildInfosList(OCRClassificationResult result) {
    return result.infos.entries
        .map(
          (entry) => Padding(
            padding: const EdgeInsets.only(bottom: 12.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(entry.key.replaceAll('_', ' ').toUpperCase()),
                Text(entry.value.toString()), // '${(entry.value * 100).toStringAsFixed(1)}%'),
              ],
            ),
          ),
        )
        .toList();
  }

  Color _getColorForDocType(String docType) {
    switch (docType.toLowerCase()) {
      case 'receipt':
        return Colors.orange;
      case 'invoice':
        return Colors.blue;
      case 'doctor_note':
        return Colors.red;
      case 'care_leave':
        return Colors.purple;
      case 'delivery_note':
        return Colors.green;
      case 'master_data':
        return Colors.indigo;
      default:
        return Colors.grey;
    }
  }
}