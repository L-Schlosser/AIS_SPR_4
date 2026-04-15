import 'package:flutter/material.dart';

class ClassificationCard extends StatelessWidget {
  final String? selectedDocumentType;
  final List<String> documentTypes;
  final ValueChanged<String?> onDocumentTypeChanged;
  final double confidence;

  const ClassificationCard({
    super.key,
    required this.selectedDocumentType,
    required this.documentTypes,
    required this.onDocumentTypeChanged,
    required this.confidence,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Dokumenttyp',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: selectedDocumentType,
              decoration: const InputDecoration(
                labelText: 'Dokumenttyp auswählen',
                border: OutlineInputBorder(),
              ),
              items: documentTypes
                  .map(
                    (type) => DropdownMenuItem<String>(
                      value: type,
                      child: Text(_documentTypeLabel(type)),
                    ),
                  )
                  .toList(),
              onChanged: onDocumentTypeChanged,
            ),
            const SizedBox(height: 12),
            Text('Konfidenz: ${(confidence * 100).toStringAsFixed(0)}%'),
          ],
        ),
      ),
    );
  }

  String _documentTypeLabel(String type) {
    switch (type) {
      case 'receipt':
        return 'Kassenbeleg';
      case 'invoice':
        return 'Rechnung';
      case 'delivery_note':
        return 'Lieferschein';
      case 'doctor_note':
        return 'Arztbestätigung';
      case 'caregiving_leave_confirmation':
        return 'Bestätigung Pflegefreistellung';
      case 'master_data_change':
        return 'Stammdatenänderung';
      case 'unknown':
        return 'Unbekannt';
      default:
        return type;
    }
  }
}
