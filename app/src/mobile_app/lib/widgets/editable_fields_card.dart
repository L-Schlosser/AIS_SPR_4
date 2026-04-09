import 'package:flutter/material.dart';

import '../models/editable_field_data.dart';

class EditableFieldsCard extends StatelessWidget {
  final List<EditableFieldData> fields;

  const EditableFieldsCard({super.key, required this.fields});

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
              'Felder bearbeiten',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            if (fields.isEmpty)
              const Text('Keine Felder vorhanden.')
            else
              ...List.generate(fields.length, (index) {
                final field = fields[index];

                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: TextField(
                    controller: field.valueController,
                    decoration: InputDecoration(
                      labelText: _fieldLabelGerman(field.fieldKey),
                      border: const OutlineInputBorder(),
                    ),
                    maxLines: null,
                  ),
                );
              }),
          ],
        ),
      ),
    );
  }

  String _fieldLabelGerman(String key) {
    switch (key) {
      case 'vendor':
        return 'Geschäft / Aussteller';
      case 'date':
        return 'Datum';
      case 'time':
        return 'Uhrzeit';
      case 'total_amount':
        return 'Gesamtbetrag';
      case 'currency':
        return 'Währung';
      case 'uid':
        return 'UID';
      case 'invoice_number':
        return 'Rechnungsnummer';
      case 'issue_date':
        return 'Rechnungsdatum';
      case 'due_date':
        return 'Fälligkeitsdatum';
      case 'issuer':
        return 'Aussteller';
      case 'issuer_name':
        return 'Name des Ausstellers';
      case 'patient_name':
        return 'Name des Patienten';
      case 'doctor_name':
        return 'Name des Arztes';
      case 'delivery_number':
        return 'Lieferscheinnummer';
      case 'delivery_date':
        return 'Lieferdatum';
      case 'recipient':
        return 'Empfänger';
      case 'summary':
        return 'Zusammenfassung';
      case 'name':
        return 'Name';
      case 'address':
        return 'Adresse';
      case 'diagnosis':
        return 'Diagnose';
      case 'tax_note':
        return 'Steuerhinweis';
      default:
        return _fallbackGermanLabel(key);
    }
  }

  String _fallbackGermanLabel(String key) {
    return key
        .split('_')
        .map((part) {
          if (part.isEmpty) return part;
          return part[0].toUpperCase() + part.substring(1);
        })
        .join(' ');
  }
}
