import 'package:flutter/material.dart';

import '../models/structured_document.dart';

/// Card widget that displays all extracted fields from a structured document.
class ExtractedFieldsCard extends StatelessWidget {
  final StructuredDocument document;

  const ExtractedFieldsCard({
    super.key,
    required this.document,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: document.fields.isEmpty
            ? Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Extracted Fields',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'No fields extracted yet.',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              )
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Extracted Fields',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  const SizedBox(height: 12),
                  ...document.fields.map(
                    (field) => Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: _FieldTile(field: field),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

class _FieldTile extends StatelessWidget {
  final ExtractedField field;

  const _FieldTile({
    required this.field,
  });

  @override
  Widget build(BuildContext context) {
    final labelStyle = Theme.of(context).textTheme.bodyMedium?.copyWith(
          fontWeight: FontWeight.w700,
        );

    final secondaryStyle = Theme.of(context).textTheme.bodySmall?.copyWith(
          color: Colors.grey.shade700,
        );

    final confidenceText = field.confidence != null
        ? '${(field.confidence! * 100).toStringAsFixed(0)}%'
        : null;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_formatKey(field.key), style: labelStyle),
          const SizedBox(height: 4),
          Text(
            field.value.isEmpty ? '—' : field.value,
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          if (confidenceText != null) ...[
            const SizedBox(height: 6),
            Text('Confidence: $confidenceText', style: secondaryStyle),
          ],
          if (field.sourceText != null && field.sourceText!.trim().isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              'Source: ${field.sourceText}',
              style: secondaryStyle,
            ),
          ],
          if (field.boundingBox != null) ...[
            const SizedBox(height: 4),
            Text(
              'Box: '
              '(${field.boundingBox!.left.toStringAsFixed(1)}, '
              '${field.boundingBox!.top.toStringAsFixed(1)}) - '
              '(${field.boundingBox!.right.toStringAsFixed(1)}, '
              '${field.boundingBox!.bottom.toStringAsFixed(1)})',
              style: secondaryStyle,
            ),
          ],
        ],
      ),
    );
  }

  String _formatKey(String key) {
    if (key.trim().isEmpty) return 'Unknown Field';

    return key
        .split('_')
        .map((part) {
          if (part.isEmpty) return part;
          return part[0].toUpperCase() + part.substring(1);
        })
        .join(' ');
  }
}
