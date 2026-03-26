import 'package:flutter/material.dart';

import '../models/structured_document.dart';

/// Simple card widget showing the most important document-level information.
class DocumentSummaryCard extends StatelessWidget {
  final StructuredDocument document;

  const DocumentSummaryCard({
    super.key,
    required this.document,
  });

  @override
  Widget build(BuildContext context) {
    final confidenceText = document.classificationConfidence != null
        ? '${(document.classificationConfidence! * 100).toStringAsFixed(0)}%'
        : 'Unknown';

    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Document Summary',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            _InfoRow(
              label: 'Document type',
              value: _formatDocumentType(document.documentType),
            ),
            const SizedBox(height: 8),
            _InfoRow(
              label: 'Classification confidence',
              value: confidenceText,
            ),
            const SizedBox(height: 8),
            _InfoRow(
              label: 'Extracted fields',
              value: document.fields.length.toString(),
            ),
            const SizedBox(height: 8),
            _InfoRow(
              label: 'Detected entities',
              value: document.entities.length.toString(),
            ),
            if (document.rawText.trim().isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(
                'Raw text preview',
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const SizedBox(height: 6),
              Text(
                _buildPreview(document.rawText),
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ],
        ),
      ),
    );
  }

  String _formatDocumentType(String input) {
    if (input.trim().isEmpty) return 'Unknown';

    return input
        .split('_')
        .map((part) {
          if (part.isEmpty) return part;
          return part[0].toUpperCase() + part.substring(1);
        })
        .join(' ');
  }

  String _buildPreview(String text, {int maxLength = 180}) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.length <= maxLength) return normalized;
    return '${normalized.substring(0, maxLength)}...';
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _InfoRow({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    final labelStyle = Theme.of(context).textTheme.bodyMedium?.copyWith(
          fontWeight: FontWeight.w600,
        );

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 170,
          child: Text(label, style: labelStyle),
        ),
        Expanded(
          child: Text(value),
        ),
      ],
    );
  }
}
