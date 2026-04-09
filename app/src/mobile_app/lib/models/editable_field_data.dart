import 'package:flutter/material.dart';

import 'structured_document.dart';

class EditableFieldData {
  final String fieldKey;
  final TextEditingController valueController;
  final ExtractedField? originalField;

  EditableFieldData({
    required this.fieldKey,
    required this.valueController,
    this.originalField,
  });

  void dispose() {
    valueController.dispose();
  }
}
