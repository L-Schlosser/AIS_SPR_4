import 'dart:ffi';
import 'dart:io';

import 'package:flutter/material.dart';

import '../models/editable_field_data.dart';
import '../models/review_image_item.dart';
import '../models/structured_document.dart';
import '../services/classification_service.dart';
import '../services/extraction_service.dart';
import '../services/json_service.dart';
import '../services/ocr_service.dart';
import '../widgets/classification_card.dart';
import '../widgets/editable_fields_card.dart';
import '../services/backend_upload_service.dart';
import 'dart:async';

const String screen2ResultNeedNewImages = 'need_new_images';
const String screen2ResultSavedAll = 'saved_all';
const int snackbarTimer = 6;

class Screen2ReviewPage extends StatefulWidget {
  final List<File> imageFiles;

  const Screen2ReviewPage({super.key, required this.imageFiles});

  @override
  State<Screen2ReviewPage> createState() => _Screen2ReviewPageState();
}

class _Screen2ReviewPageState extends State<Screen2ReviewPage> {
  final OcrService _ocrService = OcrService();
  final ClassificationService _classificationService = ClassificationService();
  final ExtractionService _extractionService = ExtractionService();
  final JsonService _jsonService = const JsonService();
  final BackendUploadService _backendUploadService =
      const BackendUploadService();

  final List<String> _documentTypes = const [
    'receipt',
    'invoice',
    'delivery_note',
    'doctor_note',
    'caregiving_leave_confirmation',
    'master_data_change',
    'unknown',
  ];

  late final List<ReviewImageItem> _items;
  int _currentIndex = 0;
  bool _isBusy = false;

  Timer? _pendingSaveTimer;

  ReviewImageItem? get _currentItem {
    if (_items.isEmpty) return null;
    return _items[_currentIndex];
  }

  @override
  void initState() {
    super.initState();
    _items = widget.imageFiles
        .map((file) => ReviewImageItem(imageFile: file))
        .toList();

    _ensureCurrentProcessed();
  }

  @override
  void dispose() {
    _pendingSaveTimer?.cancel();

    for (final item in _items) {
      item.dispose();
    }
    _ocrService.dispose();
    super.dispose();
  }

  Future<void> _ensureCurrentProcessed() async {
    final item = _currentItem;
    if (item == null) return;

    if (item.isProcessing || item.structuredDocument != null) return;

    await _processItem(item);
  }

  Future<void> _processItem(ReviewImageItem item) async {
    item.isProcessing = true;
    setState(() {});

    try {
      final ocrDocument = await _ocrService.processImageFromPath(
        item.imageFile.path,
      );

      final classificationResult = _classificationService.classifyDocument(
        ocrDocument,
      );

      final structuredDocument = _extractionService.extractDocument(
        ocrDocument: ocrDocument,
        documentType: classificationResult.documentType,
        classificationConfidence: classificationResult.confidence,
      );

      item.ocrDocument = ocrDocument;
      item.classificationResult = classificationResult;
      item.structuredDocument = structuredDocument;
      item.selectedDocumentType = structuredDocument.documentType;

      _loadEditableFields(item, structuredDocument);
    } finally {
      item.isProcessing = false;
      if (mounted) {
        setState(() {});
      }
    }
  }

  void _loadEditableFields(ReviewImageItem item, StructuredDocument document) {
    for (final field in item.editableFields) {
      field.dispose();
    }
    item.editableFields.clear();

    for (final field in document.fields) {
      item.editableFields.add(
        EditableFieldData(
          fieldKey: field.key,
          valueController: TextEditingController(text: field.value),
          originalField: field,
        ),
      );
    }
  }

  StructuredDocument? _buildEditedStructuredDocument(ReviewImageItem item) {
    final base = item.structuredDocument;
    if (base == null) return null;

    final editedFields = <ExtractedField>[];

    for (final editable in item.editableFields) {
      final key = editable.fieldKey.trim();
      final value = editable.valueController.text.trim();

      if (key.isEmpty) continue;

      editedFields.add(
        ExtractedField(
          key: key,
          value: value,
          confidence: editable.originalField?.confidence,
          boundingBox: editable.originalField?.boundingBox,
          sourceText: editable.originalField?.sourceText,
        ),
      );
    }

    return StructuredDocument(
      documentType: item.selectedDocumentType ?? base.documentType,
      classificationConfidence: base.classificationConfidence,
      rawText: base.rawText,
      fields: editedFields,
      entities: base.entities,
      ocrDocument: base.ocrDocument,
    );
  }

  Future<void> _onDocumentTypeChanged(String? newType) async {
    final item = _currentItem;
    if (item == null || newType == null || item.ocrDocument == null) return;

    setState(() {
      item.selectedDocumentType = newType;
      item.isProcessing = true;
    });

    try {
      final reExtracted = _extractionService.extractDocument(
        ocrDocument: item.ocrDocument!,
        documentType: newType,
        classificationConfidence: item.classificationResult?.confidence,
      );

      item.structuredDocument = reExtracted;
      _loadEditableFields(item, reExtracted);
    } finally {
      item.isProcessing = false;
      if (mounted) {
        setState(() {});
      }
    }
  }

  void _showSaveDialog() async {
    final item = _currentItem;
    if (item == null) return;

    final choice = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Speichern'),
        content: const Text('Was möchten Sie tun?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop('cancel'),
            child: const Text('Abbrechen'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop('save_all'),
            child: const Text('Alle speichern'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(context).pop('save_one'),
            child: const Text('Speichern'),
          ),
        ],
      ),
    );

    if (choice == 'save_one') {
      await _saveCurrentWithUndo();
    } else if (choice == 'save_all') {
      await _saveAll();
    }
  }

  Future<void> _saveCurrentWithUndo() async {
    final item = _currentItem;
    if (item == null) return;

    if (item.structuredDocument == null) {
      await _ensureCurrentProcessed();
    }

    final current = _currentItem;
    if (current == null) return;

    final editedDocument = _buildEditedStructuredDocument(current);
    if (editedDocument == null) return;

    final removedIndex = _currentIndex;
    final removedItem = current;

    bool undone = false;

    _pendingSaveTimer?.cancel();

    setState(() {
      _items.removeAt(removedIndex);

      if (_items.isEmpty) {
        _currentIndex = 0;
      } else if (_currentIndex >= _items.length) {
        _currentIndex = _items.length - 1;
      }
    });

    if (_items.isNotEmpty) {
      _ensureCurrentProcessed();
    }

    if (!mounted) return;

    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();

    _pendingSaveTimer = Timer(const Duration(seconds: snackbarTimer), () async {
      if (undone) return;

      await _jsonService.appendDocumentsToArrayFile([editedDocument]);

      // send asynchronously, do not block UI
      // unawaited(_backendUploadService.enqueueAndTryUpload([editedDocument]));

      removedItem.dispose();

      if (!mounted) return;

      final messenger = ScaffoldMessenger.of(context);
      messenger.hideCurrentSnackBar();
      messenger.clearSnackBars();

      if (_items.isEmpty) {
        Navigator.of(context).pop(screen2ResultNeedNewImages);
      }
    });

    messenger.showSnackBar(
      SnackBar(
        duration: const Duration(seconds: snackbarTimer),
        content: TweenAnimationBuilder<double>(
          tween: Tween(begin: 1.0, end: 0.0),
          duration: const Duration(seconds: snackbarTimer),
          builder: (context, value, child) {
            return Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Dokument gespeichert'),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(999),
                  child: LinearProgressIndicator(value: value, minHeight: 4),
                ),
              ],
            );
          },
        ),
        action: SnackBarAction(
          label: 'Rückgängig',
          onPressed: () {
            if (undone) return;

            undone = true;
            _pendingSaveTimer?.cancel();
            _pendingSaveTimer = null;

            if (!mounted) return;

            setState(() {
              final insertIndex = removedIndex.clamp(0, _items.length);
              _items.insert(insertIndex, removedItem);
              _currentIndex = insertIndex;
            });

            _ensureCurrentProcessed();
          },
        ),
      ),
    );
  }

  Future<void> _saveAll() async {
    if (_items.isEmpty) return;

    _pendingSaveTimer?.cancel();
    _pendingSaveTimer = null;

    setState(() {
      _isBusy = true;
    });

    try {
      final documentsToSave = <StructuredDocument>[];

      for (final item in _items) {
        if (item.structuredDocument == null) {
          await _processItem(item);
        }

        final edited = _buildEditedStructuredDocument(item);
        if (edited != null) {
          documentsToSave.add(edited);
        }
      }

      if (!mounted) return;

      final removedItems = List<ReviewImageItem>.from(_items);
      bool undone = false;

      setState(() {
        _items.clear();
        _currentIndex = 0;
        _isBusy = false;
      });

      final messenger = ScaffoldMessenger.of(context);
      messenger.hideCurrentSnackBar();

      _pendingSaveTimer = Timer(
        const Duration(seconds: snackbarTimer),
        () async {
          if (undone) return;

          if (documentsToSave.isNotEmpty) {
            await _jsonService.appendDocumentsToArrayFile(documentsToSave);

            // send asynchronously, do not block UI
            // unawaited(
            //   _backendUploadService.enqueueAndTryUpload(documentsToSave),
            // );
          }

          for (final item in removedItems) {
            item.dispose();
          }

          if (!mounted) return;

          final messenger = ScaffoldMessenger.of(context);
          messenger.hideCurrentSnackBar();
          messenger.clearSnackBars();

          Navigator.of(context).pop(screen2ResultSavedAll);
        },
      );

      messenger.showSnackBar(
        SnackBar(
          duration: const Duration(seconds: snackbarTimer),
          content: TweenAnimationBuilder<double>(
            tween: Tween(begin: 1.0, end: 0.0),
            duration: const Duration(seconds: snackbarTimer),
            builder: (context, value, child) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Alle Dokumente gespeichert'),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(999),
                    child: LinearProgressIndicator(value: value, minHeight: 4),
                  ),
                ],
              );
            },
          ),
          action: SnackBarAction(
            label: 'Rückgängig',
            onPressed: () {
              if (undone) return;

              undone = true;
              _pendingSaveTimer?.cancel();
              _pendingSaveTimer = null;

              if (!mounted) return;

              setState(() {
                _items.addAll(removedItems);
                _currentIndex = 0;
              });

              _ensureCurrentProcessed();
            },
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _isBusy = false;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Fehler beim Speichern: $e')));
    }
  }

  void _goToPreviousImage() {
    if (_items.isEmpty) return;

    setState(() {
      _currentIndex = (_currentIndex - 1 + _items.length) % _items.length;
    });

    _ensureCurrentProcessed();
  }

  void _goToNextImage() {
    if (_items.isEmpty) return;

    setState(() {
      _currentIndex = (_currentIndex + 1) % _items.length;
    });

    _ensureCurrentProcessed();
  }

  Future<void> _deleteCurrentImage() async {
    final item = _currentItem;
    if (item == null) return;

    final shouldDelete = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Bild löschen'),
        content: const Text('Möchten Sie dieses Bild wirklich löschen?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Abbrechen'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Löschen'),
          ),
        ],
      ),
    );

    if (shouldDelete != true) return;

    final removedItem = item;
    final removedIndex = _currentIndex;

    setState(() {
      _items.removeAt(removedIndex);
      removedItem.dispose();

      if (_items.isEmpty) {
        _currentIndex = 0;
      } else if (_currentIndex >= _items.length) {
        _currentIndex = _items.length - 1;
      }
    });

    if (_items.isEmpty) {
      if (!mounted) return;
      Navigator.of(context).pop(screen2ResultNeedNewImages);
      return;
    }

    _ensureCurrentProcessed();
  }

  Widget _buildImageSlideshow() {
    final item = _currentItem;
    if (item == null) {
      return const SizedBox.shrink();
    }

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade400),
        borderRadius: BorderRadius.circular(12),
        color: Colors.white,
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          Stack(
            alignment: Alignment.center,
            children: [
              SizedBox(
                height: 260,
                width: double.infinity,
                child: Image.file(
                  item.imageFile,
                  fit: BoxFit.contain,
                  gaplessPlayback: true,
                ),
              ),
              Positioned(
                left: 0,
                child: CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    onPressed: _goToPreviousImage,
                    icon: const Icon(Icons.chevron_left, color: Colors.white),
                  ),
                ),
              ),
              Positioned(
                right: 0,
                child: CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    onPressed: _goToNextImage,
                    icon: const Icon(Icons.chevron_right, color: Colors.white),
                  ),
                ),
              ),
              Positioned(
                top: 0,
                right: 0,
                child: CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    onPressed: _deleteCurrentImage,
                    icon: const Icon(Icons.close, color: Colors.white),
                    tooltip: 'Bild löschen',
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text('${_currentIndex + 1} / ${_items.length}'),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final item = _currentItem;

    return Scaffold(
      appBar: AppBar(title: const Text('Klassifikation')),
      bottomNavigationBar: SafeArea(
        top: false,
        child: Container(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
          decoration: BoxDecoration(
            color: Theme.of(context).scaffoldBackgroundColor,
            border: Border(top: BorderSide(color: Colors.grey.shade300)),
          ),
          child: Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => Navigator.of(context).pop(),
                  icon: const Icon(Icons.arrow_back),
                  label: const Text('Zurück'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: (_isBusy || item == null) ? null : _showSaveDialog,
                  icon: const Icon(Icons.save),
                  label: const Text('Speichern'),
                ),
              ),
            ],
          ),
        ),
      ),
      body: item == null
          ? const Center(child: CircularProgressIndicator())
          : Stack(
              children: [
                ListView(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 110),
                  children: [
                    _buildImageSlideshow(),
                    const SizedBox(height: 12),
                    if (item.classificationResult != null ||
                        item.structuredDocument != null)
                      ClassificationCard(
                        selectedDocumentType: item.selectedDocumentType,
                        documentTypes: _documentTypes,
                        onDocumentTypeChanged: _onDocumentTypeChanged,
                        confidence:
                            item.structuredDocument?.classificationConfidence ??
                            item.classificationResult?.confidence ??
                            0.0,
                      ),
                    const SizedBox(height: 12),
                    EditableFieldsCard(fields: item.editableFields),
                  ],
                ),
                if (_isBusy || item.isProcessing)
                  Positioned.fill(
                    child: Container(
                      color: Colors.black.withOpacity(0.08),
                      child: const Center(child: CircularProgressIndicator()),
                    ),
                  ),
              ],
            ),
    );
  }
}
