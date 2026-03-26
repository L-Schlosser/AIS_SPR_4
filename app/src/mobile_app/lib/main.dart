import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'models/ocr_models.dart';
import 'models/structured_document.dart';
import 'painters/bounding_box_painter.dart';
import 'services/classification_service.dart';
import 'services/extraction_service.dart';
import 'services/json_service.dart';
import 'services/ocr_service.dart';
import 'widgets/document_summary_card.dart';
import 'widgets/extracted_fields_card.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OCR Demo',
      theme: ThemeData(useMaterial3: true),
      debugShowCheckedModeBanner: false,
      home: const OcrPage(),
    );
  }
}

class OcrPage extends StatefulWidget {
  const OcrPage({super.key});

  @override
  State<OcrPage> createState() => _OcrPageState();
}

class _OcrPageState extends State<OcrPage> {
  final OcrService _ocrService = OcrService();
  final ClassificationService _classificationService = ClassificationService();
  final ExtractionService _extractionService = ExtractionService();
  final JsonService _jsonService = const JsonService();
  final ImagePicker _imagePicker = ImagePicker();

  // Toggle this:
  // false = use bundled sample image
  // true  = take a new photo with the camera
  static const bool useCameraImage = true;

  bool _isProcessing = false;

  File? _localImageFile;
  String? _savedJsonPath;
  String _jsonOutput = '';

  OcrDocument? _ocrDocument;
  StructuredDocument? _structuredDocument;
  ClassificationResult? _classificationResult;

  Future<void> _runOcr() async {
    setState(() {
      _isProcessing = true;
      _jsonOutput = '';
      _savedJsonPath = null;
      _ocrDocument = null;
      _structuredDocument = null;
      _classificationResult = null;
    });

    try {
      final imageFile = await _getInputImageFile();

      if (imageFile == null) {
        setState(() {
          _jsonOutput = 'No image selected / captured.';
          _isProcessing = false;
        });
        return;
      }

      final ocrDocument = await _ocrService.processImageFromPath(
        imageFile.path,
      );
      final classificationResult = _classificationService.classifyDocument(
        ocrDocument,
      );

      final structuredDocument = _extractionService.extractDocument(
        ocrDocument: ocrDocument,
        documentType: classificationResult.documentType,
        classificationConfidence: classificationResult.confidence,
      );

      final savedFile = await _jsonService.saveToDocumentsDirectory(
        structuredDocument,
      );

      final jsonOutput = _jsonService.toPrettyJson(structuredDocument);

      setState(() {
        _localImageFile = imageFile;
        _ocrDocument = ocrDocument;
        _classificationResult = classificationResult;
        _structuredDocument = structuredDocument;
        _savedJsonPath = savedFile.path;
        _jsonOutput = jsonOutput;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _jsonOutput = 'Error during OCR processing: $e';
        _isProcessing = false;
      });
    }
  }

  Future<File?> _getInputImageFile() async {
    if (useCameraImage) {
      final XFile? capturedImage = await _imagePicker.pickImage(
        source: ImageSource.camera,
        imageQuality: 100,
      );

      if (capturedImage == null) {
        return null;
      }

      return File(capturedImage.path);
    }

    return _ocrService.copyAssetToFile('assets/bill.jpg', 'bill.jpg');
  }

  @override
  void initState() {
    super.initState();
    _runOcr();
  }

  @override
  void dispose() {
    _ocrService.dispose();
    super.dispose();
  }

  Widget _buildBox(String title, Widget child) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade400),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          child,
        ],
      ),
    );
  }

  List<BoundingBoxItem> _buildBoundingBoxes() {
    final items = <BoundingBoxItem>[];

    final document = _structuredDocument;
    final ocrDocument = _ocrDocument;

    if (document != null && document.fields.isNotEmpty) {
      for (final field in document.fields) {
        if (field.boundingBox != null) {
          items.add(
            BoundingBoxItem(
              rect: field.boundingBox!,
              label: field.key,
              color: Colors.red,
            ),
          );
        }
      }
      return items;
    }

    if (ocrDocument != null) {
      for (final line in ocrDocument.allLines) {
        items.add(
          BoundingBoxItem(
            rect: line.boundingBox,
            label: line.text.length > 20
                ? '${line.text.substring(0, 20)}...'
                : line.text,
            color: Colors.blue,
          ),
        );
      }
    }

    return items;
  }

  Widget _buildImageWithOverlay() {
    if (_localImageFile == null) {
      return const Text('No image loaded.');
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        return FutureBuilder<ImageInfo>(
          future: _getImageInfo(FileImage(_localImageFile!)),
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              return const Center(child: CircularProgressIndicator());
            }

            final imageInfo = snapshot.data!;
            final originalWidth = imageInfo.image.width.toDouble();
            final originalHeight = imageInfo.image.height.toDouble();

            final maxWidth = constraints.maxWidth;
            final scale = maxWidth / originalWidth;
            final displayedWidth = originalWidth * scale;
            final displayedHeight = originalHeight * scale;

            final boundingBoxes = _buildBoundingBoxes();

            return Center(
              child: SizedBox(
                width: displayedWidth,
                height: displayedHeight,
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: Image.file(_localImageFile!, fit: BoxFit.fill),
                    ),
                    Positioned.fill(
                      child: CustomPaint(
                        painter: BoundingBoxPainter(
                          items: boundingBoxes,
                          scaleX: displayedWidth / originalWidth,
                          scaleY: displayedHeight / originalHeight,
                          showLabels: true,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Future<ImageInfo> _getImageInfo(ImageProvider imageProvider) {
    final completer = Completer<ImageInfo>();
    final stream = imageProvider.resolve(const ImageConfiguration());

    late final ImageStreamListener listener;
    listener = ImageStreamListener(
      (ImageInfo info, bool synchronousCall) {
        completer.complete(info);
        stream.removeListener(listener);
      },
      onError: (dynamic error, StackTrace? stackTrace) {
        completer.completeError(error, stackTrace);
        stream.removeListener(listener);
      },
    );

    stream.addListener(listener);
    return completer.future;
  }

  Widget _buildClassificationCard() {
    final result = _classificationResult;

    if (result == null) {
      return const SizedBox.shrink();
    }

    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Classification',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Text('Detected type: ${result.documentType}'),
            const SizedBox(height: 6),
            Text(
              'Confidence: ${(result.confidence * 100).toStringAsFixed(0)}%',
            ),
            const SizedBox(height: 12),
            Text(
              'Scores',
              style: Theme.of(
                context,
              ).textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 6),
            ...result.scores.entries.map(
              (entry) => Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Text('${entry.key}: ${entry.value}'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRawTextCard() {
    final rawText = _ocrDocument?.rawText ?? '';

    return _buildBox(
      'Raw OCR Text',
      SelectableText(rawText.isEmpty ? 'No OCR text available.' : rawText),
    );
  }

  Widget _buildJsonCard() {
    return _buildBox(
      'JSON Output',
      SelectableText(_jsonOutput.isEmpty ? 'No output.' : _jsonOutput),
    );
  }

  @override
  Widget build(BuildContext context) {
    final structuredDocument = _structuredDocument;

    return Scaffold(
      appBar: AppBar(title: const Text('OCR Demo')),
      body: _isProcessing
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: ListView(
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _runOcr,
                          child: Text(
                            useCameraImage
                                ? 'Take photo and run OCR'
                                : 'Run OCR on sample image',
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildBox('Image + Overlay', _buildImageWithOverlay()),
                  if (structuredDocument != null) ...[
                    DocumentSummaryCard(document: structuredDocument),
                    const SizedBox(height: 12),
                    _buildClassificationCard(),
                    const SizedBox(height: 12),
                    ExtractedFieldsCard(document: structuredDocument),
                    const SizedBox(height: 12),
                  ],
                  if (_savedJsonPath != null)
                    _buildBox(
                      'Saved JSON File',
                      SelectableText(_savedJsonPath!),
                    ),
                  _buildRawTextCard(),
                  _buildJsonCard(),
                ],
              ),
            ),
    );
  }
}
