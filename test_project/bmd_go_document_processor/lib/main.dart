import 'package:flutter/material.dart';
import 'dart:io';
import 'services/image_picker_service.dart';
import 'package:image_picker/image_picker.dart';
import 'services/ml_service_ocr.dart';
import 'services/ml_service_classifier.dart';
import 'screens/classification_results_screen.dart';
import 'package:pdfx/pdfx.dart';
import 'services/gliner_service.dart';
import 'services/ml_service_extractor.dart';
import 'services/ml_service_qwen.dart';
import 'services/ml_service_ner.dart';
import 'services/ml_service_newNer.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BMD Go Document Processor',
      theme: ThemeData(
        primarySwatch: Colors.orange,
        useMaterial3: true,
      ),
      home: const MainNavigationScreen(),
    );
  }
}

bool _isPdf(File file) {
  return file.path.toLowerCase().endsWith('.pdf');
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({Key? key}) : super(key: key);

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  // int _selectedIndex = 0;

  // static const List<Widget> _screens = [
  //   UploadScreen(),
  //   //HistoryScreen(),
  // ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('BMD Dokumenterfassung'),
        centerTitle: true,
        elevation: 2,
      ),
      body: const UploadScreen(), // _screens[_selectedIndex],
      // bottomNavigationBar: BottomNavigationBar(
      //   currentIndex: _selectedIndex,
      //   onTap: (index) {
      //     setState(() {
      //       _selectedIndex = index;
      //     });
      //   },
      //   items: const [
      //     BottomNavigationBarItem(
      //       icon: Icon(Icons.upload_file),
      //       label: 'Upload',
      //     ),
          // BottomNavigationBarItem(
          //   icon: Icon(Icons.history),
          //   label: 'History',
          // ),
      //],
      //),
    );
  }
}

class UploadScreen extends StatefulWidget {
  const UploadScreen({Key? key}) : super(key: key);

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final ImagePickerService _imagePickerService = ImagePickerService();
  List<File> _selectedFiles = [];  // Changed from File? to List<File>
  bool _isLoading = false;
  final Map<String, Future<int>> _pdfPageCountCache = {};

  /// Open camera with gallery option in bottom sheet
  Future<void> _openCameraWithGallery() async {
    showModalBottomSheet(
      context: context,
      builder: (context) => Container(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt, color: Colors.orange, size: 30),
              title: const Text('Kamera'),
              subtitle: const Text('Scan mit Kamera (mehrere Seiten)'),
              onTap: () {
                Navigator.pop(context);
                _takeMultiplePhotosWithCamera();
              },
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.image, color: Colors.blue, size: 30),
              title: const Text('Galerie'),
              subtitle: const Text('Mehrere Bilder auswählen'),
              onTap: () {
                Navigator.pop(context);
                _pickMultipleImagesFromGallery();
              },
            ),
          ],
        ),
      ),
    );
  }

  /// Take multiple photos with camera (one page at a time)
  Future<void> _takeMultiplePhotosWithCamera() async {
    if (!mounted) return;

    showDialog(
      context: context,
      builder: (context) => _CameraMultiPageDialog(
        imagePickerService: _imagePickerService,
        onImagesSelected: (images) {
          setState(() {
            _selectedFiles = images;
          });
          Navigator.pop(context);
        },
      ),
    );
  }

  /// Pick multiple images from gallery
  Future<void> _pickMultipleImagesFromGallery() async {
    setState(() => _isLoading = true);
    try {
      final files = await _imagePickerService.pickMultipleImagesForDocument(
        source: ImageSource.gallery,
      );
      if (files.isNotEmpty) {
        setState(() => _selectedFiles = files);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('${files.length} Bilder ausgewählt'), duration: const Duration(milliseconds: 800)),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  /// Pick document file (PDF, etc.)
  Future<void> _pickDocument() async {
    setState(() => _isLoading = true);
    try {
      final file = await _imagePickerService.pickDocumentFile();
      if (file != null) {
        setState(() => _selectedFiles = [file]);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Document selected successfully'), duration: Duration(milliseconds: 800)),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }
  /// Remove specific page
  void _removePage(int index) {
    setState(() {
      _selectedFiles.removeAt(index);
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Seite entfernt'), duration: Duration(milliseconds: 800),),
    );
  }

  /// Clear all images
  void _clearAllImages() {
    setState(() => _selectedFiles = []);
  }

  Future<int> _getPdfPageCount(File file) {
    return _pdfPageCountCache.putIfAbsent(file.path, () async {
      final document = await PdfDocument.openFile(file.path);
      try {
        return document.pagesCount;
      } finally {
        await document.close();
      }
    });
  }

  Future<int> _getTotalSelectedPageCount() async {
    var total = 0;

    for (final file in _selectedFiles) {
      if (_isPdf(file)) {
        total += await _getPdfPageCount(file);
      } else {
        total += 1;
      }
    }

    return total;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Scrollable content
        Expanded(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                children: [
                  const SizedBox(height: 20),
                  const Text(
                    'Belegupload',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 30),

                  if (_selectedFiles.isEmpty) ...[
                    const SizedBox(height: 10),
                    const Divider(),
                    const SizedBox(height: 40),

                    Center(
                      child: Column(
                        children: [
                          const SizedBox(height: 60),
                          Icon(
                            Icons.inbox_outlined,
                            size: 120,
                            color: Colors.grey.shade300,
                          ),
                          const SizedBox(height: 30),
                          const Text(
                            'Keine Einträge vorhanden.',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 10),
                          const Text(
                            'Bitte laden Sie eine Datei hoch oder starten Sie einen Scan.',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey,
                            ),
                          ),
                          const SizedBox(height: 80),
                        ],
                      ),
                    ),
                  ] else ...[
                    // Show selected images
                    Container(
                      padding: const EdgeInsets.all(15),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.orange),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          FutureBuilder<int>(
                            future: _getTotalSelectedPageCount(),
                            builder: (context, snapshot) {
                              final selectedPages = snapshot.data ?? _selectedFiles.length;

                              return Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    '$selectedPages Seite(n) ausgewählt',
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 10,
                                      vertical: 5,
                                    ),
                                    decoration: BoxDecoration(
                                      color: Colors.orange,
                                      borderRadius: BorderRadius.circular(20),
                                    ),
                                    child: Text(
                                      '$selectedPages',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                ],
                              );
                            },
                          ),
                          const SizedBox(height: 15),
                          // Thumbnail grid
                          GridView.builder(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            gridDelegate:
                                const SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 2,
                              crossAxisSpacing: 10,
                              mainAxisSpacing: 10,
                              childAspectRatio: 0.8,
                            ),
                            itemCount: _selectedFiles.length,
                            itemBuilder: (context, index) {
                              final doc = _selectedFiles[index];
                              return Stack(
                                children: [
                                  Container(
                                    decoration: BoxDecoration(
                                      border: Border.all(color: Colors.orange),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: ClipRRect(
                                      borderRadius: BorderRadius.circular(8),
                                       child: _isPdf(doc)
                                        ? PdfViewPinch(
                                            controller: PdfControllerPinch(
                                              document: PdfDocument.openFile(doc.path),
                                            ),
                                          )
                                        : Image.file(
                                            doc,
                                            fit: BoxFit.cover,
                                          ),
                                    ),
                                  ),
                                  if (_isPdf(doc))
                                    Positioned(
                                      top: 5,
                                      right: 5,
                                      child: FutureBuilder<int>(
                                        future: _getPdfPageCount(doc),
                                        builder: (context, snapshot) {
                                          final pageCountText = snapshot.hasData
                                              ? '${snapshot.data} Seiten'
                                              : 'PDF';

                                          return Container(
                                            padding: const EdgeInsets.symmetric(
                                              horizontal: 8,
                                              vertical: 4,
                                            ),
                                            decoration: BoxDecoration(
                                              color: Colors.blueGrey,
                                              borderRadius: BorderRadius.circular(12),
                                            ),
                                            child: Text(
                                              pageCountText,
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontSize: 12,
                                                fontWeight: FontWeight.bold,
                                              ),
                                            ),
                                          );
                                        },
                                      ),
                                    ),
                                  // Page number
                                  Positioned(
                                    top: 5,
                                    left: 5,
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8,
                                        vertical: 4,
                                      ),
                                      decoration: BoxDecoration(
                                        color: Colors.orange,
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Text(
                                        'S. ${index + 1}',
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 12,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                  ),
                                  // Delete button
                                  Positioned(
                                    bottom: 5,
                                    right: 5,
                                    child: GestureDetector(
                                      onTap: () => _removePage(index),
                                      child: Container(
                                        padding: const EdgeInsets.all(4),
                                        decoration: BoxDecoration(
                                          color: Colors.red,
                                          borderRadius:
                                              BorderRadius.circular(20),
                                        ),
                                        child: const Icon(
                                          Icons.close,
                                          color: Colors.white,
                                          size: 16,
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 30),
                  ]
                ],
              ),
            ),
          ),
        ),

        // Buttons at bottom
        Container(
          padding: const EdgeInsets.all(20.0),
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border(
              top: BorderSide(color: Colors.grey.shade300),
              bottom: BorderSide(color: Colors.grey.shade300),
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (_selectedFiles.isEmpty) ...[
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isLoading ? null : _pickDocument,
                        icon: const Icon(Icons.file_download_outlined),
                        label: const Text('Hochladen'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 15),
                          backgroundColor: Colors.orange,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isLoading ? null : _openCameraWithGallery,
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Scannen'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 15),
                          backgroundColor: Colors.orange,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ] else ...[
                // Add more pages button
                // SizedBox(
                //   width: double.infinity,
                //   child: ElevatedButton.icon(
                //     onPressed: _isLoading ? null : _addMorePages,
                //     icon: const Icon(Icons.add),
                //     label: const Text('Weitere Seite hinzufügen'),
                //     style: ElevatedButton.styleFrom(
                //       padding: const EdgeInsets.symmetric(vertical: 15),
                //       backgroundColor: Colors.blue,
                //       foregroundColor: Colors.white,
                //     ),
                //   ),
                // ),
                // const SizedBox(height: 12),

                // Process OCR button
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _isLoading
                        ? null
                        : () async {
                            setState(() => _isLoading = true);
                            final orcMlService = OCRMLService();
                            String extractedText;
                            if(_isPdf(_selectedFiles.first)){
                              extractedText = await orcMlService.processPdf(_selectedFiles.first);
                            } else {
                              extractedText = await orcMlService.processImages(_selectedFiles);
                            }

                            //CLASSIFICATION
                            print('Classifying OCR result');
                            final classifierService = MLServiceClassifier();
                            await classifierService.initialize();
                            final classificationResult = await classifierService.classify(extractedText);


// ___ NER - works the best
                            // // NER: funtkionerit!! 
                            // print('Extract features');
                            // final nerExtractor = MLServiceNER();
                            // await nerExtractor.initialize();
                            // final extractedInfos = await nerExtractor.extract(extractedText);
                            // classificationResult.infos.addAll(extractedInfos.felder);

//____ NER - my finetune
                            print('Extract features');
                            final nerExtractor = MLServiceNewNER();
                            await nerExtractor.initialize();

                            // documentType from your classifier: invoice, receipt, doctor_note, ...
                            final extractedInfos = await nerExtractor.extract(
                              classificationResult.documentType,
                              extractedText,
                            );

                            classificationResult.infos.addAll(extractedInfos.felder);

                            nerExtractor.dispose();
//___GLINER

                            // print('Extract features');
                            // final glinerExtractor = GLiNERService(debugModelOutput: true);
                            // await glinerExtractor.initialize();
                            // final extractedInfos = await glinerExtractor.extract(
                            //   extractedText,
                            //   documentType: classificationResult.documentType,
                            // );
                            // print('Extracted infos:' + extractedInfos.toString());
                            // final glinerInfos = <String, dynamic>{};
                            // for (final entity in extractedInfos) {
                            //   final existing = glinerInfos[entity.label];
                            //   if (existing == null) {
                            //     print("existing is null");
                            //     glinerInfos[entity.label] = entity.text;
                            //   } else if (existing is String) {
                            //     glinerInfos[entity.label] = [existing, entity.text];
                            //   } else if (existing is List) {
                            //     existing.add(entity.text);
                            //   }
                            // }
                            // print(glinerInfos);
                            // classificationResult.infos.addAll(glinerInfos);

//____Transformer Extraction
                            // print('Transformer work:');
                            // final time = DateTime.now();

                            // final extractor = MlServiceExtractor();
                            // await extractor.init();
                            // final extracted = await extractor.extract(ocrText: extractedText, documentType: classificationResult.documentType);
                            // classificationResult.infos.addAll(extracted.keyFacts);
                            // extractor.dispose();

                            // print('Extraction time: ${DateTime.now().difference(time).inSeconds} seconds');
                            // print('Classification result: ${classificationResult.documentType}, infos: ${classificationResult.infos}');
//__ Qwen Extraction                           
                            // print('Extract infos with Qwen');
                            // final qwenExtractor = MLServiceQwen();
                            // await qwenExtractor.initialize();

                            // final docType = classificationResult.documentType;
                            // // Run extraction
                            // final extractedInfos = await qwenExtractor.extract(
                            //   text: extractedText,
                            //   category: docType,
                            // );
                            // // Add extracted fields to your result map
                            // classificationResult.infos.addAll(
                            //   extractedInfos.infos,
                            // );

//__

                            setState(() => _isLoading = false);
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => ClassificationResultsScreen(result: classificationResult),
                              ),
                            );
                          },
                    icon: _isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.smart_toy),
                    label: Text(
                      _isLoading ? 'Verarbeitung...' : 'Verarbeiten',
                    ),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 15),
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(height: 12),

                // Clear button
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: _clearAllImages,
                    icon: const Icon(Icons.clear),
                    label: const Text('Alles löschen'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 15),
                      foregroundColor: Colors.red,
                      side: const BorderSide(color: Colors.red),
                    ),
                  ),
                ),
              ]
            ],
          ),
        ),
      ],
    );
  }
}

/// Dialog for taking multiple photos with camera
class _CameraMultiPageDialog extends StatefulWidget {
  final ImagePickerService imagePickerService;
  final Function(List<File>) onImagesSelected;

  const _CameraMultiPageDialog({
    required this.imagePickerService,
    required this.onImagesSelected,
  });

  @override
  State<_CameraMultiPageDialog> createState() => _CameraMultiPageDialogState();
}

class _CameraMultiPageDialogState extends State<_CameraMultiPageDialog> {
  List<File> _photos = [];
  bool _isLoading = false;

  Future<void> _takePhoto() async {
    setState(() => _isLoading = true);
    try {
      final file = await widget.imagePickerService.takePhotoWithCamera();
      if (file != null) {
        setState(() {
          _photos.add(file);
        });
      }
    } catch (e) {
      print('Error taking photo: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Padding(
            padding: EdgeInsets.all(20),
            child: Text(
              'Multi-Seiten Scan',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
          Expanded(
            child: _photos.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.image_not_supported,
                          size: 80,
                          color: Colors.grey.shade400,
                        ),
                        const SizedBox(height: 10),
                        const Text('Keine Fotos gemacht'),
                      ],
                    ),
                  )
                : GridView.builder(
                    padding: const EdgeInsets.all(20),
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      crossAxisSpacing: 10,
                      mainAxisSpacing: 10,
                    ),
                    itemCount: _photos.length,
                    itemBuilder: (context, index) {
                      return Stack(
                        children: [
                          Container(
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.orange),
                              borderRadius: BorderRadius.circular(8),
                              image: DecorationImage(
                                image: FileImage(_photos[index]),
                                fit: BoxFit.cover,
                              ),
                            ),
                          ),
                          Positioned(
                            top: 5,
                            left: 5,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.orange,
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Text(
                                'S. ${index + 1}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                          Positioned(
                            bottom: 5,
                            right: 5,
                            child: GestureDetector(
                              onTap: () {
                                setState(() {
                                  _photos.removeAt(index);
                                });
                              },
                              child: Container(
                                padding: const EdgeInsets.all(4),
                                decoration: BoxDecoration(
                                  color: Colors.red,
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: const Icon(
                                  Icons.close,
                                  color: Colors.white,
                                  size: 16,
                                ),
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                  ),
          ),
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _takePhoto,
                    icon: _isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.camera_alt),
                    label: Text(
                      _isLoading
                          ? 'Aufnahme...'
                          : 'Foto ${_photos.length + 1} machen',
                    ),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 15),
                      backgroundColor: Colors.orange,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () => Navigator.pop(context),
                        child: const Text('Abbrechen'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: _photos.isEmpty ? null : () {
                          widget.onImagesSelected(_photos);
                        },
                        child: Text('Bestätigen (${_photos.length})'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Screen 3: History
// class HistoryScreen extends StatelessWidget {
//   const HistoryScreen({Key? key}) : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return Center(
//       child: Column(
//         mainAxisAlignment: MainAxisAlignment.center,
//         children: [
//           const Icon(Icons.history, size: 80, color: Colors.orange),
//           const SizedBox(height: 20),
//           const Text(
//             'Processing History',
//             style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
//           ),
//           const SizedBox(height: 20),
//           const Text('No documents processed yet'),
//         ],
//       ),
//     );
//   }
// }