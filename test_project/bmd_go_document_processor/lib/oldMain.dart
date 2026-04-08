import 'package:flutter/material.dart';
import 'dart:io';
import 'services/image_picker_service.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
// import 'services/camera_service.dart';
import 'services/ml_service.dart';
import 'services/ml_service_test.dart';
import 'screens/classification_results_screen.dart';

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

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({Key? key}) : super(key: key);

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _selectedIndex = 0;

  // Define screens
  static const List<Widget> _screens = [
    // CameraScreen(),
    UploadScreen(),
    HistoryScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('BMD Dokumenterfassung'),
        centerTitle: true,
        elevation: 2,
      ),
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.upload_file),
            label: 'Upload',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history),
            label: 'History',
          ),
        ],
      ),
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
  File? _selectedImage;
  bool _isLoading = false;

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
              subtitle: const Text('Scan mit Kamera'),
              onTap: () {
                Navigator.pop(context);
                _takePhotoWithCamera();
              },
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.image, color: Colors.blue, size: 30),
              title: const Text('Galerie'),
              subtitle: const Text('Datei aus Galerie'),
              onTap: () {
                Navigator.pop(context);
                _pickImageFromGallery();
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _takePhotoWithCamera() async {
    setState(() => _isLoading = true);
    try {
      final file = await _imagePickerService.takePhotoWithCamera();
      if (file != null) {
        setState(() => _selectedImage = file);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Photo captured successfully')),
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

  Future<void> _pickImageFromGallery() async {
    setState(() => _isLoading = true);
    try {
      final file = await _imagePickerService.pickImageFromGallery();
      if (file != null) {
        setState(() => _selectedImage = file);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Image selected successfully')),
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

  Future<void> _pickDocument() async {
    setState(() => _isLoading = true);
    try {
      final file = await _imagePickerService.pickDocumentFile();
      if (file != null) {
        setState(() => _selectedImage = file);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Document selected successfully')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _clearImage() {
    setState(() => _selectedImage = null);
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
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

            if (_selectedImage == null) ...[
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
              // Buttons
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isLoading ? null : _pickDocument,
                      icon: const Icon(Icons.file_download_outlined),
                      label: const Text('Hochladen'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        backgroundColor: Colors.orange,  //shade 300
                        foregroundColor: Colors.white,   //shade 700
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
              // Image selected
              Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.orange, width: 2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.file(
                    _selectedImage!,
                    height: 300,
                    width: double.infinity,
                    fit: BoxFit.contain,
                  ),
                ),
              ),
              const SizedBox(height: 30),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _isLoading ? null : () {
                    print('Classifying...');
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

              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _clearImage,
                  icon: const Icon(Icons.clear),
                  label: const Text('Löschen'),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 15),
                    foregroundColor: Colors.red,
                    side: const BorderSide(color: Colors.red),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}


// Screen 3: History
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.history, size: 80, color: Colors.orange),
          const SizedBox(height: 20),
          const Text(
            'Processing History',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 20),
          const Text('No documents processed yet'),
        ],
      ),
    );
  }
}