import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/ocr_service.dart';
import 'screen_2_review_page.dart';

class Screen1InputPage extends StatefulWidget {
  const Screen1InputPage({super.key});

  @override
  State<Screen1InputPage> createState() => _Screen1InputPageState();
}

class _Screen1InputPageState extends State<Screen1InputPage> {
  final ImagePicker _imagePicker = ImagePicker();
  final OcrService _ocrService = OcrService();

  final List<File> _selectedImages = [];
  bool _isLoading = false;

  Future<void> _pickFromCamera() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.camera,
      imageQuality: 100,
    );

    if (image == null) return;

    setState(() {
      _selectedImages.add(File(image.path));
    });
  }

  Future<void> _pickFromGallery() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.gallery,
    );

    if (image == null) return;

    setState(() {
      _selectedImages.add(File(image.path));
    });
  }

  Future<void> _loadTestImage() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final file = await _ocrService.copyAssetToFile(
        'assets/bill.jpg',
        'bill.jpg',
      );

      setState(() {
        _selectedImages.add(file);
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _loadTestImage2() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final file = await _ocrService.copyAssetToFile(
        'assets/bill1.jpg',
        'bill1.jpg',
      );

      setState(() {
        _selectedImages.add(file);
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _removeImage(int index) {
    setState(() {
      _selectedImages.removeAt(index);
    });
  }

  Future<void> _continueToScreen2() async {
    if (_selectedImages.isEmpty) return;

    final result = await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) =>
            Screen2ReviewPage(imageFiles: List<File>.from(_selectedImages)),
      ),
    );

    if (!mounted) return;

    if (result == screen2ResultNeedNewImages) {
      setState(() {
        _selectedImages.clear();
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Keine Bilder mehr vorhanden. Bitte neue Bilder auswählen.',
          ),
        ),
      );
    } else if (result == screen2ResultSavedAll) {
      setState(() {
        _selectedImages.clear();
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Alle Dokumente wurden gespeichert.')),
      );
    }
  }

  Widget _buildActionButton({
    required IconData icon,
    required String text,
    required VoidCallback onPressed,
  }) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon),
        label: Text(text),
      ),
    );
  }

  Widget _buildImagePreviewGrid() {
    if (_selectedImages.isEmpty) {
      return Container(
        width: double.infinity,
        height: 220,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade400),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(child: Text('Noch keine Bilder ausgewählt')),
      );
    }

    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: List.generate(_selectedImages.length, (index) {
        final image = _selectedImages[index];

        return Stack(
          children: [
            Container(
              width: 150,
              height: 200,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(12),
                color: Colors.white,
              ),
              clipBehavior: Clip.antiAlias,
              child: Image.file(image, fit: BoxFit.contain),
            ),
            Positioned(
              top: 6,
              right: 6,
              child: Material(
                color: Colors.black54,
                shape: const CircleBorder(),
                child: InkWell(
                  onTap: () => _removeImage(index),
                  customBorder: const CircleBorder(),
                  child: const Padding(
                    padding: EdgeInsets.all(6),
                    child: Icon(Icons.close, color: Colors.white, size: 18),
                  ),
                ),
              ),
            ),
          ],
        );
      }),
    );
  }

  @override
  void dispose() {
    _ocrService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final canContinue = _selectedImages.isNotEmpty && !_isLoading;

    return Scaffold(
      appBar: AppBar(title: const Text('Startseite')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: ListView(
                children: [
                  Text(
                    'Bilder auswählen',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildActionButton(
                    icon: Icons.photo_camera_outlined,
                    text: 'Foto aufnehmen',
                    onPressed: _pickFromCamera,
                  ),
                  const SizedBox(height: 8),
                  _buildActionButton(
                    icon: Icons.upload_file_outlined,
                    text: 'Bild vom Gerät hochladen',
                    onPressed: _pickFromGallery,
                  ),
                  const SizedBox(height: 8),
                  _buildActionButton(
                    icon: Icons.science_outlined,
                    text: 'bill.jpg als Testbild laden',
                    onPressed: _loadTestImage,
                  ),
                  const SizedBox(height: 8),
                  _buildActionButton(
                    icon: Icons.science_outlined,
                    text: 'bill1.jpg als Testbild laden',
                    onPressed: _loadTestImage2,
                  ),
                  const SizedBox(height: 16),
                  _buildImagePreviewGrid(),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: canContinue ? _continueToScreen2 : null,
                      child: const Text('Weiter'),
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}
