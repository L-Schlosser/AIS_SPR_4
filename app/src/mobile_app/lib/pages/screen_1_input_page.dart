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
  final PageController _pageController = PageController();
  int _currentPreviewIndex = 0;
  bool _isLoading = false;

  Future<void> _pickFromCamera() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.camera,
      imageQuality: 100,
    );

    if (image == null) return;

    _addImageAndJumpToIt(File(image.path));
  }

  void _addImageAndJumpToIt(File file) {
    setState(() {
      _selectedImages.add(file);
      _currentPreviewIndex = _selectedImages.length - 1;
    });

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_pageController.hasClients) {
        _pageController.animateToPage(
          _currentPreviewIndex,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeInOut,
        );
      }
    });
  }

  Future<void> _pickFromGallery() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.gallery,
    );

    if (image == null) return;

    _addImageAndJumpToIt(File(image.path));
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

      _addImageAndJumpToIt(file);
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

      _addImageAndJumpToIt(file);
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _removeImage(int index) {
    setState(() {
      _selectedImages.removeAt(index);
      _fixPreviewIndexAfterRemoval();
    });

    if (_selectedImages.isNotEmpty) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_pageController.hasClients) {
          _pageController.jumpToPage(_currentPreviewIndex);
        }
      });
    }
  }

  void _fixPreviewIndexAfterRemoval() {
    if (_selectedImages.isEmpty) {
      _currentPreviewIndex = 0;
      return;
    }

    if (_currentPreviewIndex >= _selectedImages.length) {
      _currentPreviewIndex = _selectedImages.length - 1;
    }
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

  Widget _buildImagePreviewGallery() {
    if (_selectedImages.isEmpty) {
      return Container(
        width: double.infinity,
        height: 260,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade400),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(child: Text('Noch keine Bilder ausgewählt')),
      );
    }

    return Column(
      children: [
        SizedBox(
          height: 260,
          child: Stack(
            children: [
              PageView.builder(
                controller: _pageController,
                itemCount: _selectedImages.length,
                onPageChanged: (index) {
                  setState(() {
                    _currentPreviewIndex = index;
                  });
                },
                itemBuilder: (context, index) {
                  final image = _selectedImages[index];

                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade400),
                        borderRadius: BorderRadius.circular(12),
                        color: Colors.white,
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: Image.file(
                        image,
                        fit: BoxFit.contain,
                        width: double.infinity,
                        gaplessPlayback: true,
                      ),
                    ),
                  );
                },
              ),
              Positioned(
                top: 8,
                right: 8,
                child: Material(
                  color: Colors.black54,
                  shape: const CircleBorder(),
                  child: InkWell(
                    onTap: () => _removeImage(_currentPreviewIndex),
                    customBorder: const CircleBorder(),
                    child: const Padding(
                      padding: EdgeInsets.all(8),
                      child: Icon(Icons.close, color: Colors.white, size: 20),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Text('${_currentPreviewIndex + 1} / ${_selectedImages.length}'),
      ],
    );
  }

  @override
  void dispose() {
    _pageController.dispose();
    _ocrService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final canContinue = _selectedImages.isNotEmpty && !_isLoading;

    return Scaffold(
      appBar: AppBar(title: const Text('Startseite')),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: ListView(
              children: [
                Text(
                  'Bilder auswählen',
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                _buildActionButton(
                  icon: Icons.photo_camera_outlined,
                  text: 'Foto aufnehmen',
                  onPressed: _isLoading ? () {} : _pickFromCamera,
                ),
                const SizedBox(height: 8),
                _buildActionButton(
                  icon: Icons.upload_file_outlined,
                  text: 'Bild vom Gerät hochladen',
                  onPressed: _isLoading ? () {} : _pickFromGallery,
                ),
                const SizedBox(height: 8),
                _buildActionButton(
                  icon: Icons.science_outlined,
                  text: 'bill.jpg als Testbild laden',
                  onPressed: _isLoading ? () {} : _loadTestImage,
                ),
                const SizedBox(height: 8),
                _buildActionButton(
                  icon: Icons.science_outlined,
                  text: 'bill1.jpg als Testbild laden',
                  onPressed: _isLoading ? () {} : _loadTestImage2,
                ),
                const SizedBox(height: 16),
                _buildImagePreviewGallery(),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: (_selectedImages.isNotEmpty && !_isLoading)
                        ? _continueToScreen2
                        : null,
                    child: const Text('Weiter'),
                  ),
                ),
              ],
            ),
          ),
          if (_isLoading)
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
