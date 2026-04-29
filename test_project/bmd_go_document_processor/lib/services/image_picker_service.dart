import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';


class ImagePickerService {
  final ImagePicker _imagePicker = ImagePicker();

  /// Take photo with camera
  Future<File?> takePhotoWithCamera() async {
    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.camera,
        imageQuality: 95,
        preferredCameraDevice: CameraDevice.rear,
      );

      if (pickedFile != null) {
        return File(pickedFile.path);
      }
      return null;
    } catch (e) {
      print('Error taking photo: $e');
      return null;
    }
  }

  /// Pick document from file manager
  Future<File?> pickDocumentFile() async {
    try {
      final FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx'],
        allowMultiple: false,
      );

      if (result != null && result.files.single.path != null) {
        return File(result.files.single.path!);
      }
      return null;
    } catch (e) {
      print('Error picking document: $e');
      return null;
    }
  }


  /// Pick or take multiple images for multi-page documents
  Future<List<File>> pickMultipleImagesForDocument({required ImageSource source}) async {
    try {
      if (source == ImageSource.gallery) {
        final List<XFile> pickedFiles = await _imagePicker.pickMultiImage(
          imageQuality: 95,
        );
        return pickedFiles.map((file) => File(file.path)).toList();
      } else {
        return [];
      }
    } catch (e) {
      print('Error picking multiple images: $e');
      return [];
    }
  }
}