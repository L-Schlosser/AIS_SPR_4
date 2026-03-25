#!/bin/bash
~/Android/Sdk/platform-tools/adb shell run-as com.example.mobile_app cat /data/user/0/com.example.mobile_app/app_flutter/ocr_result.json > ocr_result.json
echo "saved to ocr_result.json"
