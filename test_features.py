#!/usr/bin/env python3
"""
Test script to verify TTS and image generation features are working.
"""

import os
import sys
import time

# Set ROCm environment variables
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HCC_AMDGPU_TARGET'] = 'gfx1030'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCM_PATH'] = '/opt/rocm'

def test_tts():
    """Test text-to-speech functionality."""
    print("🔊 Testing Text-to-Speech...")
    try:
        from tts_utils import TTSManager
        tts = TTSManager()
        tts.set_speaker("p230")
        tts.start_tts("Hello! This is a test of the text to speech system.")
        print("✅ TTS test completed - you should have heard audio!")
        time.sleep(3)  # Wait for audio to finish
        tts.stop_tts()
        return True
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

def test_image_generation():
    """Test image generation functionality."""
    print("🎨 Testing Image Generation...")
    try:
        from image_gen_utils import generate_scene_image
        image_path = generate_scene_image("A majestic dragon flying over a medieval castle at sunset")
        if image_path and os.path.exists(image_path):
            print(f"✅ Image generated successfully: {image_path}")
            return True
        else:
            print("❌ Image generation failed - no image file created")
            return False
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection."""
    print("🖥️ Testing GPU Detection...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            return True
        else:
            print("❌ No GPU detected")
            return False
    except Exception as e:
        print(f"❌ GPU detection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Eldoria Quest Features...")
    print("=" * 50)
    
    # Test GPU detection
    gpu_ok = test_gpu_detection()
    print()
    
    # Test TTS
    tts_ok = test_tts()
    print()
    
    # Test image generation
    img_ok = test_image_generation()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results:")
    print(f"GPU Detection: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"Text-to-Speech: {'✅ PASS' if tts_ok else '❌ FAIL'}")
    print(f"Image Generation: {'✅ PASS' if img_ok else '❌ FAIL'}")
    
    if all([gpu_ok, tts_ok, img_ok]):
        print("\n🎉 All tests passed! Your setup is working perfectly!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.") 