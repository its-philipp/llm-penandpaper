#!/usr/bin/env python3
"""
Debug script to test Edge TTS and identify audio issues.
"""

import asyncio
import edge_tts
import os
import time
import subprocess

async def test_edge_tts():
    """Test Edge TTS generation and playback."""
    print("🔊 Testing Edge TTS...")
    
    # Test text
    text = "Hello! This is a test of the text to speech system."
    voice = "en-US-GuyNeural"
    temp_wav = f"debug_speech_{int(time.time())}.wav"
    
    try:
        print(f"Generating audio for: '{text}'")
        print(f"Using voice: {voice}")
        
        # Generate audio
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_wav)
        
        print(f"Audio file generated: {temp_wav}")
        
        # Check file size
        if os.path.exists(temp_wav):
            size = os.path.getsize(temp_wav)
            print(f"File size: {size} bytes")
            
            if size > 0:
                print("✅ Audio file generated successfully!")
                
                # Try to play with different players
                audio_players = ["aplay", "paplay", "mpg123", "ffplay"]
                
                for player in audio_players:
                    print(f"Trying to play with {player}...")
                    try:
                        result = subprocess.run([player, temp_wav], 
                                              capture_output=True, 
                                              text=True, 
                                              timeout=10)
                        if result.returncode == 0:
                            print(f"✅ {player} played successfully!")
                            break
                        else:
                            print(f"❌ {player} failed: {result.stderr}")
                    except FileNotFoundError:
                        print(f"❌ {player} not found")
                    except subprocess.TimeoutExpired:
                        print(f"❌ {player} timed out")
                else:
                    print("❌ No audio player worked")
            else:
                print("❌ Audio file is empty!")
        else:
            print("❌ Audio file was not created!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Keep the file for inspection
        print(f"Audio file kept for inspection: {temp_wav}")

if __name__ == "__main__":
    asyncio.run(test_edge_tts()) 