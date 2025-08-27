import os
import subprocess
import threading
import time
import asyncio
import edge_tts
import random

# Voice mapping for different speakers by language
EDGE_VOICE_MAP = {
    "en": {
        "p230": "en-US-GuyNeural",
        "p225": "en-US-ChristopherNeural",
        "p226": "en-US-EricNeural",
        "p227": "en-US-DavisNeural",
    },
    "de": {
        # Valid German voices
        "p230": "de-DE-ConradNeural",
        "p225": "de-DE-KatjaNeural",
        # Fallback to one of the above if not present
        "p226": "de-DE-ConradNeural",
        "p227": "de-DE-KatjaNeural",
    }
}

class TTSManager:
    """Manages text-to-speech functionality with multiple voice options."""
    
    def __init__(self):
        self.audio_process = None
        self.current_speaker = "p230"
        self.current_language = "en"  # 'en' or 'de'
        self._stop_flag = threading.Event()
        self._speaking_thread = None

    def _speak_edgetts(self, text: str, speaker: str) -> None:
        """Generate and play speech using Edge TTS."""
        lang_map = EDGE_VOICE_MAP.get(self.current_language, EDGE_VOICE_MAP["en"]) 
        voice = lang_map.get(speaker, lang_map.get("p230", "en-US-GuyNeural"))
        temp_audio = f"temp_speech_{int(time.time())}.mp3"
        temp_alt_audio = None  # used by fallbacks (e.g., .aiff on macOS, .wav on Linux)
        
        try:
            # Generate audio asynchronously with retry/backoff (handles transient 403s, network hiccups)
            async def generate_audio():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(temp_audio)

            max_retries = 3
            base_delay = 0.8
            for attempt in range(1, max_retries + 1):
                if self._stop_flag.is_set():
                    return
                try:
                    asyncio.run(generate_audio())
                    break
                except Exception as e:
                    if attempt == max_retries:
                        print(f"[TTS Error] EdgeTTS failed after {attempt} attempts: {e}")
                        # macOS fallback using 'say'
                        try:
                            if os.uname().sysname == 'Darwin':
                                temp_alt_audio = f"temp_speech_{int(time.time())}.aiff"
                                # Select a reasonable default voice per language
                                mac_voice = "Samantha" if self.current_language == "en" else "Anna"
                                subprocess.run(["say", "-v", mac_voice, "-o", temp_alt_audio, text], check=True)
                                self.audio_process = subprocess.Popen(["afplay", temp_alt_audio])
                                return
                        except Exception:
                            pass

                        # Linux fallback using 'espeak'/'espeak-ng' or 'pico2wave'
                        try:
                            if os.name != 'nt' and os.uname().sysname != 'Darwin':
                                # Try espeak (direct playback)
                                try:
                                    espeak_lang = "en" if self.current_language == "en" else "de"
                                    self.audio_process = subprocess.Popen(["espeak", "-s", "165", "-v", espeak_lang, text])
                                    return
                                except FileNotFoundError:
                                    # Try espeak-ng if espeak not found
                                    try:
                                        espeak_lang = "en" if self.current_language == "en" else "de"
                                        self.audio_process = subprocess.Popen(["espeak-ng", "-s", "165", "-v", espeak_lang, text])
                                        return
                                    except FileNotFoundError:
                                        pass
                                # Try pico2wave -> wav -> play
                                try:
                                    temp_alt_audio = f"temp_speech_{int(time.time())}.wav"
                                    pico_lang = "en-US" if self.current_language == "en" else "de-DE"
                                    subprocess.run(["pico2wave", "-l", pico_lang, "-w", temp_alt_audio, text], check=True)
                                    for player in ["aplay", "paplay", "ffplay"]:
                                        try:
                                            if player == "ffplay":
                                                self.audio_process = subprocess.Popen([player, "-nodisp", "-autoexit", temp_alt_audio])
                                            else:
                                                self.audio_process = subprocess.Popen([player, temp_alt_audio])
                                            break
                                        except FileNotFoundError:
                                            continue
                                    return
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        return
                    delay = min(base_delay * (2 ** (attempt - 1)), 4.0) + (0.1 * attempt)
                    print(f"[TTS Warning] EdgeTTS error (attempt {attempt}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            # Check if stop was requested during generation
            if self._stop_flag.is_set():
                return
            
            # Play the audio (cross-platform)
            if os.name == 'nt':  # Windows
                self.audio_process = subprocess.Popen(["start", temp_audio], shell=True)
            elif os.uname().sysname == 'Darwin':  # macOS
                self.audio_process = subprocess.Popen(["afplay", temp_audio])
            else:  # Linux
                # Try different audio players in order of preference for MP3
                audio_players = ["mpg123", "ffplay", "paplay"]
                for player in audio_players:
                    try:
                        if player == "ffplay":
                            self.audio_process = subprocess.Popen([player, "-nodisp", "-autoexit", temp_audio])
                        else:
                            self.audio_process = subprocess.Popen([player, temp_audio])
                        break
                    except FileNotFoundError:
                        continue
                else:
                    print(f"[TTS Warning] No MP3 audio player found. Tried: {audio_players}")
                    return
            
            # Wait for completion or stop signal
            while self.audio_process and self.audio_process.poll() is None and not self._stop_flag.is_set():
                time.sleep(0.1)
            
            # Stop if requested
            if self.audio_process and self.audio_process.poll() is None:
                self.audio_process.terminate()
                self.audio_process.wait()
                
        except Exception as e:
            print(f"[TTS Error] EdgeTTS failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except OSError:
                    pass  # File might already be deleted
            if temp_alt_audio and os.path.exists(temp_alt_audio):
                try:
                    os.remove(temp_alt_audio)
                except OSError:
                    pass

    def start_tts(self, text: str) -> None:
        """Start text-to-speech for the given text."""
        self.stop_tts()  # Stop any current speech
        self._stop_flag.clear()
        
        # Start new speech thread
        self._speaking_thread = threading.Thread(
            target=self._speak_edgetts, 
            args=(text, self.current_speaker),
            daemon=True
        )
        self._speaking_thread.start()

    def stop_tts(self) -> None:
        """Stop current text-to-speech."""
        self._stop_flag.set()
        
        if self.audio_process and self.audio_process.poll() is None:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.audio_process.kill()
            except Exception as e:
                print(f"Error stopping audio process: {e}")
        
        self.audio_process = None

    def set_speaker(self, speaker_id: str) -> None:
        """Set the current speaker voice."""
        lang_map = EDGE_VOICE_MAP.get(self.current_language, EDGE_VOICE_MAP["en"]) 
        if speaker_id in lang_map:
            self.current_speaker = speaker_id
            resolved = lang_map.get(speaker_id, lang_map.get("p230"))
            print(f"TTS speaker set to: {speaker_id} ({resolved})")
        else:
            print(f"Unknown speaker ID: {speaker_id}. Keeping current speaker.")

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return (self.audio_process is not None and 
                self.audio_process.poll() is None and 
                not self._stop_flag.is_set())

    def set_language(self, language_code: str) -> None:
        """Set language preference for TTS ('en' or 'de')."""
        if language_code in ("en", "de"):
            self.current_language = language_code
            # Emit a log line for visibility
            print(f"TTS language set to: {language_code}")
        else:
            print(f"Unsupported language code: {language_code}. Keeping {self.current_language}.")

# Create global TTS manager instance
tts_manager = TTSManager() 