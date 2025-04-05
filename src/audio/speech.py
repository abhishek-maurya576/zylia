"""
ZYLIA - Speech Processing Module
Handles Speech-to-Text (STT) and Text-to-Speech (TTS) conversions
"""
import logging
import pyttsx3
import os
import threading
import re
import random
from pathlib import Path
from faster_whisper import WhisperModel

logger = logging.getLogger("ZYLIA.Audio.Speech")

class SpeechToText:
    """Converts speech to text using Whisper model"""
    
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        """Initialize the speech-to-text converter
        
        Args:
            model_size: Whisper model size (default: "tiny" for faster loading)
            device: Device to run the model on (default: "cpu")
            compute_type: Computation type (default: "int8")
        """
        logger.info(f"Initializing Whisper model (size={model_size}, device={device})")
        try:
            # Check if model is available in models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Initialize the Whisper model
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(models_dir)
            )
            logger.info("Whisper model initialized successfully")
            
            # Store whether model was initialized
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {e}")
            self.initialized = False
    
    def transcribe(self, audio_file_path):
        """Transcribe audio file to text
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not self.initialized:
            return "Sorry, speech recognition is not available. Please type your message instead."
            
        if not audio_file_path or not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return "I couldn't hear anything. Please try again."
        
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            print("Transcribing audio...")
            
            # Perform transcription with more lenient settings
            segments, info = self.model.transcribe(
                audio_file_path, 
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),  # Shorter silence threshold
                temperature=0.2,  # Lower temperature for more accurate transcription
                beam_size=5,      # Larger beam size for better results
                word_timestamps=True,  # Get word-level timestamps for better segmentation
                condition_on_previous_text=True,  # Use context
                no_speech_threshold=0.3,  # More lenient no-speech detection
                compression_ratio_threshold=2.4  # More lenient compression ratio
            )
            
            logger.info(f"Transcription language: {info.language} with probability {info.language_probability:.2f}")
            
            # Join all segments to get the full text
            transcription = " ".join([segment.text for segment in segments])
            
            # Clean up the transcription
            transcription = transcription.strip()
            
            # If transcription is empty, return a default message
            if not transcription or transcription.isspace():
                logger.warning("Empty transcription result")
                print("No speech detected in the recording. Please try again.")
                return "I didn't catch that. Could you please speak again?"
                
            logger.info(f"Transcription result: {transcription}")
            print(f"Transcription: \"{transcription}\"")
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "Sorry, I had trouble understanding that. Please try again."


class TextToSpeech:
    """Converts text to speech using pyttsx3 with threading to avoid GIL issues"""
    
    def __init__(self, voice=None, rate=190, volume=1.0):
        """Initialize the text-to-speech converter
        
        Args:
            voice: Voice ID to use (default: None = system default)
            rate: Speech rate (default: 190 - normal speaking pace)
            volume: Speech volume (default: 1.0 - normal volume)
        """
        logger.info("Initializing text-to-speech engine")
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.engine = None
        self.init_engine()
        
    def init_engine(self):
        """Initialize the TTS engine in the current thread"""
        try:
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Try to find and use a female voice with priority
            found_female_voice = False
            voices = self.engine.getProperty('voices')
            
            # First, try to find Microsoft Zira voice (high quality female voice on Windows)
            for voice in voices:
                voice_id = voice.id.lower()
                voice_name = voice.name.lower() if hasattr(voice, 'name') else ""
                
                if "zira" in voice_id or "zira" in voice_name:
                    self.engine.setProperty('voice', voice.id)
                    logger.info(f"Selected Zira voice: {voice.name}")
                    found_female_voice = True
                    break
            
            # If Zira not found, try other female voices
            if not found_female_voice:
                for voice in voices:
                    voice_id = voice.id.lower()
                    voice_name = voice.name.lower() if hasattr(voice, 'name') else ""
                    
                    # Look for any female-sounding voice identifiers
                    if any(term in voice_id or term in voice_name for term in ["female", "woman", "f", "girl", "en-us-f", "microsoft-en-us-aria"]):
                        self.engine.setProperty('voice', voice.id)
                        logger.info(f"Selected female voice: {voice.name}")
                        found_female_voice = True
                        break
            
            # If no female voice found, just log it
            if not found_female_voice:
                logger.info("No specific female voice found, using default voice")
            
            # Log available voices
            logger.info(f"Available voices: {len(voices)}")
            for i, v in enumerate(voices):
                voice_name = v.name if hasattr(v, 'name') else "Unknown"
                logger.debug(f"Voice {i}: {v.id} ({voice_name})")
            
            logger.info("Text-to-speech engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing text-to-speech engine: {e}")
            self.engine = None
    
    def _create_whisper_effect(self, text):
        """Create a nurturing, gentle voice effect by modifying the text
        
        Args:
            text: Original text
            
        Returns:
            Modified text with gentle, caring speech markers
        """
        words = text.split()
        
        # Add soft emphasis on emotional or supportive words
        emotional_words = ["sorry", "love", "care", "feel", "understand", "here", "help", 
                         "hug", "listen", "support", "precious", "important", "worry"]
        
        for i in range(len(words)):
            word_lower = words[i].lower().strip(".,!?")
            
            # Add gentle pauses for a caring rhythm, not too frequent
            if i > 0:
                if i % 4 == 0:  # Every 4th word gets a soft pause
                    words[i] = ", " + words[i]
                elif i % 8 == 0:  # Every 8th word gets a longer, thoughtful pause
                    words[i] = "... " + words[i]
            
            # Add slight emphasis to emotional words
            if any(ew in word_lower for ew in emotional_words):
                # Don't add pauses too frequently
                if i > 0 and not words[i].startswith(",") and not words[i].startswith("."):
                    words[i] = ", " + words[i] + ","
        
        # Add natural-sounding expressions of empathy
        caring_text = " ".join(words)
        
        # Soften tone for a nurturing effect
        caring_text = caring_text.replace("!", ".")
        
        # Add a gentle beginning and ending for a caring tone
        caring_text = ".. " + caring_text + ".."
        
        return caring_text
    
    def _speak_in_thread(self, text):
        """Internal function to run TTS in a separate thread to avoid GIL issues"""
        # Create a new engine instance in this thread
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        
        # Get available voices
        voices = engine.getProperty('voices')
        voice_set = False
        
        # Try to apply the same voice selection as in init_engine
        for voice in voices:
            voice_id = voice.id.lower()
            voice_name = voice.name.lower() if hasattr(voice, 'name') else ""
            
            if "zira" in voice_id or "zira" in voice_name:
                engine.setProperty('voice', voice.id)
                voice_set = True
                break
        
        if not voice_set:
            for voice in voices:
                voice_id = voice.id.lower()
                voice_name = voice.name.lower() if hasattr(voice, 'name') else ""
                
                if any(term in voice_id or term in voice_name for term in ["female", "woman", "f", "girl", "en-us-f", "microsoft-en-us-aria"]):
                    engine.setProperty('voice', voice.id)
                    voice_set = True
                    break
        
        # Speak the text directly without whisper effect
        engine.say(text)
        engine.runAndWait()
        
        # Clean up
        engine.stop()
    
    def speak(self, text):
        """Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
        """
        if not text:
            logger.warning("Empty text provided for speech synthesis")
            return
        
        try:
            logger.info(f"Converting text to speech: {text[:50]}...")
            
            # Start a new thread for TTS to avoid GIL issues
            tts_thread = threading.Thread(target=self._speak_in_thread, args=(text,))
            tts_thread.daemon = True  # Allow the thread to be terminated when the main program exits
            tts_thread.start()
            
            logger.info("Speech playback started in separate thread")
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
    
    def save_to_file(self, text, output_file):
        """Save synthesized speech to a file
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
        """
        if not self.engine:
            self.init_engine()
            
        try:
            logger.info(f"Saving speech to file: {output_file}")
            
            # Create a new engine instance for this operation
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            if self.voice:
                engine.setProperty('voice', self.voice)
                
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            engine.stop()
            
            logger.info("Speech saved to file successfully")
        except Exception as e:
            logger.error(f"Error saving speech to file: {e}")