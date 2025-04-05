"""
ZYLIA - Neural Text-to-Speech Module
Provides high-quality speech synthesis using neural networks
"""
import os
import logging
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

logger = logging.getLogger("ZYLIA.Audio.NeuralTTS")

class NeuralTextToSpeech:
    """High-quality neural text-to-speech using Mozilla TTS"""
    
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                 vocoder_name="vocoder_models/en/ljspeech/univnet", 
                 use_cuda=False, 
                 whisper_effect=False,
                 voice_speed=1.0):
        """Initialize the neural TTS system
        
        Args:
            model_name: TTS model to use
            vocoder_name: Vocoder model to use
            use_cuda: Whether to use GPU acceleration
            whisper_effect: Whether to apply whisper effect (default: False)
            voice_speed: Speed multiplier (default: 1.0 - normal speed)
        """
        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self.use_cuda = use_cuda
        self.whisper_effect = whisper_effect
        self.voice_speed = voice_speed
        self.tts = None
        self.models_dir = Path("models/tts")
        self.output_dir = Path("temp")
        self.output_dir.mkdir(exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Define available female voice models for more pleasant output
        self.female_voice_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",  # Default female voice
            "tts_models/en/vctk/vits",               # Multi-speaker model with female voices
            "tts_models/en/jenny/jenny",             # Jenny voice
            "tts_models/en/vctk/fast_pitch"          # Multi-speaker British English
        ]
        
        self.init_tts()
    
    def init_tts(self):
        """Initialize the TTS engine in the current thread"""
        try:
            logger.info(f"Initializing Neural TTS with model {self.model_name}")
            
            # Ensure models directory exists
            self.models_dir.mkdir(exist_ok=True, parents=True)
            
            # Attempt to load the TTS system
            try:
                # Try to use a female voice model first
                if self.model_name not in self.female_voice_models:
                    for model in self.female_voice_models:
                        try:
                            self.tts = TTS(model_name=model, 
                                     vocoder_name=self.vocoder_name,
                                     progress_bar=False,
                                     gpu=self.use_cuda)
                            logger.info(f"Loaded female voice model: {model}")
                            self.model_name = model
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load model {model}: {e}")
                            continue
                
                # If no female models worked, try the specified model
                if self.tts is None:
                    self.tts = TTS(model_name=self.model_name, 
                             vocoder_name=self.vocoder_name,
                             progress_bar=False,
                             gpu=self.use_cuda)
                    logger.info(f"Loaded specified model: {self.model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load specified model, trying default: {e}")
                # If that fails, use the default model
                self.tts = TTS(progress_bar=False, gpu=self.use_cuda)
                logger.info("Loaded default TTS model")
            
            # Get available voices if using a multi-speaker model
            self.speakers = None
            if self.tts.is_multi_speaker:
                self.speakers = self.tts.speakers
                logger.info(f"Available speakers: {', '.join(self.speakers[:5])}...")
                
                # Try to select a female speaker
                female_keywords = ["female", "woman", "girl", "lady"]
                for speaker in self.speakers:
                    if any(keyword in speaker.lower() for keyword in female_keywords):
                        self.default_speaker = speaker
                        logger.info(f"Selected female speaker: {speaker}")
                        break
                else:
                    # If no clearly female speaker, use the first one
                    self.default_speaker = self.speakers[0]
                    logger.info(f"Using default speaker: {self.default_speaker}")
            
            # Get available emotions if the model supports it
            self.emotions = None
            if hasattr(self.tts, 'emotions') and self.tts.emotions:
                self.emotions = self.tts.emotions
                logger.info(f"Available emotions: {', '.join(self.emotions)}")
                # Use a gentle emotion if available
                gentle_emotions = ["calm", "soft", "gentle", "peaceful"]
                for emotion in gentle_emotions:
                    if emotion in self.emotions:
                        self.default_emotion = emotion
                        break
                else:
                    self.default_emotion = self.emotions[0]
                
            logger.info("Neural TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Neural TTS: {e}")
            self.tts = None
            raise
    
    def _create_whisper_effect(self, wav, sr):
        """Apply whisper effect to the audio
        
        Args:
            wav: Audio waveform
            sr: Sample rate
            
        Returns:
            Modified waveform
        """
        if not self.whisper_effect:
            return wav, sr
            
        try:
            # Scale down volume for whisper effect
            wav = wav * 0.7
            
            # Add a bit of breathiness (gentle high-pass filter)
            from scipy import signal
            b, a = signal.butter(3, 0.03, 'highpass', analog=False)
            breath = signal.filtfilt(b, a, wav) * 0.05
            wav = wav + breath
            
            # Slow down the audio for whisper effect without changing pitch
            if self.voice_speed < 1.0:
                import librosa
                wav = librosa.effects.time_stretch(wav, rate=self.voice_speed)
            
            return wav, sr
            
        except Exception as e:
            logger.error(f"Error applying whisper effect: {e}")
            return wav, sr
    
    def preprocess_text(self, text):
        """Apply preprocessing to text for better synthesis with caring tone
        
        Args:
            text: Input text
            
        Returns:
            Processed text with nurturing, emotional markers
        """
        # Add strategic pauses for caring, emotional effect
        if self.whisper_effect:
            # Identify emotional or supportive words
            emotional_words = ["love", "care", "feel", "here", "understand", 
                              "sorry", "support", "hug", "listen", "important", "precious"]
            
            # Add pauses between sentences for thoughtfulness
            text = text.replace(". ", "... ")
            text = text.replace("! ", "... ")
            text = text.replace("? ", "... ")
            
            # Soften exclamations for gentleness
            text = text.replace("!", ".")
            
            # Add emphasis to emotional words using SSML-like markers
            for word in emotional_words:
                if f" {word} " in text.lower():
                    text = text.lower().replace(f" {word} ", f"... {word}... ")
            
            # Add a gentle beginning and ending
            if not text.startswith("..."):
                text = "... " + text
            if not text.endswith("..."):
                text = text + " ..."
                
        return text
            
    def _speak_in_thread(self, text, output_path=None):
        """Internal function to run TTS in a separate thread
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file (optional)
        """
        try:
            # Use original text without preprocessing if whisper effect is disabled
            processed_text = self.preprocess_text(text) if self.whisper_effect else text
            
            # Generate speech based on model capabilities
            kwargs = {}
            
            if self.tts.is_multi_speaker and hasattr(self, 'default_speaker'):
                kwargs['speaker'] = self.default_speaker
                
            if hasattr(self, 'emotions') and self.emotions and hasattr(self, 'default_emotion'):
                kwargs['emotion'] = self.default_emotion
                
            # Generate the audio
            if output_path:
                # Generate directly to file
                self.tts.tts_to_file(processed_text, file_path=output_path, **kwargs)
                logger.info(f"Generated speech to file: {output_path}")
            else:
                # Generate and play
                wav = self.tts.tts(processed_text, **kwargs)
                
                # Apply whisper effect if enabled (typically off for normal voice)
                if self.whisper_effect and isinstance(wav, tuple) and len(wav) == 2:
                    wav, sr = self._create_whisper_effect(wav[0], wav[1])
                elif self.whisper_effect and isinstance(wav, np.ndarray):
                    wav, sr = self._create_whisper_effect(wav, 22050)
                else:
                    # Use the audio as-is without modifications
                    wav, sr = wav if isinstance(wav, tuple) and len(wav) == 2 else (wav, 22050)
                
                # Play the audio
                sd.play(wav, sr)
                sd.wait()
                logger.info("Finished playing synthesized speech")
        
        except Exception as e:
            logger.error(f"Error in TTS thread: {e}")
    
    def speak(self, text):
        """Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
        """
        if not text:
            logger.warning("Empty text provided for speech synthesis")
            return
            
        if not self.tts:
            logger.error("TTS not initialized")
            return
        
        try:
            logger.info(f"Converting text to speech: {text[:50]}...")
            
            # Submit TTS task to thread pool
            self.executor.submit(self._speak_in_thread, text)
            logger.info("Speech synthesis task submitted")
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
    
    def save_to_file(self, text, output_file):
        """Save synthesized speech to a file
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
        """
        if not self.tts:
            logger.error("TTS not initialized")
            return
            
        try:
            logger.info(f"Saving speech to file: {output_file}")
            self._speak_in_thread(text, output_path=output_file)
            logger.info("Speech saved to file successfully")
        except Exception as e:
            logger.error(f"Error saving speech to file: {e}")
            
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
            logger.info("TTS executor shutdown") 