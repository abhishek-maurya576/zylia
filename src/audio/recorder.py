"""
ZYLIA - Audio Recorder Module
Handles audio recording from the microphone
"""
import numpy as np
import sounddevice as sd
import logging
from scipy.io import wavfile
import tempfile
import os
import time
from pathlib import Path

logger = logging.getLogger("ZYLIA.Audio.Recorder")

class AudioRecorder:
    """Records audio from the microphone"""
    
    def __init__(self, sample_rate=16000, channels=1):
        """Initialize the audio recorder
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.temp_dir = Path(tempfile.gettempdir()) / "zylia"
        os.makedirs(self.temp_dir, exist_ok=True)
        self._check_audio_devices()
        logger.info(f"AudioRecorder initialized with sample rate {sample_rate}Hz, {channels} channels")
    
    def _check_audio_devices(self):
        """Check available audio devices and log them"""
        try:
            devices = sd.query_devices()
            logger.info(f"Found {len(devices)} audio devices")
            
            # Find default input device
            default_input = sd.query_devices(kind='input')
            logger.info(f"Default input device: {default_input['name']}")
            
            # List all input devices for debugging
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.debug(f"Input device {i}: {device['name']}")
        except Exception as e:
            logger.error(f"Error checking audio devices: {e}")
    
    def record_audio(self, duration=8.0, silence_threshold=0.005, silence_duration=1.0, min_duration=0.5):
        """Record audio from the default microphone
        
        Args:
            duration: Maximum recording duration in seconds (default: 8.0)
            silence_threshold: Amplitude threshold to detect silence (default: 0.005) - lower value is more sensitive
            silence_duration: Duration of silence to stop recording (default: 1.0)
            min_duration: Minimum recording duration in seconds (default: 0.5)
            
        Returns:
            Path to the recorded audio file
        """
        logger.info(f"Recording audio for up to {duration} seconds (min: {min_duration}s)")
        print("\n" + "="*50)
        print("ZYLIA VOICE RECORDING")
        print("="*50)
        print("Please speak clearly into your microphone.")
        print("Recording will stop after silence is detected.")
        print("="*50 + "\n")
        
        # Calculate total number of samples for the maximum duration
        max_samples = int(duration * self.sample_rate)
        min_samples = int(min_duration * self.sample_rate)
        
        # Buffer to store the recorded audio
        audio_buffer = np.zeros((max_samples, self.channels), dtype=np.float32)
        
        # Variables for silence detection
        silence_samples = int(silence_duration * self.sample_rate)
        silent_samples_count = 0
        recording_started = False
        voice_detected = False
        
        # Callback function for the audio stream
        def callback(indata, frames, time, status):
            nonlocal position, silent_samples_count, recording_started, voice_detected
            
            if status:
                logger.warning(f"Audio recording status: {status}")
            
            # Store the recorded audio in the buffer
            if position + frames <= max_samples:
                audio_buffer[position:position + frames] = indata
                position += frames
                
                # Calculate volume level (RMS)
                volume_level = np.sqrt(np.mean(indata**2))
                
                # Check if voice is detected - more sensitive threshold
                if volume_level > silence_threshold:
                    if not recording_started:
                        recording_started = True
                        logger.debug(f"Voice detected (level: {volume_level:.4f})")
                    voice_detected = True
                    silent_samples_count = 0
                else:
                    # Only count silence after voice has been detected
                    if voice_detected:
                        silent_samples_count += frames
                
                # Stop recording if silence duration is reached after minimum duration
                if voice_detected and silent_samples_count >= silence_samples and position >= min_samples:
                    logger.debug(f"Stopped recording due to silence, position: {position}, min: {min_samples}")
                    raise sd.CallbackStop()
        
        # Start recording
        position = 0
        try:
            # Print available devices before recording
            logger.info("Starting audio recording...")
            
            # Record with slightly higher volume and automatic gain
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels, 
                callback=callback,
                blocksize=1024,  # Smaller blocksize for more frequent callbacks
                device=None,     # Use default device
                dtype='float32'
            ):
                # Show audio meter while recording
                print("Recording... (Speak now)")
                meter_update_time = 0.05  # Update meter more frequently (50ms)
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    # Wait a bit
                    time.sleep(meter_update_time)
                    
                    # If we have data, show a simple meter
                    if position > 0:
                        current_audio = audio_buffer[:position]
                        if len(current_audio) > 0:
                            # Get the last 100ms of audio for the meter
                            recent_audio = current_audio[-int(0.1*self.sample_rate):]
                            volume = np.sqrt(np.mean(recent_audio**2))
                            
                            # Make the meter more visible
                            meter_chars = int(volume * 200)  # Scale up for visibility
                            meter = "█" * min(meter_chars, 40)  # Limit to 40 chars max
                            
                            # Color coding based on volume
                            if volume > silence_threshold:
                                status = "\033[92mVOICE DETECTED\033[0m"  # Green text
                            else:
                                status = "\033[91mSilence\033[0m"  # Red text
                                
                            print(f"\rLevel: {meter:<40} {status}", end="")
                
                print("\nMaximum recording time reached")
                        
        except sd.CallbackStop:
            logger.info("Recording stopped due to silence detection")
        except KeyboardInterrupt:
            logger.info("Recording stopped by user")
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
        
        # Trim the buffer to the actual recorded length
        recorded_audio = audio_buffer[:position]
        
        # Check if any voice was detected
        if not voice_detected or position < (self.sample_rate * 0.3):  # Less than 0.3 seconds (even more lenient)
            logger.warning("No voice detected or recording too short")
            print("\nNo voice detected or recording too short. Please try again.")
            return None
            
        # Apply some gain to increase volume
        gain = 1.5  # Increase volume by 50%
        recorded_audio = np.clip(recorded_audio * gain, -1.0, 1.0)
            
        # Save to a temporary WAV file
        temp_file = self.temp_dir / f"recording_{np.random.randint(0, 10000)}.wav"
        try:
            # Convert float32 to int16
            int16_data = (recorded_audio * 32767).astype(np.int16)
            wavfile.write(temp_file, self.sample_rate, int16_data)
            logger.info(f"Audio saved to {temp_file} ({position/self.sample_rate:.2f} seconds)")
            print(f"\nRecording complete! Duration: {position/self.sample_rate:.2f} seconds")
            
            return str(temp_file)
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None 