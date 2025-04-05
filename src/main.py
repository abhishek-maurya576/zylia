#!/usr/bin/env python3
"""
ZYLIA - Personal Voice Assistant
Main orchestrator script that integrates all components
"""
import os
import sys
import logging
from pathlib import Path
import traceback
import datetime
import re
import time
import threading
import json
from dotenv import load_dotenv

# Add the project root to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ZYLIA")

# Import components
from src.ui.interface import ZyliaUI
from src.audio.recorder import AudioRecorder
from src.audio.speech import SpeechToText, TextToSpeech
try:
    from src.audio.neural_tts import NeuralTextToSpeech
    NEURAL_TTS_AVAILABLE = True
except ImportError:
    logger.warning("Neural TTS module not available, falling back to default TTS")
    NEURAL_TTS_AVAILABLE = False
    
from src.db.memory import MemoryManager

try:
    from src.db.vector_store import VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    logger.warning("Vector store module not available")
    VECTOR_STORE_AVAILABLE = False
    
from src.ai.gemini import GeminiAPI

try:
    from src.ai.local_llm import LocalLLM
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    logger.warning("Local LLM module not available")
    LOCAL_LLM_AVAILABLE = False

class Zylia:
    """Main Zylia assistant class that orchestrates all components"""
    
    def __init__(self):
        """Initialize Zylia and its components"""
        logger.info("Initializing ZYLIA assistant...")
        
        # Check for API key
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables!")
            self.api_key = "AIzaSyAh_yhZr8YEzqz2ucu63yoEFkQ4bPY35sQ"  # Using the key from blueprint
        
        self.components = {}  # Track initialized components
        self.fully_initialized = False
        
        try:
            # Initialize database components
            self.memory = MemoryManager()
            self.components['memory'] = True
            
            # Initialize vector store if available
            if VECTOR_STORE_AVAILABLE:
                try:
                    self.vector_store = VectorStore()
                    self.components['vector_store'] = True
                    logger.info("Vector store initialized")
                except Exception as e:
                    logger.warning(f"Vector store initialization failed: {e}")
                    self.vector_store = None
                    self.components['vector_store'] = False
            else:
                self.vector_store = None
                self.components['vector_store'] = False
            
            # Initialize API connections - Gemini API first
            self.gemini = GeminiAPI(api_key=self.api_key)
            self.components['gemini'] = True
            
            # Initialize Local LLM if available (for fallback)
            if LOCAL_LLM_AVAILABLE:
                try:
                    # Start LLM loading in a separate thread to avoid blocking startup
                    self.local_llm = LocalLLM(
                        model_id="TheBloke/Llama-2-7B-Chat-GGUF",
                        model_file="llama-2-7b-chat.Q4_K_M.gguf"
                    )
                    self.components['local_llm'] = True
                    
                    # Start model loading in background
                    threading.Thread(
                        target=self._load_llm_in_background,
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    logger.warning(f"Local LLM initialization failed: {e}")
                    self.local_llm = None
                    self.components['local_llm'] = False
            else:
                self.local_llm = None
                self.components['local_llm'] = False
            
            # Initialize audio components
            self.recorder = AudioRecorder()
            self.components['recorder'] = True
            
            self.stt = SpeechToText(model_size="base")  # Use base model for better accuracy
            self.components['stt'] = True
            
            # Try to initialize Neural TTS first if available
            if NEURAL_TTS_AVAILABLE:
                try:
                    logger.info("Initializing Neural TTS...")
                    self.tts = NeuralTextToSpeech(
                        whisper_effect=False,  # Normal voice, no whisper effect
                        voice_speed=1.0        # Normal speaking pace
                    )
                    self.components['tts'] = 'neural'
                    logger.info("Neural TTS initialized successfully")
                except Exception as e:
                    logger.warning(f"Neural TTS initialization failed: {e}, falling back to default TTS")
                    self.tts = TextToSpeech()  # Fallback to standard TTS
                    self.components['tts'] = 'standard'
            else:
                self.tts = TextToSpeech()  # Standard TTS with normal voice
                self.components['tts'] = 'standard'
            
            # Initialize UI last (as it may reference other components)
            self.ui = ZyliaUI(self)
            self.components['ui'] = True
            
            # State tracking
            self.offline_mode = False
            self.conversation_history = []
            
            # Set flag to indicate full initialization
            self.fully_initialized = True
            logger.info("ZYLIA assistant initialized successfully!")
            logger.info(f"Components status: {json.dumps(self.components, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            logger.error(traceback.format_exc())
            self.fully_initialized = False
            # We'll still create the UI to display the error
            try:
                self.ui = ZyliaUI(self)
                self.components['ui'] = True
            except:
                logger.critical("Failed to initialize UI")
                self.components['ui'] = False
    
    def _load_llm_in_background(self):
        """Load the local LLM in a background thread"""
        if self.local_llm:
            logger.info("Loading local LLM in background...")
            try:
                self.local_llm.load_model()
                logger.info("Local LLM loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local LLM: {e}")
    
    def process_voice_command(self):
        """Main function to process a voice command from the user"""
        if not self.fully_initialized:
            error_msg = "ZYLIA is not fully initialized. Please check the logs for errors."
            self.ui.display_ai_message(error_msg)
            return
            
        try:
            # Update UI status
            self.ui.update_status("Listening...")
            
            # Record audio with improved settings
            audio_data = self.recorder.record_audio(
                duration=10.0,         # Longer max duration 
                silence_threshold=0.005, # More sensitive threshold
                silence_duration=1.0,   # Standard silence duration
                min_duration=0.5        # Shorter minimum duration
            )
            
            # Check if audio recording was successful
            if not audio_data:
                self.ui.update_status("No voice detected")
                self.ui.display_ai_message("I didn't hear anything. Please try speaking louder or closer to the microphone, or type your message below.")
                self.ui.update_status("Ready")
                return
            
            # Transcribe audio
            self.ui.update_status("Transcribing...")
            user_text = self.stt.transcribe(audio_data)
            
            # Check if transcription was successful by looking for common error messages
            error_responses = [
                "I didn't catch that",
                "I couldn't hear anything",
                "Sorry, I had trouble understanding that"
            ]
            
            if any(error in user_text for error in error_responses):
                self.ui.update_status("Transcription failed")
                self.ui.display_ai_message(user_text)
                self.ui.update_status("Ready")
                return
            
            # Check for special case date/data patterns
            if re.search(r"(what|tell me|when).*(today|current)?.*(date|data)", user_text.lower()):
                if "data" in user_text.lower():
                    # Let's clarify what the user meant
                    self.ui.display_user_message(user_text)
                    clarification = f"I think you're asking for today's date. Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}. Is that what you wanted to know?"
                    self.ui.display_ai_message(clarification)
                    self.tts.speak(clarification)
                    self.ui.update_status("Ready")
                    return
            
            # Process the text input
            self.process_text_input(user_text)
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            logger.error(traceback.format_exc())
            self.ui.update_status("Error occurred")
            self.ui.display_ai_message("Sorry, I encountered an error. Please try typing your message instead.")
    
    def process_text_input(self, text):
        """Process text input from either voice or text entry"""
        try:
            # Skip empty or very short input
            if not text or len(text.strip()) < 2:
                self.ui.update_status("Empty input")
                return
                
            # Check for local commands that don't need API
            text_lower = text.lower().strip()
            
            # Handle special system commands
            if text_lower in ["go offline", "switch to offline", "use local model"]:
                self.offline_mode = True
                self.ui.display_user_message(text)
                self.ui.display_ai_message("I've switched to offline mode. I'll use my local processing capabilities now.")
                self.tts.speak("I've switched to offline mode.")
                self.ui.update_status("Ready (Offline Mode)")
                return
                
            if text_lower in ["go online", "switch to online", "use cloud model"]:
                self.offline_mode = False
                self.ui.display_user_message(text)
                self.ui.display_ai_message("I'm back in online mode. I'll use the Gemini API for processing.")
                self.tts.speak("I'm back in online mode.")
                self.ui.update_status("Ready")
                return
            
            # Handle date queries directly
            if re.search(r"(what|tell me|when).*(today).*(date|day)", text_lower) or \
               text_lower in ["what's the date", "what is the date", "date", "what day is it"]:
                
                self.ui.display_user_message(text)
                date_response = f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."
                self.ui.display_ai_message(date_response)
                self.tts.speak(date_response)
                
                # Store in both memory systems
                self.memory.save_interaction(text, date_response)
                if self.vector_store:
                    self.vector_store.add_conversation(text, date_response)
                    
                self.ui.update_status("Ready")
                return
            
            # Handle time queries directly
            if re.search(r"(what|tell me).*(time)", text_lower) or \
               text_lower in ["what's the time", "what time is it", "time"]:
                
                self.ui.display_user_message(text)
                time_response = f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
                self.ui.display_ai_message(time_response)
                self.tts.speak(time_response)
                
                # Store in both memory systems
                self.memory.save_interaction(text, time_response)
                if self.vector_store:
                    self.vector_store.add_conversation(text, time_response)
                    
                self.ui.update_status("Ready")
                return
            
            # Update UI with user text
            self.ui.display_user_message(text)
            
            # Track the conversation for context
            self.conversation_history.append({"role": "user", "content": text})
            
            # Update status
            self.ui.update_status("Thinking..." + (" (Offline Mode)" if self.offline_mode else ""))
            
            # Get response based on mode
            ai_response = self._get_ai_response(text)
            
            # Update conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Trim conversation history if it gets too long (keep last 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Update UI with AI response
            self.ui.display_ai_message(ai_response)
            
            # Store interaction in both memory systems
            self.memory.save_interaction(text, ai_response)
            if self.vector_store:
                self.vector_store.add_conversation(text, ai_response)
            
            # Speak the response
            self.ui.update_status("Speaking...")
            self.tts.speak(ai_response)
            
            # Set status back to ready
            self.ui.update_status("Ready" + (" (Offline Mode)" if self.offline_mode else ""))
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            logger.error(traceback.format_exc())
            self.ui.update_status("Error occurred")
            self.ui.display_ai_message("Sorry, I encountered an error processing your request. Please try again with a different question.")
    
    def _get_ai_response(self, text):
        """Get AI response from the appropriate model based on current mode
        
        Args:
            text: User input text
            
        Returns:
            AI response text
        """
        try:
            # Get relevant memories if available
            memories = ""
            if self.vector_store:
                memories = self.vector_store.get_relevant_memories(text)
                if memories:
                    logger.info(f"Found relevant memories: {memories[:100]}...")
            
            # Get context from memory manager
            context = self.memory.get_recent_history(5)
            
            # Add memories to the context if available
            if memories:
                context = context + "\n\n" + memories
            
            # Use local LLM in offline mode if available
            if self.offline_mode and self.local_llm:
                logger.info("Using local LLM for response generation")
                return self.local_llm.get_response(text, self.conversation_history)
            
            # Otherwise use Gemini API
            logger.info("Using Gemini API for response generation")
            try:
                return self.gemini.generate_response(text, context)
            except Exception as api_error:
                logger.error(f"Error from Gemini API: {api_error}")
                
                # Try local LLM as fallback if available
                if self.local_llm:
                    logger.info("Falling back to local LLM due to API error")
                    return self.local_llm.get_response(text, self.conversation_history)
                
                # Try a simpler fallback prompt if the API fails and no local LLM
                try:
                    fallback_prompt = f"Answer the following question briefly: {text}"
                    return self.gemini.generate_response(fallback_prompt, None)
                except:
                    # If even the fallback fails, use a hardcoded response
                    return "I'm sorry, I'm having trouble connecting to my intelligence service right now. Please try again in a moment."
                    
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I encountered an error while processing your request. Please try again."
    
    def run(self):
        """Start the ZYLIA assistant and run the UI main loop"""
        # Welcome message with a warm, personal tone but normal voice
        welcome_msg = "Hello, I'm ZYLIA. I'm here for you whenever you need someone to talk to. How are you feeling today?"
        self.ui.display_ai_message(welcome_msg)
        
        if self.fully_initialized:
            # Speak welcome message if everything is initialized
            self.tts.speak(welcome_msg)
            
            # Add system features info in a more personal way
            features = []
            if self.components.get('tts') == 'neural':
                features.append("a more natural voice")
            if self.components.get('vector_store'):
                features.append("the ability to remember our conversations")
            if self.components.get('local_llm'):
                features.append("the ability to be here for you even without internet")
                
            if features:
                if len(features) == 1:
                    feature_msg = f"I've been updated with {features[0]} so I can support you better."
                elif len(features) == 2:
                    feature_msg = f"I've been updated with {features[0]} and {features[1]} so I can be here for you in more ways."
                else:
                    feature_list = ", ".join(features[:-1]) + f", and {features[-1]}"
                    feature_msg = f"I've been updated with {feature_list} so I can support you better in every way."
                    
                self.ui.display_ai_message(feature_msg)
                self.tts.speak(feature_msg)
        else:
            # Display error message with normal voice
            error_msg = "I'm sorry, but I'm having some trouble getting fully set up. Some of my features might not work correctly right now, but I'll still do my best to be here for you."
            self.ui.display_ai_message(error_msg)
        
        # Start the UI main loop
        self.ui.run()
        
    def cleanup(self):
        """Clean up resources before exit"""
        logger.info("Cleaning up resources...")
        
        # Close vector store if available
        if self.vector_store:
            try:
                self.vector_store.close()
            except:
                pass
                
        # Unload local LLM if available
        if self.local_llm:
            try:
                self.local_llm.unload_model()
            except:
                pass
                
        # Clean up TTS resources
        if hasattr(self.tts, 'cleanup'):
            try:
                self.tts.cleanup()
            except:
                pass
                
        logger.info("Cleanup complete")

if __name__ == "__main__":
    assistant = Zylia()
    try:
        assistant.run()
    finally:
        # Ensure cleanup happens even if there's an error
        assistant.cleanup() 