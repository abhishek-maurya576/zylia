"""
ZYLIA - Local LLM Module
Provides local language model capabilities for offline operation
"""

import os
import logging
import json
import time
from pathlib import Path
from threading import Lock
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

logger = logging.getLogger("ZYLIA.AI.LocalLLM")

class LocalLLM:
    """Provides local language model inference using llama-cpp"""
    
    def __init__(self, model_path=None, model_id="TheBloke/Llama-2-7B-Chat-GGUF", 
                 model_file="llama-2-7b-chat.Q4_K_M.gguf", n_ctx=4096, 
                 n_gpu_layers=0, temperature=0.7, top_p=0.95, max_tokens=1024,
                 system_prompt="You are ZYLIA, a helpful and friendly personal assistant."):
        """Initialize the local LLM
        
        Args:
            model_path: Path to local model file
            model_id: Hugging Face model ID to download if model_path is None
            model_file: Specific file to download from the model_id
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt to use for initialization
        """
        self.model_path = model_path
        self.model_id = model_id
        self.model_file = model_file
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        # Setup directories
        self.models_dir = Path("models/llm")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Thread lock for model loading
        self.load_lock = Lock()
        
        # Model instance
        self.llm = None
        
        logger.info("LocalLLM module initialized")
    
    def _ensure_model_available(self):
        """Ensure the model file is available, downloading if necessary
        
        Returns:
            Path to the model file
        """
        if self.model_path and os.path.exists(self.model_path):
            # Use the provided model path
            return self.model_path
        
        # Try to find the model file in the models directory
        local_model_path = self.models_dir / self.model_file
        if local_model_path.exists():
            logger.info(f"Using existing model at {local_model_path}")
            return str(local_model_path)
        
        # Download the model
        logger.info(f"Downloading model {self.model_id}/{self.model_file}")
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.model_id,
                filename=self.model_file,
                cache_dir=str(self.models_dir)
            )
            logger.info(f"Model downloaded to {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            
            # Try to find any compatible model file
            logger.info("Searching for alternative model files")
            for file in self.models_dir.glob("*.gguf"):
                logger.info(f"Found alternative model: {file}")
                return str(file)
            
            raise ValueError("No suitable model file found")
    
    def load_model(self):
        """Load the language model
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        with self.load_lock:
            if self.llm is not None:
                return True
                
            try:
                # Ensure model file is available
                model_path = self._ensure_model_available()
                
                # Determine device configuration
                logger.info(f"Loading model from {model_path}")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                logger.info("Model loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.llm = None
                return False
    
    def get_response(self, prompt, conversation_history=None, streaming=False):
        """Get a response from the local LLM
        
        Args:
            prompt: User prompt
            conversation_history: List of previous conversation messages
            streaming: Whether to stream the response
            
        Returns:
            Generated response text
        """
        if not self.load_model():
            return "I'm unable to access my local knowledge. Please check the logs."
        
        try:
            # Build the messages array
            messages = []
            
            # Add system prompt
            messages.append({"role": "system", "content": self.system_prompt})
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append(msg)
            
            # Add the current user prompt
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"Generating response for: {prompt[:50]}...")
            start_time = time.time()
            
            # Generate response
            if streaming:
                # Return a generator for streaming
                return self._stream_response(messages)
            else:
                # Generate complete response
                response = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                
                # Extract the assistant's message
                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']
                    
                    # Log response time
                    elapsed = time.time() - start_time
                    tokens = response.get('usage', {}).get('completion_tokens', 0)
                    logger.info(f"Response generated in {elapsed:.2f}s ({tokens} tokens)")
                    
                    return content
                else:
                    logger.warning(f"Unexpected response format: {response}")
                    return "I encountered an issue processing your request."
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while trying to respond. Please try again."
    
    def _stream_response(self, messages):
        """Stream response from the model
        
        Args:
            messages: Conversation messages
            
        Yields:
            Text chunks as they are generated
        """
        try:
            response_stream = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in response_stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        yield delta['content']
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield "\nI encountered an error while responding. Please try again."
    
    def unload_model(self):
        """Unload the model to free up memory"""
        with self.load_lock:
            if self.llm is not None:
                # There's no explicit unload in llama-cpp-python, 
                # so we just remove the reference
                self.llm = None
                import gc
                gc.collect()
                logger.info("Model unloaded") 