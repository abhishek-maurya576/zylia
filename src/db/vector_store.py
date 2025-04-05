"""
ZYLIA - Vector Database Module
Provides long-term memory and knowledge storage using vector embeddings
"""

import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("ZYLIA.DB.VectorStore")

class VectorStore:
    """Vector database for semantic search and knowledge management"""
    
    def __init__(self, db_directory="data/vector_db", 
                 embedding_model="all-MiniLM-L6-v2",
                 collection_name="zylia_memory"):
        """Initialize the vector database
        
        Args:
            db_directory: Directory to store the vector database
            embedding_model: Sentence transformers model for embeddings
            collection_name: Name of the default collection
        """
        self.db_directory = Path(db_directory)
        self.db_directory.mkdir(exist_ok=True, parents=True)
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Database client
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        # Initialize the database
        self._init_db()
        
        logger.info("Vector store initialized")
    
    def _init_db(self):
        """Initialize the ChromaDB client and collection"""
        try:
            # Initialize ChromaDB with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.db_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception as e:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(name=self.collection_name)
            
            # Load embedding model
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                # Fallback to simpler model if available
                try:
                    self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
                    logger.info("Loaded fallback embedding model")
                except:
                    logger.error("Failed to load fallback embedding model")
        
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            self.client = None
            self.collection = None
    
    def _generate_embedding(self, text):
        """Generate embeddings for text using the sentence transformer model
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            logger.error("Embedding model not available")
            return None
            
        try:
            return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def add_memory(self, text, metadata=None, timestamp=None, category="general"):
        """Add a memory entry to the vector store
        
        Args:
            text: Text content to store
            metadata: Additional metadata (dict)
            timestamp: Timestamp (default: current time)
            category: Memory category (e.g., 'conversation', 'fact', 'preference')
            
        Returns:
            ID of the added memory
        """
        if self.collection is None:
            logger.error("Vector collection not available")
            return None
            
        try:
            # Generate embedding if using external model
            embeddings = None
            if self.embedding_model is not None:
                embeddings = self._generate_embedding(text)
            
            # Generate unique ID based on timestamp
            timestamp = timestamp or datetime.now().isoformat()
            memory_id = f"mem_{int(time.time() * 1000)}_{hash(text) % 10000:04d}"
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            metadata.update({
                "timestamp": timestamp,
                "category": category
            })
            
            # Add to collection
            if embeddings:
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embeddings],
                    metadatas=[metadata],
                    documents=[text]
                )
            else:
                # Let ChromaDB handle embeddings
                self.collection.add(
                    ids=[memory_id],
                    metadatas=[metadata],
                    documents=[text]
                )
                
            logger.info(f"Added memory: {memory_id} - {text[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None
    
    def search_memories(self, query, n_results=5, category=None, filter_metadata=None):
        """Search for memories similar to the query
        
        Args:
            query: Search query
            n_results: Number of results to return
            category: Filter by category
            filter_metadata: Additional metadata filters
            
        Returns:
            List of matching memories with similarity scores
        """
        if self.collection is None:
            logger.error("Vector collection not available")
            return []
            
        try:
            # Generate query embedding if using external model
            query_embedding = None
            if self.embedding_model is not None:
                query_embedding = self._generate_embedding(query)
            
            # Prepare filters
            where_clause = {}
            if category:
                where_clause["category"] = category
                
            if filter_metadata:
                where_clause.update(filter_metadata)
                
            # Execute search
            if query_embedding:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause if where_clause else None
                )
            else:
                # Let ChromaDB handle embeddings
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None
                )
            
            # Format results
            memories = []
            if results and 'ids' in results and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    memory = {
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                    }
                    if 'distances' in results:
                        memory['similarity'] = 1.0 - results['distances'][0][i]
                    memories.append(memory)
            
            logger.info(f"Found {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def get_memory_by_id(self, memory_id):
        """Retrieve a specific memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory dict or None if not found
        """
        if self.collection is None:
            logger.error("Vector collection not available")
            return None
            
        try:
            result = self.collection.get(ids=[memory_id])
            
            if result and 'documents' in result and len(result['documents']) > 0:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0] if 'metadatas' in result else {}
                }
            
            logger.warning(f"Memory not found: {memory_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return None
    
    def delete_memory(self, memory_id):
        """Delete a memory from the store
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            logger.error("Vector collection not available")
            return False
            
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
    
    def add_conversation(self, user_message, assistant_response):
        """Add a conversation exchange to the vector store
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            
        Returns:
            ID of the added memory
        """
        # Format the conversation for storage
        conversation_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Add metadata
        metadata = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "type": "conversation"
        }
        
        return self.add_memory(
            text=conversation_text,
            metadata=metadata,
            category="conversation"
        )
    
    def add_fact(self, fact, source=None):
        """Add a factual piece of information
        
        Args:
            fact: The factual information
            source: Source of the information (optional)
            
        Returns:
            ID of the added memory
        """
        metadata = {"type": "fact"}
        if source:
            metadata["source"] = source
            
        return self.add_memory(
            text=fact,
            metadata=metadata,
            category="fact"
        )
    
    def add_preference(self, preference_key, preference_value, user_stated=True):
        """Add a user preference
        
        Args:
            preference_key: Type of preference (e.g., "favorite_color")
            preference_value: Value of the preference
            user_stated: Whether this was explicitly stated by the user
            
        Returns:
            ID of the added memory
        """
        text = f"User preference: {preference_key} = {preference_value}"
        
        metadata = {
            "preference_key": preference_key,
            "preference_value": preference_value,
            "user_stated": user_stated,
            "type": "preference"
        }
        
        return self.add_memory(
            text=text,
            metadata=metadata,
            category="preference"
        )
    
    def get_relevant_memories(self, query, n_results=5):
        """Get memories relevant to the current query
        
        Args:
            query: Current user query
            n_results: Number of memories to retrieve
            
        Returns:
            Formatted string with relevant memories
        """
        memories = self.search_memories(query, n_results=n_results)
        
        if not memories:
            return ""
            
        context_parts = ["Relevant information from your memory:"]
        
        for memory in memories:
            timestamp = memory.get('metadata', {}).get('timestamp', 'Unknown time')
            if isinstance(timestamp, str) and len(timestamp) > 10:
                # Format ISO timestamp to just show the date
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d")
                except:
                    pass
                    
            memory_type = memory.get('metadata', {}).get('type', 'information')
            similarity = memory.get('similarity', 0.0)
            
            # Only include highly relevant memories (similarity > 0.7)
            if similarity > 0.7:
                context_parts.append(f"- [{memory_type} from {timestamp}]: {memory['text']}")
        
        return "\n".join(context_parts) if len(context_parts) > 1 else ""
        
    def close(self):
        """Close the database connection"""
        self.client = None
        self.collection = None
        
        # Unload the embedding model
        if self.embedding_model:
            self.embedding_model = None
            import gc
            gc.collect()
            
        logger.info("Vector store closed") 