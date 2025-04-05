"""
ZYLIA - Memory Manager Module
Handles persistent storage and retrieval of conversation history and user preferences
"""
import sqlite3
import logging
import os
from pathlib import Path
import json
import datetime

logger = logging.getLogger("ZYLIA.DB.Memory")

class MemoryManager:
    """Manages persistent storage for ZYLIA using SQLite"""
    
    def __init__(self, db_path=None):
        """Initialize the memory manager
        
        Args:
            db_path: Path to the SQLite database file (default: zylia.db in the user's data directory)
        """
        # Set up the database path
        if db_path is None:
            # Use a platform-specific data directory
            if os.name == 'nt':  # Windows
                data_dir = Path(os.environ.get('APPDATA', '.')) / 'ZYLIA'
            else:  # macOS/Linux
                data_dir = Path(os.environ.get('HOME', '.')) / '.zylia'
            
            # Create the directory if it doesn't exist
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Set the database path
            self.db_path = data_dir / 'zylia.db'
        else:
            self.db_path = Path(db_path)
        
        logger.info(f"Using database at: {self.db_path}")
        
        # Initialize the database
        self.initialize_db()
    
    def initialize_db(self):
        """Create the database tables if they don't exist"""
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create the preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # Create the conversation history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            ''')
            
            # Create the schedule table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_date DATE NOT NULL,
                    event_time TIME,
                    description TEXT NOT NULL,
                    is_completed INTEGER DEFAULT 0
                )
            ''')
            
            # Create the notes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    content TEXT NOT NULL,
                    tags TEXT
                )
            ''')
            
            # Save changes and close connection
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_interaction(self, user_text, ai_text):
        """Save a conversation interaction to the database
        
        Args:
            user_text: The user's message text
            ai_text: The AI's response text
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Save user message
            cursor.execute(
                "INSERT INTO conversation_history (role, content) VALUES (?, ?)",
                ("user", user_text)
            )
            
            # Save AI response
            cursor.execute(
                "INSERT INTO conversation_history (role, content) VALUES (?, ?)",
                ("assistant", ai_text)
            )
            
            conn.commit()
            conn.close()
            
            logger.debug("Conversation turn saved to database")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_recent_history(self, num_turns=5):
        """Get the most recent conversation history
        
        Args:
            num_turns: Number of conversation turns to retrieve (default: 5)
            
        Returns:
            List of dictionaries with 'user' and 'assistant' keys
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get the latest interactions, ordered by ID (most recent last)
            cursor.execute(
                "SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?",
                (num_turns * 2,)  # Multiply by 2 because each turn has a user and assistant message
            )
            
            # Fetch all rows
            rows = cursor.fetchall()
            conn.close()
            
            # Reverse the results to get chronological order (oldest first)
            rows.reverse()
            
            # Group into conversation turns
            turns = []
            current_turn = {}
            
            for role, content in rows:
                if role == "user":
                    if current_turn and "user" in current_turn:
                        # New turn
                        turns.append(current_turn)
                        current_turn = {"user": content}
                    else:
                        current_turn["user"] = content
                elif role == "assistant":
                    if "user" in current_turn:
                        current_turn["assistant"] = content
                        turns.append(current_turn)
                        current_turn = {}
            
            # Add the last turn if it has both user and assistant
            if current_turn and "user" in current_turn and "assistant" in current_turn:
                turns.append(current_turn)
            
            # Return only the requested number of complete turns
            return turns[-num_turns:] if turns else []
        
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def save_preference(self, key, value):
        """Save a user preference to the database
        
        Args:
            key: Preference key/name
            value: Preference value (can be string, number, boolean, or object)
        """
        try:
            # Convert non-string values to JSON
            if not isinstance(value, str):
                value = json.dumps(value)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert or update the preference
            cursor.execute(
                "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
                (key, value)
            )
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Preference saved: {key}")
        except Exception as e:
            logger.error(f"Error saving preference: {e}")
    
    def get_preference(self, key, default=None):
        """Get a user preference from the database
        
        Args:
            key: Preference key/name
            default: Default value to return if the preference doesn't exist
            
        Returns:
            The preference value, or the default if not found
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get the preference
            cursor.execute(
                "SELECT value FROM preferences WHERE key = ?",
                (key,)
            )
            
            # Fetch the result
            row = cursor.fetchone()
            conn.close()
            
            if row:
                value = row[0]
                
                # Try to parse as JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Return as string if not valid JSON
                    return value
                    
            return default
            
        except Exception as e:
            logger.error(f"Error retrieving preference: {e}")
            return default
    
    def add_schedule_item(self, date, description, time=None):
        """Add a schedule item to the database
        
        Args:
            date: Date string (YYYY-MM-DD)
            description: Description of the event
            time: Optional time string (HH:MM)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert the schedule item
            cursor.execute(
                "INSERT INTO schedule (event_date, event_time, description) VALUES (?, ?, ?)",
                (date, time, description)
            )
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Schedule item added: {description} on {date}")
        except Exception as e:
            logger.error(f"Error adding schedule item: {e}")
    
    def get_schedule_for_date(self, date):
        """Get scheduled items for a specific date
        
        Args:
            date: Date string (YYYY-MM-DD) or datetime.date object
            
        Returns:
            List of dictionaries with schedule item details
        """
        try:
            # Convert date object to string if needed
            if isinstance(date, datetime.date):
                date = date.strftime("%Y-%m-%d")
                
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Configure SQLite to return dictionaries
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get schedule items for the specified date
            cursor.execute(
                "SELECT id, event_time, description, is_completed FROM schedule WHERE event_date = ? ORDER BY event_time",
                (date,)
            )
            
            # Fetch all rows
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            schedule_items = []
            for row in rows:
                schedule_items.append({
                    "id": row["id"],
                    "time": row["event_time"],
                    "description": row["description"],
                    "is_completed": bool(row["is_completed"])
                })
            
            return schedule_items
            
        except Exception as e:
            logger.error(f"Error retrieving schedule: {e}")
            return [] 