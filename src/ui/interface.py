"""
ZYLIA - User Interface Module
Provides a graphical interface for interacting with the ZYLIA assistant
"""
import tkinter as tk
from tkinter import scrolledtext, ttk
import logging

logger = logging.getLogger("ZYLIA.UI")

class ZyliaUI:
    """Main UI class for the ZYLIA assistant"""
    
    def __init__(self, zylia_instance):
        """Initialize the UI components
        
        Args:
            zylia_instance: Reference to the main Zylia orchestrator instance
        """
        self.zylia = zylia_instance
        self.root = tk.Tk()
        self.setup_ui()
        
    def setup_ui(self):
        """Configure the main UI components"""
        # Configure main window
        self.root.title("ZYLIA Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Create header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        # ZYLIA title
        title_label = tk.Label(
            header_frame, 
            text="ZYLIA", 
            font=("Arial", 24, "bold"), 
            fg="white", 
            bg="#2c3e50"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Status indicator
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = tk.Label(
            header_frame,
            textvariable=self.status_var,
            font=("Arial", 12),
            fg="white",
            bg="#2c3e50"
        )
        status_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Create conversation area (main content)
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Conversation history display
        self.conversation_display = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            height=20
        )
        self.conversation_display.pack(fill=tk.BOTH, expand=True, pady=10)
        self.conversation_display.config(state=tk.DISABLED)  # Read-only
        
        # Bottom control area
        control_frame = tk.Frame(self.root, bg="#f0f0f0", height=100)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)
        
        # Record button
        self.talk_button = ttk.Button(
            control_frame,
            text="Talk to ZYLIA",
            command=self.zylia.process_voice_command,
            style="TButton"
        )
        self.talk_button.pack(side=tk.LEFT, padx=10)
        
        # Text input for typing instead of speaking
        self.text_input = ttk.Entry(control_frame, width=50)
        self.text_input.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.text_input.bind("<Return>", self.on_text_submit)
        
        # Send button
        self.send_button = ttk.Button(
            control_frame,
            text="Send",
            command=self.on_text_submit
        )
        self.send_button.pack(side=tk.RIGHT, padx=10)
        
        # Style configuration
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12))
        
        # Set keyboard focus to text input
        self.text_input.focus_set()
        
        # Add keyboard shortcuts
        self.root.bind('<F5>', lambda e: self.zylia.process_voice_command())
        
        # Add status bar at bottom
        status_bar = ttk.Label(
            self.root, 
            text="F5: Voice Input | Enter: Send Text", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        logger.info("UI setup complete")
    
    def on_text_submit(self, event=None):
        """Handle text submission from the entry field"""
        text = self.text_input.get().strip()
        if text:
            self.text_input.delete(0, tk.END)
            # Process the text through main Zylia handler
            self.zylia.process_text_input(text)
    
    def update_status(self, status):
        """Update the status indicator
        
        Args:
            status: String describing current status
        """
        self.status_var.set(status)
        self.root.update()
        logger.debug(f"Status updated: {status}")
    
    def display_user_message(self, message):
        """Display a user message in the conversation area
        
        Args:
            message: The text message from the user
        """
        self.conversation_display.config(state=tk.NORMAL)
        self.conversation_display.insert(tk.END, "You: ", "user_tag")
        self.conversation_display.insert(tk.END, f"{message}\n\n", "user_text")
        self.conversation_display.tag_configure("user_tag", foreground="#2c3e50", font=("Arial", 11, "bold"))
        self.conversation_display.tag_configure("user_text", foreground="#2c3e50")
        self.conversation_display.see(tk.END)
        self.conversation_display.config(state=tk.DISABLED)
    
    def display_ai_message(self, message):
        """Display an AI response in the conversation area
        
        Args:
            message: The text message from the AI
        """
        self.conversation_display.config(state=tk.NORMAL)
        self.conversation_display.insert(tk.END, "ZYLIA: ", "ai_tag")
        self.conversation_display.insert(tk.END, f"{message}\n\n", "ai_text")
        self.conversation_display.tag_configure("ai_tag", foreground="#27ae60", font=("Arial", 11, "bold"))
        self.conversation_display.tag_configure("ai_text", foreground="#333333")
        self.conversation_display.see(tk.END)
        self.conversation_display.config(state=tk.DISABLED)
    
    def run(self):
        """Start the UI main loop"""
        logger.info("Starting UI main loop")
        self.root.mainloop() 