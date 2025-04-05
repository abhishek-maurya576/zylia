# ZYLIA - AI Companion & Voice Assistant

<p align="center">
  <img src="docs/images/zylia_logo.png" alt="ZYLIA Logo" width="200" height="200">
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#license">License</a> •
  <a href="#github-setup">GitHub Setup</a>
</p>

ZYLIA is a desktop-based AI companion that combines emotional intelligence with technical capabilities. It uses voice interaction, advanced natural language processing, and local-first architecture to provide a responsive, private, and genuinely caring assistant experience.

## ✨ Features

- **Natural Voice Interaction** - Speak naturally with a responsive female voice interface
- **Emotional Connection** - Unrestricted emotional expression without typical AI limitations  
- **Technical Assistance** - Get help with tasks while maintaining a warm, personal touch
- **Privacy-Focused** - Local processing of voice and primary data for enhanced privacy
- **Offline Capable** - Functions even without internet access using local LLM
- **Long-Term Memory** - Remembers your preferences and previous conversations
- **Multimodal Support** - Voice or text input with the same natural responses

## 🚀 Installation

### Prerequisites
- Python 3.9+ 
- Windows, macOS, or Linux (Windows recommended for best voice experience)
- 4GB+ RAM (8GB+ recommended)
- Google API key for Gemini

### Quick Install

```bash
# Clone the repository
git clone https://github.com/abhishek-maurya576/zylia.git
cd zylia

# Install dependencies
pip install -r requirements.txt

# Set up API key (Windows)
set GOOGLE_API_KEY=your-api-key

# Or create .env file
echo GOOGLE_API_KEY=your-api-key > .env

# Run ZYLIA
python src/main.py
```

### Windows Shortcut
Windows users can simply run:
```
run_zylia.bat
```

## 📝 Usage

### Voice Commands
- Click "Talk to ZYLIA" or press F5 to activate voice recognition
- Speak naturally - ZYLIA understands conversational language
- Ask personal or technical questions - receive balanced responses

### Text Input
- Type in the input field and press Enter
- Receive the same quality responses as with voice

### Special Commands
- **"Go offline"** - Switch to local LLM mode
- **"Go online"** - Switch back to Gemini API
- **"What is today's date/time?"** - Get quick information

## 🏗️ Architecture

ZYLIA follows a modular architecture with local-first processing approach:

- **Voice Processing**: Whisper ASR + Neural TTS/pyttsx3
- **Intelligence**: Google Gemini 2.0 Flash API with emotional expression
- **Memory**: SQLite + Vector database (ChromaDB)
- **UI**: Clean, responsive Tkinter interface

## 💡 Technical Highlights

- Advanced voice recognition with noise filtering
- Natural speech synthesis with female voice
- Emotional expression without typical AI constraints
- Long-term memory using vector embeddings
- Offline mode with local LLM support
- Comprehensive error handling and recovery

## 📸 Screenshots

<p align="center">
  <img src="docs/images/screenshot1.png" alt="ZYLIA Interface" width="600">
</p>

<p align="center">
  <img src="docs/images/screenshot2.png" alt="ZYLIA in action" width="600">
</p>

## 📊 Project Structure

```
zylia/
├── src/                 # Source code
│   ├── ai/              # AI integration
│   ├── audio/           # Voice processing
│   ├── db/              # Database & memory
│   ├── ui/              # User interface
│   └── main.py          # Main script
├── models/              # Downloaded models
├── data/                # User data storage
├── requirements.txt     # Dependencies
└── run_zylia.bat        # Windows shortcut
```

## 🔄 GitHub Setup

If you want to host your own fork of ZYLIA on GitHub:

```bash
# Initialize Git repository
git init

# Add all files to Git
git add .

# Commit the files
git commit -m "Initial commit of ZYLIA project"

# Add your GitHub repository as remote
git remote add origin https://github.com/your-username/zylia.git

# Push to GitHub
git push -u origin main
```

Be sure to create a new repository on GitHub before pushing. The `.gitignore` file is already configured to exclude sensitive and large files.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for advanced language processing
- OpenAI Whisper for speech recognition
- Mozilla TTS for neural voice synthesis
- Llama models for local language processing
- ChromaDB for vector storage capabilities 