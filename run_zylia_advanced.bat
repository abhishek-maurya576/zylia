@echo off
echo.
echo ===================================================
echo            ZYLIA ADVANCED VOICE ASSISTANT           
echo ===================================================
echo.

REM Set the Google API key as an environment variable (replace with your own or use .env file)
set GOOGLE_API_KEY=AIzaSyAh_yhZr8YEzqz2ucu63yoEFkQ4bPY35sQ

echo Initializing ZYLIA Advanced...
echo.
echo This version includes:
echo  - Neural voice synthesis (whisper lady voice)
echo  - Local LLM for offline operation
echo  - Vector memory with semantic search
echo  - Enhanced speech recognition
echo.
echo First-time startup may take several minutes to download models.
echo.

echo MICROPHONE TIPS:
echo  - Make sure your microphone is connected and selected as default
echo  - Speak clearly and directly into the microphone
echo  - Try to minimize background noise
echo.

echo VOICE INFORMATION:
echo  - ZYLIA now speaks in a soft whisper-like lady voice
echo  - The voice is designed to be gentle and soothing
echo  - If you can't hear ZYLIA clearly, check your speaker volume
echo.

echo SPECIAL COMMANDS:
echo  - "Go offline" - Switch to local LLM mode (no internet needed)
echo  - "Go online" - Switch back to using Gemini API
echo  - "What is today's date?" - Get the current date
echo  - "What time is it?" - Get the current time
echo.

echo PERFORMANCE NOTES:
echo  - First-time model downloads require internet connection
echo  - Local LLM requires 4GB+ RAM to operate
echo  - Neural TTS may use significant CPU resources
echo.

echo ===================================================
echo.
echo Starting ZYLIA...
echo.

python src/main.py

echo.
echo Thank you for using ZYLIA. Press any key to exit.
pause > nul 