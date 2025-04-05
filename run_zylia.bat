@echo off
echo.
echo ===================================================
echo            ZYLIA VOICE ASSISTANT SETUP             
echo ===================================================
echo.

REM Set the Google API key as an environment variable
set GOOGLE_API_KEY=AIzaSyAh_yhZr8YEzqz2ucu63yoEFkQ4bPY35sQ

echo Checking microphone settings...
echo.
echo MICROPHONE TIPS:
echo  - Make sure your microphone is connected and selected as default
echo  - Speak clearly and directly into the microphone
echo  - Try to minimize background noise
echo  - If voice recognition doesn't work, you can always type your commands
echo.
echo PRONUNCIATION TIPS:
echo  - When asking for today's date, pronounce DATE as "dayt" (not "dayta")
echo  - Try using full sentences like "What is today's date?"
echo  - Speak at a moderate pace, not too fast
echo.
echo VOICE INFORMATION:
echo  - ZYLIA now speaks in a soft whisper-like lady voice
echo  - The voice is designed to be gentle and soothing
echo  - If you can't hear ZYLIA clearly, check your speaker volume
echo.
echo EXAMPLE COMMANDS:
echo  - "What is today's date?"
echo  - "What time is it?"
echo  - "Tell me about [any topic]"
echo.
echo ===================================================
echo.
echo Starting ZYLIA...

python src/main.py

echo.
echo Thank you for using ZYLIA. Press any key to exit.
pause > nul 