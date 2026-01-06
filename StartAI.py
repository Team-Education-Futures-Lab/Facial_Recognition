import sys
import os
import subprocess

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

scripts = [
    os.path.join(base_path, "Server", "server.py"),
    os.path.join(base_path, "FacialRecognition", "Face_Hand_Recognition.py"),
    os.path.join(base_path, "AI_Vocal_Tone_Recognition", "mainVocalAI.py")
]

processes = [subprocess.Popen([sys.executable, script]) for script in scripts]

for p in processes:
    p.wait()