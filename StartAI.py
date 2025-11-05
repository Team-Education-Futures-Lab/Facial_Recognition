import subprocess

scripts = ["./FacialRecognition/Face_Hand_Recognition.py", "./AI_Vocal_Tone_Recognition/mainVocalAI.py", "./Server/server.py"]

processes = [subprocess.Popen(["python", script]) for script in scripts]

# Wait for all to finish (optional)
for p in processes:
    p.wait()
