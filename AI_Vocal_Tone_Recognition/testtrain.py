import glob, os

base_path = os.path.join(os.path.dirname(__file__), "data", "audio_speech_actors_01-24")
print("🔍 Looking in:", os.path.abspath(base_path))
files = glob.glob(os.path.join(base_path, "**", "*.wav"), recursive=True)
print("✅ Found", len(files), "audio files")
if files:
    print("Example file:", files[0])