import subprocess

files_list = [
    "central.py", "speaker.py", "ai_handler.py","transcribe.py", "mic.py"
]

for script in files_list:
    subprocess.run(["pkill", "-f", script], check=False)
    print(f"Stopped {script}")
