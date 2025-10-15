import subprocess

files_list = [
    "central.py", "speaker.py", "ai_handler.py","transcribe.py", "mic.py", "ui_client.py","session_mgr.py"
]

for script in files_list:
    subprocess.run(["pkill", "-f", script], check=False)
    print(f"Stopped {script}")
