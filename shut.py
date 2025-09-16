import subprocess

files_list = [
    "central.py",
    "mic.py",
    "transcribe.py",
    "session_mgr.py",
    "speaker.py",
    "ui_client.py"
]

for script in files_list:
    subprocess.run(["pkill", "-f", script], check=False)
    print(f"Stopped {script}")
