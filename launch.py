import subprocess

python_path = "/home/nischay/linenv311/bin/python"
files_list = [
    "central.py", "mic.py", "transcribe.py",
    "session_mgr.py", "speaker.py", "ui_client.py"
]

for script in files_list:
    subprocess.Popen([
        "xterm", "-hold", "-e", f"{python_path} {script}"
    ])
