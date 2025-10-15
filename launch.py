import subprocess

python_path = "/home/nischay/linenv311/bin/python"
files_list = [
    "central.py", "speaker.py", "ai_handler.py","transcribe.py", "mic.py"
]

for script in files_list:
    subprocess.Popen([
        "xterm", "-hold", "-e", f"{python_path} {script}"
    ])
