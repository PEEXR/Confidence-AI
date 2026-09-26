import os
import shutil
import subprocess
import sys

url = sys.argv[1] if len(sys.argv) > 1 else "https://sb-061b95ff14d70cc0.sb.molab.run/"

# Locate opencode.exe binary directly on Windows
npm_exe = os.path.expandvars(r"%APPDATA%\npm\node_modules\opencode-ai\bin\opencode.exe")
opencode_cmd = npm_exe if os.path.exists(npm_exe) else shutil.which("opencode")

if not opencode_cmd:
    print("Error: Could not locate opencode executable.")
    sys.exit(1)

print(f"Fetching Molab pairing token and prompt for: {url} ...")
try:
    prompt = subprocess.check_output(
        ["uvx", "marimo@latest", "pair", "prompt", "--url", url, "--with-token", "--opencode"],
        text=True
    )
except subprocess.CalledProcessError as e:
    print(f"Error fetching pair prompt: {e}")
    sys.exit(1)

print("Starting OpenCode paired with Molab...")
# Launch OpenCode directly with the prompt
subprocess.run([opencode_cmd, "--prompt", prompt])
