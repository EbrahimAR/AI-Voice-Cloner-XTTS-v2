import webbrowser
import threading
import time
import os
import sys
import platform
import subprocess

APP_FILE = "app.py"   # your Streamlit app filename
PORT = "8502"

def open_browser():
    """Open the default web browser once after server starts."""
    time.sleep(2)  # wait for Streamlit to boot
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    # Start background thread to open browser
    threading.Thread(target=open_browser, daemon=True).start()

    # Detect platform and run streamlit
    if platform.system() == "Windows":
        os.system(f"streamlit run {APP_FILE} --server.port {PORT} --server.headless true")
    else:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", APP_FILE,
            "--server.port", PORT,
            "--server.headless", "true"
        ])
