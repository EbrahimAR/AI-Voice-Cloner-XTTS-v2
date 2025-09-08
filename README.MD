# 🗣️ AI Voice Cloner (XTTS v2)

A **Streamlit web app** for cloning voices using [Coqui TTS](https://github.com/coqui-ai/TTS).  
Supports uploading, recording, or selecting reference voices and generating natural-sounding speech in multiple languages.

---

## 🚀 Features
- 🎤 Record voice directly in the browser
- 📤 Upload reference voice samples (WAV, MP3, M4A, OGG, FLAC)
- 🎵 Save and reuse voices from a gallery
- 🌎 Multilingual support (English, Spanish, French, German, Chinese, Japanese, etc.)
- 🎭 Control speech style, speed, and emotion
- 📂 Keep track of recent generations with playback & downloads
- 📥 Export audio as WAV or MP3

---

## 🛠 Tech Stack
### **Frontend / UI**
  - [Streamlit](https://streamlit.io/): Web interface for interaction and real-time updates
  - **streamlit-option-menu**: Sidebar navigation with a clean UI

### **Backend / Logic**
  - [Coqui TTS (XTTS v2)](https://github.com/coqui-ai/TTS): Voice cloning & text-to-speech model
  - **pydub**: Audio file conversion (WebM/MP3 ↔ WAV)
  - **librosa & soundfile**: Audio signal processing
  - **NumPy**: Array and numerical operations

### **Utilities**
  - **requests**: API calls (if extending functionality)
  - **python-dotenv**: Manage secrets/API keys
  - **ffmpeg (system dependency)**: Required for audio encoding/decoding

### **Environment**
  - **requests**: API calls (if extending functionality)
  - Virtual Environment (`venv/`) for dependency isolation


---

## 📂 Project Structure
  voice_clone_app/
  │
  ├── outputs/                  # Generated outputs (auto-created)
  │   ├── _meta/                # Metadata of generations
  │   └── xtts_*.wav            # Generated files
  │
  ├── voices/                   # Reference voices (auto-created)
  │   ├── myvoice.wav
  │   ├── ref_*.wav             # Uploaded/recorded samples
  │
  ├── app.py                    # Main Streamlit application
  ├── run_app.py                 # Launcher (cross-platform, auto-opens browser)
  ├── test.py                    # Experimental/testing script
  ├── requirements.txt           # Python dependencies
  ├── LICENSE                    # Open-source license
  ├── .gitignore                 # Ignored files/folders


---

## ⚙️ Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/EbrahimAR/AI-Voice-Cloner-XTTS-v2.git
cd AI-Voice-Cloner-XTTS-v2
```

### **2. Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate    # On Linux/Mac
.venv\Scripts\activate       # On Windows
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Install system dependencies**
- **ffmpeg** is required for MP3 conversion:
  - Windows: `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

## ▶️ Run the app

```bash
streamlit run app.py
```

Or use the helper script:

```bash
python run_app.py
```
App will open at: http://localhost:8502

---

## 👨‍💻 Author
Ebrahim Abdul Raoof

[LinkedIn](https://www.linkedin.com/in/ebrahim-ar/)

[GitHub](https://github.com/EbrahimAR)

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](https://github.com/EbrahimAR/AIExamSystem/blob/main/LICENSE) for details.
