# AI Voice Cloner (XTTS v2) - Streamlit app
# NOTE: requires ffmpeg installed on system for MP3 export (not a pip package).

import os
import io
import time
import json
import uuid
import shutil
import threading
import base64
import librosa
import warnings
warnings.filterwarnings("ignore", message=".*Torchaudio.*")
import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
from datetime import datetime
from TTS.api import TTS

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="AI Voice Cloner (XTTS v2)",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session-state defaults (persist across reruns)
# -----------------------------
# We use session_state to store actual data (file paths, base64 strings)
# so we never attempt to iterate over Streamlit DeltaGenerator objects.
if "speaker_path" not in st.session_state:
    st.session_state.speaker_path = None
if "ref_display_name" not in st.session_state:
    st.session_state.ref_display_name = None
if "recorded_b64" not in st.session_state:
    st.session_state.recorded_b64 = None

# -----------------------------
# Custom Dark Theme Styling
# -----------------------------
st.markdown("""
<style>
/* General page tweaks */
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
}

/* Card-like look for containers */
.block-container {
    padding: 2rem 2rem;
    border-radius: 12px;
}

/* Text areas and inputs */
textarea, input, .stTextInput>div>div>input {
    background-color: #161B22 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #2E343F !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #1f1c2c, #928DAB) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #485563, #29323c) !important;
    transform: scale(1.03);
}

/* Radio buttons and selects */
.stRadio>div, .stSelectbox>div>div {
    background-color: #161B22 !important;
    border-radius: 6px !important;
    padding: 0.4rem !important;
}
.stRadio label, .stSelectbox label {
    color: #FFFFFF !important;
}

/* Fix dropdown text + background visibility (Base Web Select + its portal menu) */
.stApp [data-baseweb="select"] > div {
    color: #FFFFFF !important;
    background-color: #161B22 !important;
    border: 1px solid #2E343F !important;
}
.stApp [data-baseweb="select"] input {
    color: #FFFFFF !important;
}
.stApp [data-baseweb="select"] svg {
    fill: #FFFFFF !important;
}
/* The options menu is rendered in a portal with data-baseweb="menu" */
.stApp div[data-baseweb="menu"] {
    background-color: #161B22 !important;
    color: #FFFFFF !important;
    border: 1px solid #2E343F !important;
}
.stApp li[data-baseweb="menu-item"] {
    background-color: #161B22 !important;
    color: #FFFFFF !important;
}
.stApp li[data-baseweb="menu-item"]:hover {
    background-color: #242B36 !important;
}
.stApp li[data-baseweb="menu-item"][aria-selected="true"] {
    background-color: #2A313D !important;
}

/* Download buttons */
.stDownloadButton>button {
    background: linear-gradient(135deg, #1f1c2c, #928DAB) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton>button:hover {
    background: linear-gradient(135deg, #485563, #29323c) !important;
    transform: scale(1.02);
}

/* Audio player styling */
audio {
    border-radius: 10px;
    outline: none;
    margin-top: 0.5rem;
}

/* Divider styling */
hr {
    border: none;
    border-top: 1px solid #2E343F;
    margin: 2rem 0;
}

/* Card styling */
.card {
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 12px;
    background: #1E222A;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
}

/* Remove empty space at top */
.stApp > header {
    display: none;
}

/* Fix for empty bar issue */
.appview-container .main .block-container {
    padding-top: 2rem;
}

/* Fix dropdown selected text being cut off */
div[data-baseweb="select"] > div {
    height: auto !important;
    min-height: 38px !important;
    padding-top: 6px !important;
    padding-bottom: 6px !important;
    white-space: normal !important;
    overflow: visible !important;
}

/* Make selected value fully visible */
div[data-baseweb="select"] span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants & Dirs
# -----------------------------
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cpu"  # change to "cuda" if you set up GPU and compatible TTS/torch

VOICES_DIR = "voices"
OUTPUTS_DIR = "outputs"
META_DIR = os.path.join(OUTPUTS_DIR, "_meta")
for d in (VOICES_DIR, OUTPUTS_DIR, META_DIR):
    os.makedirs(d, exist_ok=True)

RECENTS_LIMIT = 8  # how many generations to show in gallery

# -----------------------------
# Helpers (TTS & audio)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_tts_model():
    """Load XTTS model once and cache. No Streamlit calls inside."""
    return TTS(MODEL_NAME, progress_bar=True).to(DEVICE)

def _trim_silence(y, sr, top_db=28, min_len_sec=0.1):
    """Trim long silent regions from audio signal."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    chunks = []
    for start, end in intervals:
        if (end - start) / sr >= min_len_sec:
            chunks.append(y[start:end])
    if not chunks:
        return y
    return np.concatenate(chunks)

def ensure_wav_mono16(file_bytes: bytes, *_ignore, do_trim=True, trim_db=28) -> str:
    """
    Convert input bytes to WAV mono 16k PCM and save to VOICES_DIR.
    Returns the saved path.
    """
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    if do_trim:
        y = _trim_silence(y, sr, top_db=trim_db)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 0:
        y = 0.98 * y / peak
    out_path = os.path.join(VOICES_DIR, f"ref_{int(time.time())}.wav")
    sf.write(out_path, y, sr, subtype="PCM_16")
    return out_path

def synthesize(tts, text: str, speaker_wav_path: str, language: str = "en",
               style: str = "neutral", speed_adv: float | None = None,
               emotion_adv: str | None = None) -> str:
    """Run XTTS to synthesize to a new WAV file; returns path to WAV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = os.path.join(OUTPUTS_DIR, f"xtts_{ts}.wav")

    style_map = {
        "neutral": {"speed": 1.0, "emotion": None},
        "fast": {"speed": 1.1, "emotion": None},
        "expressive": {"speed": 1.0, "emotion": "happy"},
    }
    params = style_map.get(style, {"speed": 1.0, "emotion": None})

    if speed_adv is not None:
        params["speed"] = float(speed_adv)
    if emotion_adv is not None and emotion_adv != "None":
        params["emotion"] = emotion_adv
    if params.get("emotion") == "None":
        params["emotion"] = None

    tts.tts_to_file(
        text=text,
        file_path=out_wav,
        speaker_wav=speaker_wav_path,
        language=language,
        speed=params["speed"],
        emotion=params["emotion"],
    )
    return out_wav

def normalize_loudness(wav_path: str) -> str:
    """Normalize loudness and return path to normalized WAV."""
    snd = AudioSegment.from_wav(wav_path)
    change_needed = -1.0 - snd.max_dBFS if hasattr(snd, "max_dBFS") and snd.max_dBFS is not None else 0
    if np.isfinite(change_needed) and abs(change_needed) > 0.1:
        snd = snd.apply_gain(change_needed)
    norm_path = wav_path.replace(".wav", "_norm.wav")
    snd.export(norm_path, format="wav")
    return norm_path

def convert_to_mp3(wav_path: str) -> str:
    """Convert WAV to MP3 using ffmpeg via pydub."""
    try:
        sound = AudioSegment.from_wav(wav_path)
    except Exception as e:
        raise RuntimeError(f"Could not open WAV for MP3 conversion: {e}")
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        sound.export(mp3_path, format="mp3", bitrate="192k")
    except FileNotFoundError:
        raise RuntimeError(
            "MP3 conversion requires ffmpeg. Install it and ensure it's on PATH.\n"
            "• Windows (choco): choco install ffmpeg\n"
            "• macOS (brew): brew install ffmpeg\n"
            "• Linux (apt): sudo apt-get install ffmpeg"
        )
    return mp3_path

def get_voice_gallery():
    return sorted([f for f in os.listdir(VOICES_DIR) if f.lower().endswith(".wav")])

def save_generation_meta(out_wav: str, text: str, language: str, style: str,
                         ref_name: str | None, extras: dict | None = None):
    """Save generation metadata JSON for the recent gallery."""
    meta = {
        "id": str(uuid.uuid4()),
        "created": datetime.now().isoformat(),
        "wav": os.path.basename(out_wav),
        "text": text[:5000],
        "language": language,
        "style": style,
        "reference": ref_name,
        "extras": extras or {},
    }
    meta_path = os.path.join(META_DIR, meta["id"] + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path

def load_recent_meta(n=RECENTS_LIMIT):
    files = [os.path.join(META_DIR, f) for f in os.listdir(META_DIR) if f.endswith(".json")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    metas = []
    for p in files[:n]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                metas.append(json.load(f))
        except Exception:
            continue
    return metas

# -----------------------------
# Browser-based audio recording (returns base64 to Python)
# - This component uses window.parent.postMessage to send a dataURL back.
# - components.html will return the posted value (a string) when available.
# - We still guard the value: components.html may return a DeltaGenerator or None
#   on initial renders; we therefore only accept/process strings.
# -----------------------------
def audio_recorder_component() -> str | None:
    """Inline HTML5 recorder with pulsing mic animation while recording.

    Returns:
        dataURL string (e.g. "data:audio/webm;base64,....") when recording is complete,
        otherwise returns None / DeltaGenerator (which we ignore).
    """
    # Note: components.html returns the value set via Streamlit.setComponentValue (or postMessage)
    # when the component posts it. On first renders it may return a DeltaGenerator - do not rely on it.
    return components.html(
        """
        <style>
        .rec-btn {
            background: linear-gradient(135deg, #1f1c2c, #928DAB);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin: 6px 0;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.18s ease-in-out;
        }
        .rec-btn:hover:enabled {
            background: linear-gradient(135deg, #485563, #29323c);
            transform: scale(1.02);
        }
        .rec-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        #recording-status {
            margin: 10px 0;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            color: #FFFFFF;
        }
        /* Mic pulse animation */
        .pulse {
            display: inline-block;
            font-size: 18px;
            animation: pulseAnim 1s infinite;
        }
        @keyframes pulseAnim {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.25); opacity: 0.6; }
            100% { transform: scale(1); opacity: 1; }
        }
        </style>

        <div id="audio-recorder" style="text-align:center;">
            <button id="startBtn" class="rec-btn">🎤 Start Recording</button>
            <button id="stopBtn" class="rec-btn" disabled>⏹️ Stop Recording</button>
            <div id="recording-status">Ready</div>
            <audio id="audio-playback" controls style="width: 100%; margin: 10px 0; display: none; border-radius: 8px;"></audio>
        </div>

        <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('recording-status');
        const audioEl = document.getElementById('audio-playback');
        let mediaRecorder; let chunks = [];

        function setHeight(){ if (window.Streamlit) { window.Streamlit.setFrameHeight(300); } }
        setHeight();

        startBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                chunks = [];
                mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
                mediaRecorder.onstop = async () => {
                    const blob = new Blob(chunks, { type: 'audio/webm' });
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64data = reader.result; // data:audio/webm;base64,...
                        audioEl.src = base64data;
                        audioEl.style.display = 'block';
                        statusDiv.innerHTML = '<span style="color:#7ee787">✅ Recording captured</span>';
                        // Post the value back to Streamlit component (components.html captures this)
                        if (window.parent) {
                            window.parent.postMessage({
                                isStreamlitMessage: true,
                                type: "streamlit:setComponentValue",
                                value: base64data
                            }, "*");
                        }
                    };
                    reader.readAsDataURL(blob);
                };
                mediaRecorder.start();
                startBtn.disabled = true; stopBtn.disabled = false;
                statusDiv.innerHTML = '<span class="pulse">🎤 Recording...</span>';
            } catch (err) {
                statusDiv.innerHTML = '<span style="color:#ff6b6b">Error: ' + err.message + '</span>';
            }
        });

        stopBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                startBtn.disabled = false; stopBtn.disabled = true;
                statusDiv.innerText = 'Processing...';
            }
        });
        </script>
        """,
        height=320,
    )

# -----------------------------
# Save recorded/uploaded audio safely
# - save_audio_from_base64 expects a dataURL string "data:...;base64,AAAA.."
# - It returns saved WAV path or None on error
# -----------------------------
def save_audio_from_base64(base64_audio: str) -> str | None:
    """Save base64 audio (data URL) to WAV mono 16k for XTTS. Returns filepath or None."""
    try:
        if not base64_audio or not isinstance(base64_audio, str):
            raise ValueError("No valid base64 data URL provided.")

        # Extract payload, data URLs have form: data:<mime>;base64,<payload>
        if "," in base64_audio:
            header, payload = base64_audio.split(",", 1)
        else:
            # if not a dataURL, assume payload only
            payload = base64_audio

        audio_bytes = io.BytesIO(base64.b64decode(payload))

        # Load/trim/resample
        y, sr = librosa.load(audio_bytes, sr=None, mono=True)
        y = _trim_silence(y, sr, top_db=28)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        peak = np.max(np.abs(y)) if y.size else 1.0
        if peak > 0:
            y = 0.98 * y / peak

        final_path = os.path.join(VOICES_DIR, f"recorded_{int(time.time())}.wav")
        sf.write(final_path, y, sr, subtype="PCM_16")
        return final_path
    except Exception as e:
        # Provide actionable error for the user (don't raise unhandled)
        st.error(f"Error processing audio: {e}")
        return None

# -----------------------------
# Sidebar (Settings)
# -----------------------------
with st.sidebar:
    st.subheader("⚙️ Settings")
    language = st.selectbox(
        "Language",
        ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hi"],
        index=0,
        help="Select the language for voice synthesis"
    )

    style = st.radio("Voice style", ["neutral", "fast", "expressive"], index=0)
    advanced = st.checkbox("Advanced controls", value=False)
    speed_adv = None
    emotion_adv = None
    if advanced:
        speed_adv = st.slider("Speed (0.8–1.5)", min_value=0.8, max_value=1.5, value=1.0, step=0.01)
        emotion_adv = st.selectbox("Emotion (XTTS v2)", ["None", "happy", "sad", "angry", "surprised", "fearful"], index=0)

    st.markdown("---")
    st.markdown("**Natural Voice Settings**")
    voice_stability = st.slider("Voice stability", 0.0, 1.0, 0.5, 0.1,
                               help="Higher values make voice more stable but less expressive")
    voice_similarity = st.slider("Voice similarity", 0.0, 1.0, 0.75, 0.1,
                                help="Higher values make output more similar to reference voice")

    normalize_out = st.checkbox("Normalize output loudness", value=True)
    st.markdown(f"**Device:** `{DEVICE}`")

    if st.button("🚀 Load / Warm up model", use_container_width=True):
        t0 = time.time()
        with st.spinner("Loading XTTS model..."):
            _ = load_tts_model()
        st.success(f"Model ready on {DEVICE} in {time.time() - t0:.1f}s")

# -----------------------------
# Header & Text
# -----------------------------
st.title("🗣️ AI Voice Cloner")
st.caption("Coqui XTTS v2 • For best results, use 20–30 seconds of clean reference speech.")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📝 Enter Text")
text = st.text_area(
    "Text to speak",
    "Hello! This is my cloned voice running from a Streamlit web app. The voice should sound natural and expressive.",
    height=140,
    label_visibility="collapsed",
    help="Use punctuation and natural phrasing for better results. Avoid very long sentences."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Reference Voice Selection
# -----------------------------
st.markdown("### 🎵 Choose Reference Voice")

voice_option = st.radio(
    "Select input method",
    ["Upload new voice", "Use voice from gallery", "Record live"],
    horizontal=True
)

speaker_path = st.session_state.speaker_path
ref_display_name = st.session_state.ref_display_name

# --- Upload new voice
if voice_option == "Upload new voice":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload reference voice (20–30s clean speech)",
        type=["wav", "mp3", "m4a", "flac", "ogg"]
    )
    trim_ref = st.checkbox("Trim long silences from reference", value=True)
    trim_db = st.slider("Trim sensitivity (dB)", 20, 40, 28)
    if uploaded:
        file_bytes = uploaded.read()
        with st.spinner("Preprocessing uploaded voice..."):
            try:
                speaker_path = ensure_wav_mono16(file_bytes, uploaded.name, do_trim=trim_ref, trim_db=trim_db)
                ref_display_name = os.path.basename(speaker_path)
                st.success(f"Prepared reference: {ref_display_name}")
                st.audio(speaker_path)
                y, sr = librosa.load(speaker_path, sr=None, mono=True)
                dur = len(y) / sr if sr else 0.0
                if dur < 8:
                    st.info("ℹ️ Reference is quite short (<8s). Longer samples often clone better.")
                elif dur > 60:
                    st.info("ℹ️ Reference is long (>60s). 20–30s usually works well.")
                st.session_state.speaker_path = speaker_path
                st.session_state.ref_display_name = ref_display_name
            except Exception as e:
                st.error(f"Upload error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Use saved voice from gallery
elif voice_option == "Use voice from gallery":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    existing_refs = get_voice_gallery()
    if existing_refs:
        cols = st.columns([3, 2])
        with cols[0]:
            use_existing = st.selectbox("Saved voices", existing_refs)
        with cols[1]:
            if st.button("🗑️ Delete selected voice", use_container_width=True):
                try:
                    os.remove(os.path.join(VOICES_DIR, use_existing))
                    st.success(f"Deleted {use_existing}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        if existing_refs:
            speaker_path = os.path.join(VOICES_DIR, use_existing)
            ref_display_name = use_existing
            st.audio(speaker_path)
            st.session_state.speaker_path = speaker_path
            st.session_state.ref_display_name = ref_display_name
    else:
        st.warning("No saved voices yet. Upload or record one.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Record live in browser
elif voice_option == "Record live":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎤 Record your voice in browser")

    # Call recorder component - it may return None or a DeltaGenerator during initial renders.
    recorded_b64 = audio_recorder_component()

    # IMPORTANT: components.html can sometimes return a DeltaGenerator or None on initial runs.
    # Only accept/process the value when it's a string (dataURL). Store it in session_state.
    if isinstance(recorded_b64, str) and recorded_b64.strip():
        st.session_state.recorded_b64 = recorded_b64

    # If we have a recorded base64 in session_state, process it (save, normalize, etc.)
    if st.session_state.get("recorded_b64"):
        with st.spinner("Processing recorded audio..."):
            try:
                # Use helper to convert dataURL -> WAV mono16k and save to voices/
                new_path = save_audio_from_base64(st.session_state.recorded_b64)
                if new_path:
                    speaker_path = new_path
                    ref_display_name = os.path.basename(new_path)
                    st.success(f"Prepared reference: {ref_display_name}")
                    st.audio(speaker_path)
                    st.session_state.speaker_path = speaker_path
                    st.session_state.ref_display_name = ref_display_name
                    # Clear recorded_b64 if you prefer one-time processing:
                    # st.session_state.recorded_b64 = None
            except Exception as e:
                st.error(f"Recording error: {e}")

    st.markdown("---")
    st.markdown("### 📤 Or upload a recording")
    fallback_upload = st.file_uploader(
        "Upload your recorded voice",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        key="fallback_upload"
    )

    if fallback_upload:
        file_bytes = fallback_upload.read()
        with st.spinner("Processing uploaded recording..."):
            try:
                speaker_path = ensure_wav_mono16(file_bytes, fallback_upload.name, do_trim=True, trim_db=28)
                ref_display_name = os.path.basename(speaker_path)
                st.success(f"Prepared reference: {ref_display_name}")
                st.audio(speaker_path)
                st.session_state.speaker_path = speaker_path
                st.session_state.ref_display_name = ref_display_name
            except Exception as e:
                st.error(f"Upload error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Generate Voice
# -----------------------------
st.markdown("### 🎙️ Generate")

btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    generate_disabled = (not text.strip()) or (not st.session_state.speaker_path)
    generate = st.button("🎬 Generate Voice", type="primary", use_container_width=True, disabled=generate_disabled)
with btn_col2:
    if st.button("♻️ Reset UI", use_container_width=True):
        # Reset only the states we want cleared
        st.session_state.speaker_path = None
        st.session_state.ref_display_name = None
        st.session_state.recorded_b64 = None
        st.rerun()

if generate and st.session_state.speaker_path:
    t0 = time.time()
    with st.spinner("Synthesizing..."):
        try:
            tts = load_tts_model()
            out_wav = synthesize(
                tts, text.strip(), st.session_state.speaker_path,
                language=language, style=style,
                speed_adv=speed_adv, emotion_adv=emotion_adv
            )
            final_wav = out_wav
            if normalize_out:
                try:
                    final_wav = normalize_loudness(out_wav)
                except Exception:
                    final_wav = out_wav

            extras = {
                "device": DEVICE,
                "normalized": normalize_out,
                "speed": speed_adv,
                "emotion": emotion_adv,
                "stability": voice_stability,
                "similarity": voice_similarity
            }
            save_generation_meta(final_wav, text.strip(), language, style, st.session_state.ref_display_name, extras)

            st.success(f"✅ Done in {time.time() - t0:.1f}s")
            col_wav, col_mp3 = st.columns(2)
            with col_wav:
                st.markdown("**WAV**")
                st.audio(final_wav, format="audio/wav")
                with open(final_wav, "rb") as f:
                    st.download_button("⬇️ Download WAV", data=f.read(),
                                       file_name=os.path.basename(final_wav), mime="audio/wav", use_container_width=True)
            with col_mp3:
                st.markdown("**MP3**")
                try:
                    out_mp3 = convert_to_mp3(final_wav)
                    st.audio(out_mp3, format="audio/mpeg")
                    with open(out_mp3, "rb") as f:
                        st.download_button("⬇️ Download MP3", data=f.read(),
                                           file_name=os.path.basename(out_mp3), mime="audio/mpeg", use_container_width=True)
                except Exception as e:
                    st.warning(f"MP3 export unavailable: {e}")

        except Exception as e:
            st.error(f"Generation failed: {e}")

st.markdown("---")

# -----------------------------
# Recent Generations Gallery
# -----------------------------
st.markdown("### 📁 Recent Generations")
metas = load_recent_meta(RECENTS_LIMIT)
if not metas:
    st.info("No generations yet. Create one to see it here.")
else:
    for m in metas:
        with st.expander(f"🕒 {m['created']} • {m['language']} • {m['style']} • ref: {m.get('reference') or '—'}"):
            wav_path = os.path.join(OUTPUTS_DIR, m["wav"])
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown("**Preview**")
                if os.path.exists(wav_path):
                    st.audio(wav_path)
                else:
                    st.error("Original WAV missing.")
            with cols[1]:
                if os.path.exists(wav_path):
                    with open(wav_path, "rb") as f:
                        st.download_button("⬇️ WAV", data=f.read(), file_name=os.path.basename(wav_path), mime="audio/wav", use_container_width=True)
                if st.button("🔁 Re-synthesize with same settings", key=m["id"]):
                    try:
                        tts = load_tts_model()
                        ref_file = None
                        if m.get("reference"):
                            ref_candidate = os.path.join(VOICES_DIR, m["reference"])
                            if os.path.exists(ref_candidate):
                                ref_file = ref_candidate
                        if not ref_file:
                            st.error("Original reference voice not found in gallery.")
                        else:
                            out_wav2 = synthesize(
                                tts, m["text"], ref_file,
                                language=m["language"],
                                style=m["style"],
                                speed_adv=m.get("extras", {}).get("speed"),
                                emotion_adv=m.get("extras", {}).get("emotion"),
                            )
                            final2 = out_wav2
                            if m.get("extras", {}).get("normalized", False):
                                try:
                                    final2 = normalize_loudness(out_wav2)
                                except Exception:
                                    final2 = out_wav2
                            save_generation_meta(final2, m["text"], m["language"], m["style"], m.get("reference"), m.get("extras"))
                            st.success("Re-synthesis done.")
                            st.audio(final2)
                    except Exception as e:
                        st.error(f"Re-synthesis failed: {e}")

st.markdown("---")

# -----------------------------
# Tips
# -----------------------------
with st.expander("🧪 Tips for natural sounding voice"):
    st.markdown("""
**For more natural sounding voice:**

1. **Reference Audio Quality:**
   - Use 20-30 seconds of clean, single-speaker audio
   - Record in a quiet environment with minimal background noise
   - Use a good quality microphone

2. **Text Input:**
   - Use proper punctuation (commas, periods, question marks)
   - Break long sentences into shorter ones
   - Use natural phrasing and contractions ("don't" instead of "do not")

3. **Settings:**
   - Use "neutral" style for most natural results
   - Adjust speed to 0.9-1.1 for more natural pacing
   - Try different emotions for different contexts
   - Increase voice stability for consistent output
   - Increase voice similarity to match reference more closely

4. **Reference Speech:**
   - Use reference audio with similar emotion to desired output
   - Match the speaking style (conversational, narrative, etc.)
   - Ensure reference audio has clear pronunciation

**For browser recording:**
- Chrome/Firefox recommended
- Allow microphone permissions when prompted
- Speak clearly and consistently at a steady distance
- Record in a quiet environment
    """)
