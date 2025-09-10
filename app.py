#!/usr/bin/env python3
"""
AI Voice Cloner using Coqui XTTS v2

A Streamlit web application for voice cloning using the XTTS v2 model.
Users can upload reference audio or record their own voice to clone,
then generate speech in the cloned voice from text input.

Requirements:
- Python 3.8+
- FFmpeg (for MP3 export)
- CUDA-compatible GPU (optional, for faster inference)

Author: AI Voice Cloner Project
License: MIT
"""

# Standard library imports
import os
import io
import time
import json
import uuid
import shutil
import threading
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any

# Third-party imports
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
from TTS.api import TTS

# Suppress torchaudio warnings
import warnings
warnings.filterwarnings("ignore", message=".*Torchaudio.*")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Streamlit page configuration
st.set_page_config(
    page_title="AI Voice Cloner (XTTS v2)",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Model and device configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cpu"  # Change to "cuda" if you have a compatible GPU setup

# Directory configuration
VOICES_DIR = "voices"           # Directory for reference voice files
OUTPUTS_DIR = "outputs"         # Directory for generated audio files
META_DIR = os.path.join(OUTPUTS_DIR, "_meta")  # Metadata storage

# Application settings
RECENTS_LIMIT = 8  # Number of recent generations to show in gallery

# Create directories if they don't exist
for directory in (VOICES_DIR, OUTPUTS_DIR, META_DIR):
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables with default values."""
    default_states = {
        "speaker_path": None,
        "ref_display_name": None,
        "recorded_b64": None,
        "recording_processed": False,
        "pending_delete": None,
        "active_voice": None,  # Currently active voice filename
        "batch_delete": [],
        "confirm_batch_delete": False
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# =============================================================================
# STYLING
# =============================================================================

def apply_custom_styling():
    """Apply custom CSS styling for the Streamlit app."""
    st.markdown("""
    <style>
    /* General page styling */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }

    /* Container styling */
    .block-container {
        padding: 2rem 2rem;
        border-radius: 12px;
    }

    /* Input field styling */
    textarea, input, .stTextInput>div>div>input {
        background-color: #161B22 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid #2E343F !important;
    }

    /* Button styling */
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

    /* Radio buttons and select boxes */
    .stRadio>div, .stSelectbox>div>div {
        background-color: #161B22 !important;
        border-radius: 6px !important;
        padding: 0.4rem !important;
    }
    .stRadio label, .stSelectbox label {
        color: #FFFFFF !important;
    }

    /* Dropdown styling */
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

    /* Download button styling */
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

    /* Layout improvements */
    .appview-container .main .block-container {
        padding-top: 2rem;
    }

    /* Dropdown text visibility fixes */
    div[data-baseweb="select"] > div {
        height: auto !important;
        min-height: 38px !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        white-space: normal !important;
        overflow: visible !important;
    }
    div[data-baseweb="select"] span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    /* Active voice badge */
    .active-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, #2ee082, #1fb5a6);
        color: #042023;
        font-weight: 700;
        margin-left: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply styling
apply_custom_styling()

# =============================================================================
# CORE TTS AND AUDIO PROCESSING FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner=True)
def load_tts_model() -> TTS:
    """
    Load and cache the XTTS v2 model.
    
    Returns:
        TTS: Loaded TTS model instance
        
    Note:
        This function is cached to avoid reloading the model on every run.
        The model is loaded once and reused across sessions.
    """
    return TTS(MODEL_NAME, progress_bar=True).to(DEVICE)


def trim_silence(audio_data: np.ndarray, sample_rate: int, 
                top_db: int = 28, min_length_sec: float = 0.1) -> np.ndarray:
    """
    Trim long silent regions from audio signal.
    
    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        top_db: Threshold for silence detection
        min_length_sec: Minimum length of audio segments to keep
        
    Returns:
        np.ndarray: Trimmed audio signal
    """
    intervals = librosa.effects.split(audio_data, top_db=top_db)
    if len(intervals) == 0:
        return audio_data
        
    chunks = []
    for start, end in intervals:
        if (end - start) / sample_rate >= min_length_sec:
            chunks.append(audio_data[start:end])
    
    if not chunks:
        return audio_data
        
    return np.concatenate(chunks)


def preprocess_audio_file(file_bytes: bytes, do_trim: bool = True, 
                         trim_db: int = 28) -> str:
    """
    Convert input audio bytes to WAV mono 16kHz PCM format for XTTS.
    
    Args:
        file_bytes: Raw audio file bytes
        do_trim: Whether to trim silence
        trim_db: Threshold for silence trimming
        
    Returns:
        str: Path to the processed WAV file
        
    Raises:
        Exception: If audio processing fails
    """
    try:
        # Load audio with librosa (handles most formats)
        audio_data, sample_rate = librosa.load(
            io.BytesIO(file_bytes), sr=None, mono=True
        )
        
        # Trim silence if requested
        if do_trim:
            audio_data = trim_silence(audio_data, sample_rate, top_db=trim_db)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )
            sample_rate = 16000
        
        # Normalize audio to prevent clipping
        peak = np.max(np.abs(audio_data)) if audio_data.size > 0 else 1.0
        if peak > 0:
            audio_data = 0.98 * audio_data / peak
        
        # Save to voices directory with timestamp
        output_path = os.path.join(VOICES_DIR, f"ref_{int(time.time())}.wav")
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Audio preprocessing failed: {str(e)}")


def synthesize_speech(tts_model: TTS, text: str, speaker_wav_path: str, 
                     language: str = "en", style: str = "neutral",
                     speed_override: Optional[float] = None,
                     emotion_override: Optional[str] = None) -> str:
    """
    Generate speech using XTTS v2 model.
    
    Args:
        tts_model: Loaded TTS model instance
        text: Text to synthesize
        speaker_wav_path: Path to reference speaker audio
        language: Target language code
        style: Voice style preset
        speed_override: Custom speed value (overrides style preset)
        emotion_override: Custom emotion value (overrides style preset)
        
    Returns:
        str: Path to generated WAV file
    """
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUTS_DIR, f"xtts_{timestamp}.wav")
    
    # Style presets
    style_presets = {
        "neutral": {"speed": 1.0, "emotion": None},
        "fast": {"speed": 1.1, "emotion": None},
        "expressive": {"speed": 1.0, "emotion": "happy"},
    }
    
    # Get style parameters or use defaults
    params = style_presets.get(style, {"speed": 1.0, "emotion": None})
    
    # Apply overrides if provided
    if speed_override is not None:
        params["speed"] = float(speed_override)
    if emotion_override is not None and emotion_override != "None":
        params["emotion"] = emotion_override
    if params.get("emotion") == "None":
        params["emotion"] = None
    
    # Generate speech
    tts_model.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=speaker_wav_path,
        language=language,
        speed=params["speed"],
        emotion=params["emotion"],
    )
    
    return output_path


def normalize_audio_loudness(wav_path: str) -> str:
    """
    Normalize audio loudness to a consistent level.
    
    Args:
        wav_path: Path to input WAV file
        
    Returns:
        str: Path to normalized WAV file
    """
    try:
        audio_segment = AudioSegment.from_wav(wav_path)
        
        # Calculate gain needed to reach target loudness
        target_dBFS = -1.0
        if hasattr(audio_segment, "max_dBFS") and audio_segment.max_dBFS is not None:
            gain_needed = target_dBFS - audio_segment.max_dBFS
            if np.isfinite(gain_needed) and abs(gain_needed) > 0.1:
                audio_segment = audio_segment.apply_gain(gain_needed)
        
        # Export normalized audio
        normalized_path = wav_path.replace(".wav", "_norm.wav")
        audio_segment.export(normalized_path, format="wav")
        
        return normalized_path
        
    except Exception as e:
        # Return original path if normalization fails
        print(f"Warning: Loudness normalization failed: {e}")
        return wav_path


def convert_wav_to_mp3(wav_path: str) -> str:
    """
    Convert WAV file to MP3 format using FFmpeg via pydub.
    
    Args:
        wav_path: Path to input WAV file
        
    Returns:
        str: Path to output MP3 file
        
    Raises:
        RuntimeError: If conversion fails or FFmpeg is not available
    """
    try:
        audio_segment = AudioSegment.from_wav(wav_path)
        mp3_path = wav_path.replace(".wav", ".mp3")
        audio_segment.export(mp3_path, format="mp3", bitrate="192k")
        return mp3_path
        
    except FileNotFoundError:
        raise RuntimeError(
            "MP3 conversion requires FFmpeg to be installed and available in PATH.\n"
            "Installation instructions:\n"
            "• Windows (Chocolatey): choco install ffmpeg\n"
            "• macOS (Homebrew): brew install ffmpeg\n"
            "• Linux (APT): sudo apt-get install ffmpeg"
        )
    except Exception as e:
        raise RuntimeError(f"MP3 conversion failed: {str(e)}")

# =============================================================================
# VOICE GALLERY MANAGEMENT
# =============================================================================

def get_voice_gallery() -> List[str]:
    """
    Get list of available voice files from the voices directory.
    
    Returns:
        List[str]: Sorted list of WAV filenames
    """
    try:
        files = os.listdir(VOICES_DIR)
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        return sorted(wav_files)
    except FileNotFoundError:
        return []


def delete_voice_file(filename: str) -> bool:
    """
    Safely delete a voice file and update session state if needed.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        file_path = os.path.join(VOICES_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Clear active voice if the deleted file was active
        if st.session_state.get("active_voice") == filename:
            st.session_state.active_voice = None
            st.session_state.speaker_path = None
            st.session_state.ref_display_name = None
            
        return True
        
    except Exception as e:
        st.error(f"Failed to delete {filename}: {str(e)}")
        return False

# =============================================================================
# GENERATION METADATA MANAGEMENT
# =============================================================================

def save_generation_metadata(output_wav_path: str, text: str, language: str, 
                           style: str, reference_name: Optional[str],
                           extras: Optional[Dict[str, Any]] = None) -> str:
    """
    Save metadata for a generated audio file.
    
    Args:
        output_wav_path: Path to generated WAV file
        text: Input text used for generation
        language: Language used
        style: Style used
        reference_name: Name of reference voice file
        extras: Additional metadata
        
    Returns:
        str: Path to saved metadata file
    """
    metadata = {
        "id": str(uuid.uuid4()),
        "created": datetime.now().isoformat(),
        "wav": os.path.basename(output_wav_path),
        "text": text[:5000],  # Limit text length
        "language": language,
        "style": style,
        "reference": reference_name,
        "extras": extras or {},
    }
    
    metadata_path = os.path.join(META_DIR, f"{metadata['id']}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return metadata_path


def load_recent_metadata(limit: int = RECENTS_LIMIT) -> List[Dict[str, Any]]:
    """
    Load recent generation metadata files.
    
    Args:
        limit: Maximum number of metadata entries to return
        
    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries, sorted by creation time
    """
    try:
        metadata_files = [
            os.path.join(META_DIR, f) 
            for f in os.listdir(META_DIR) 
            if f.endswith(".json")
        ]
        
        # Sort by modification time (most recent first)
        metadata_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        
        metadata_list = []
        for file_path in metadata_files[:limit]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    metadata_list.append(json.load(f))
            except Exception:
                # Skip corrupted metadata files
                continue
                
        return metadata_list
        
    except FileNotFoundError:
        return []

# =============================================================================
# AUDIO RECORDING COMPONENT
# =============================================================================

def create_audio_recorder_component() -> Optional[str]:
    """
    Create a browser-based audio recording component.
    
    Returns:
        Optional[str]: Base64-encoded audio data URL when recording is complete,
                      None otherwise
                      
    Note:
        This component uses HTML5 MediaRecorder API to capture audio in the browser
        and returns it as a base64-encoded data URL.
    """
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
        /* Pulsing animation for recording indicator */
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
        // Get DOM elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('recording-status');
        const audioEl = document.getElementById('audio-playback');
        
        let mediaRecorder;
        let audioChunks = [];

        // Set iframe height for Streamlit
        function setFrameHeight() {
            if (window.Streamlit) {
                window.Streamlit.setFrameHeight(300);
            }
        }
        setFrameHeight();

        // Start recording
        startBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const reader = new FileReader();
                    
                    reader.onloadend = () => {
                        const base64AudioData = reader.result;
                        audioEl.src = base64AudioData;
                        audioEl.style.display = 'block';
                        statusDiv.innerHTML = '<span style="color:#7ee787">✅ Recording captured</span>';
                        
                        // Send data back to Streamlit
                        if (window.parent) {
                            window.parent.postMessage({
                                isStreamlitMessage: true,
                                type: "streamlit:setComponentValue",
                                value: base64AudioData
                            }, "*");
                        }
                    };
                    reader.readAsDataURL(audioBlob);
                };

                mediaRecorder.start();
                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusDiv.innerHTML = '<span class="pulse">🎤 Recording...</span>';
                
            } catch (error) {
                statusDiv.innerHTML = '<span style="color:#ff6b6b">Error: ' + error.message + '</span>';
                console.error('Recording error:', error);
            }
        });

        // Stop recording
        stopBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                
                // Stop all tracks to release the microphone
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusDiv.innerText = 'Processing...';
            }
        });
        </script>
        """,
        height=320,
    )


def process_recorded_audio(base64_audio_data: str) -> Optional[str]:
    """
    Process base64-encoded audio data and save as WAV file.
    
    Args:
        base64_audio_data: Base64-encoded audio data URL
        
    Returns:
        Optional[str]: Path to saved WAV file, None if processing failed
    """
    try:
        if not base64_audio_data or not isinstance(base64_audio_data, str):
            raise ValueError("Invalid base64 audio data provided")

        # Parse data URL: data:<mime>;base64,<payload>
        if "," in base64_audio_data:
            header, payload = base64_audio_data.split(",", 1)
        else:
            # Assume payload only if no comma found
            payload = base64_audio_data

        # Decode base64 data
        decoded_audio = base64.b64decode(payload)

        # First attempt: try librosa directly
        try:
            audio_buffer = io.BytesIO(decoded_audio)
            audio_data, sample_rate = librosa.load(audio_buffer, sr=None, mono=True)
            
        except Exception:
            # Fallback: use pydub to handle container formats like webm
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(decoded_audio))
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                audio_data, sample_rate = librosa.load(wav_buffer, sr=None, mono=True)
                
            except Exception as e:
                raise RuntimeError(f"Could not decode audio data: {str(e)}")

        # Process audio: trim silence, resample, normalize
        audio_data = trim_silence(audio_data, sample_rate, top_db=28)
        
        if sample_rate != 16000:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )
            sample_rate = 16000

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio_data)) if audio_data.size > 0 else 1.0
        if peak > 0:
            audio_data = 0.98 * audio_data / peak

        # Save processed audio
        timestamp = int(time.time())
        output_path = os.path.join(VOICES_DIR, f"recorded_{timestamp}.wav")
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
        
        return output_path
        
    except Exception as e:
        st.error(f"Error processing recorded audio: {str(e)}")
        return None

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

def create_sidebar():
    """Create and populate the application sidebar with settings and controls."""
    with st.sidebar:
        st.subheader("⚙️ Settings")
        
        # Language selection
        language = st.selectbox(
            "Language",
            options=[
                "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
                "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hi"
            ],
            index=0,
            help="Select the target language for voice synthesis"
        )

        # Style selection
        style = st.radio(
            "Voice style", 
            options=["neutral", "fast", "expressive"], 
            index=0,
            help="Choose a preset voice style"
        )
        
        # Advanced controls
        advanced_mode = st.checkbox("Advanced controls", value=False)
        speed_override = None
        emotion_override = None
        
        if advanced_mode:
            speed_override = st.slider(
                "Speed", 
                min_value=0.8, 
                max_value=1.5, 
                value=1.0, 
                step=0.01,
                help="Adjust speaking speed (0.8 = slower, 1.5 = faster)"
            )
            emotion_override = st.selectbox(
                "Emotion", 
                options=["None", "happy", "sad", "angry", "surprised", "fearful"], 
                index=0,
                help="Add emotional expression to the voice"
            )

        st.markdown("---")
        
        # Voice quality settings
        st.markdown("**Voice Quality Settings**")
        voice_stability = st.slider(
            "Voice stability", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Higher values make voice more stable but less expressive"
        )
        voice_similarity = st.slider(
            "Voice similarity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.75, 
            step=0.1,
            help="Higher values make output more similar to reference voice"
        )

        # Output options
        normalize_output = st.checkbox(
            "Normalize output loudness", 
            value=True,
            help="Normalize audio loudness for consistent volume"
        )
        
        # Device info
        st.markdown(f"**Device:** `{DEVICE}`")

        st.markdown("---")
        
        # Active voice indicator
        st.markdown("**Active Voice**")
        if st.session_state.get("active_voice"):
            active_voice = st.session_state.active_voice
            st.markdown(
                f"<span class='active-badge'>{active_voice}</span>", 
                unsafe_allow_html=True
            )
            if st.button("Clear active voice", use_container_width=True):
                st.session_state.active_voice = None
                st.session_state.speaker_path = None
                st.session_state.ref_display_name = None
                st.success("Active voice cleared")
                st.rerun()
        else:
            st.info("No active voice selected")

        st.markdown("---")
        
        # Model loading button
        if st.button("🚀 Load / Warm up model", use_container_width=True):
            start_time = time.time()
            with st.spinner("Loading XTTS model..."):
                _ = load_tts_model()
            load_time = time.time() - start_time
            st.success(f"Model ready on {DEVICE} in {load_time:.1f}s")

    return {
        "language": language,
        "style": style,
        "speed_override": speed_override,
        "emotion_override": emotion_override,
        "voice_stability": voice_stability,
        "voice_similarity": voice_similarity,
        "normalize_output": normalize_output
    }

# =============================================================================
# VOICE SELECTION INTERFACE
# =============================================================================

def create_voice_selection_interface():
    """Create the voice selection interface with upload and gallery options."""
    st.markdown("### 🎵 Choose Reference Voice")

    voice_option = st.radio(
        "Select input method",
        options=["Upload new voice", "Use voice from gallery"],
        horizontal=True
    )

    if voice_option == "Upload new voice":
        handle_voice_upload()
    else:
        handle_voice_gallery()


def handle_voice_upload():
    """Handle new voice file upload and processing."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload reference voice (20–30s of clean speech recommended)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload a clear recording of the voice you want to clone"
    )
    
    # Audio processing options
    col1, col2 = st.columns(2)
    with col1:
        trim_silence_option = st.checkbox(
            "Trim long silences from reference", 
            value=True,
            help="Remove silent portions from the beginning and end"
        )
    with col2:
        trim_sensitivity = st.slider(
            "Trim sensitivity (dB)", 
            min_value=20, 
            max_value=40, 
            value=28,
            help="Higher values remove more quiet sounds"
        )
    
    if uploaded_file:
        try:
            file_bytes = uploaded_file.read()
            with st.spinner("Preprocessing uploaded voice..."):
                processed_path = preprocess_audio_file(
                    file_bytes, 
                    do_trim=trim_silence_option, 
                    trim_db=trim_sensitivity
                )
                
                reference_name = os.path.basename(processed_path)
                st.success(f"✅ Prepared reference: {reference_name}")
                
                # Display audio player
                st.audio(processed_path)
                
                # Show audio duration info
                audio_data, sample_rate = librosa.load(processed_path, sr=None, mono=True)
                duration = len(audio_data) / sample_rate if sample_rate else 0.0
                
                if duration < 8:
                    st.info("ℹ️ Reference is quite short (<8s). Longer samples often produce better clones.")
                elif duration > 60:
                    st.info("ℹ️ Reference is long (>60s). 20–30s usually works optimally.")
                
                # Update session state
                st.session_state.speaker_path = processed_path
                st.session_state.ref_display_name = reference_name
                st.session_state.active_voice = reference_name
                st.session_state.recording_processed = False
                
        except Exception as e:
            st.error(f"❌ Upload processing failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def handle_voice_gallery():
    """Handle voice selection from the saved gallery."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    existing_voices = get_voice_gallery()
    
    if existing_voices:
        # Voice selection dropdown
        selected_voice = st.selectbox(
            "Choose from saved voices", 
            options=existing_voices,
            key="gallery_select",
            help="Select a previously saved voice from your gallery"
        )
        
        if selected_voice:
            voice_path = os.path.join(VOICES_DIR, selected_voice)
            st.audio(voice_path)
            
            # Update session state but don't set as active automatically
            st.session_state.speaker_path = voice_path
            st.session_state.ref_display_name = selected_voice
            st.session_state.recording_processed = False

        st.markdown("---")
        create_voice_gallery_management(existing_voices)
        
    else:
        st.warning("📁 No saved voices found. Upload or record a voice to get started.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def create_voice_gallery_management(existing_voices: List[str]):
    """Create the voice gallery management interface."""
    st.subheader("🎵 Voice Gallery")
    
    # Batch delete controls
    batch_col1, batch_col2 = st.columns([3, 1])
    with batch_col1:
        st.markdown("**Batch Operations** (optional)")
        selected_for_deletion = st.multiselect(
            "Select voices for batch deletion",
            options=existing_voices,
            key="batch_select",
            help="Select multiple voices to delete at once"
        )
        st.session_state.batch_delete = selected_for_deletion
        
    with batch_col2:
        if st.session_state.batch_delete:
            if not st.session_state.confirm_batch_delete:
                if st.button("🗑️ Delete Selected", use_container_width=True):
                    st.session_state.confirm_batch_delete = True
                    st.rerun()
            else:
                st.warning(f"⚠️ Delete {len(st.session_state.batch_delete)} files?")
                if st.button("✅ Confirm Delete", use_container_width=True):
                    deleted_count = 0
                    for filename in list(st.session_state.batch_delete):
                        if delete_voice_file(filename):
                            deleted_count += 1
                    
                    st.session_state.batch_delete = []
                    st.session_state.confirm_batch_delete = False
                    st.success(f"✅ Deleted {deleted_count} files")
                    st.rerun()
                    
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_batch_delete = False
                    st.rerun()
        else:
            st.info("No selections")

    # Individual voice management
    st.markdown("**Individual Voice Controls**")
    for voice_filename in existing_voices:
        voice_path = os.path.join(VOICES_DIR, voice_filename)
        is_active = voice_filename == st.session_state.get("active_voice")
        
        # Voice name with active indicator
        if is_active:
            st.markdown(f"**✅ {voice_filename} (Active)**")
        else:
            st.markdown(f"🎵 {voice_filename}")
        
        # Controls layout
        control_cols = st.columns([3, 1, 1])
        
        with control_cols[0]:
            st.audio(voice_path)
            
        with control_cols[1]:
            if not is_active:
                if st.button("🎯 Set Active", key=f"activate_{voice_filename}"):
                    st.session_state.speaker_path = voice_path
                    st.session_state.ref_display_name = voice_filename
                    st.session_state.active_voice = voice_filename
                    st.success(f"✅ Activated {voice_filename}")
                    st.rerun()
            else:
                st.success("✅ Active")
                
        with control_cols[2]:
            # Delete confirmation flow
            if st.session_state.pending_delete == voice_filename:
                confirm_col1, confirm_col2 = st.columns([1, 1])
                with confirm_col1:
                    if st.button("✅", key=f"confirm_del_{voice_filename}"):
                        if delete_voice_file(voice_filename):
                            st.success(f"Deleted {voice_filename}")
                        st.session_state.pending_delete = None
                        st.rerun()
                with confirm_col2:
                    if st.button("❌", key=f"cancel_del_{voice_filename}"):
                        st.session_state.pending_delete = None
                        st.rerun()
            else:
                if st.button("🗑️", key=f"delete_{voice_filename}"):
                    st.session_state.pending_delete = voice_filename
                    st.rerun()
        
        st.markdown("---")

# =============================================================================
# SPEECH GENERATION INTERFACE
# =============================================================================

def create_generation_interface(settings: Dict[str, Any]):
    """Create the speech generation interface."""
    st.markdown("### 🎙️ Generate Speech")

    # Generation controls
    control_col1, control_col2 = st.columns([1, 1])
    
    with control_col1:
        generation_disabled = (
            not st.session_state.get("speaker_path") or 
            not st.text_area_value.strip()
        )
        
        generate_clicked = st.button(
            "🎬 Generate Voice", 
            type="primary",
            use_container_width=True,
            disabled=generation_disabled,
            help="Generate speech using the selected reference voice"
        )
        
    with control_col2:
        if st.button("♻️ Reset Interface", use_container_width=True):
            # Reset relevant session state
            reset_keys = [
                "speaker_path", "ref_display_name", "recorded_b64",
                "recording_processed", "active_voice"
            ]
            for key in reset_keys:
                st.session_state[key] = None
            st.rerun()

    # Handle generation
    if generate_clicked and st.session_state.speaker_path:
        handle_speech_generation(settings)


def handle_speech_generation(settings: Dict[str, Any]):
    """Handle the speech generation process."""
    start_time = time.time()
    
    with st.spinner("🎭 Synthesizing speech..."):
        try:
            # Load TTS model
            tts_model = load_tts_model()
            
            # Generate speech
            output_wav = synthesize_speech(
                tts_model=tts_model,
                text=st.text_area_value.strip(),
                speaker_wav_path=st.session_state.speaker_path,
                language=settings["language"],
                style=settings["style"],
                speed_override=settings["speed_override"],
                emotion_override=settings["emotion_override"]
            )
            
            # Apply loudness normalization if requested
            final_wav = output_wav
            if settings["normalize_output"]:
                try:
                    final_wav = normalize_audio_loudness(output_wav)
                except Exception:
                    st.warning("⚠️ Loudness normalization failed, using original audio")
                    final_wav = output_wav

            # Save generation metadata
            metadata_extras = {
                "device": DEVICE,
                "normalized": settings["normalize_output"],
                "speed": settings["speed_override"],
                "emotion": settings["emotion_override"],
                "stability": settings["voice_stability"],
                "similarity": settings["voice_similarity"]
            }
            
            save_generation_metadata(
                output_wav_path=final_wav,
                text=st.text_area_value.strip(),
                language=settings["language"],
                style=settings["style"],
                reference_name=st.session_state.ref_display_name,
                extras=metadata_extras
            )

            generation_time = time.time() - start_time
            st.success(f"✅ Generation completed in {generation_time:.1f} seconds")
            
            # Display results
            display_generation_results(final_wav)
            
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")


def display_generation_results(wav_path: str):
    """Display the generated audio results with download options."""
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("**🎵 WAV Audio**")
        st.audio(wav_path, format="audio/wav")
        
        with open(wav_path, "rb") as wav_file:
            st.download_button(
                label="⬇️ Download WAV",
                data=wav_file.read(),
                file_name=os.path.basename(wav_path),
                mime="audio/wav",
                use_container_width=True
            )
    
    with result_col2:
        st.markdown("**🎵 MP3 Audio**")
        try:
            mp3_path = convert_wav_to_mp3(wav_path)
            st.audio(mp3_path, format="audio/mpeg")
            
            with open(mp3_path, "rb") as mp3_file:
                st.download_button(
                    label="⬇️ Download MP3",
                    data=mp3_file.read(),
                    file_name=os.path.basename(mp3_path),
                    mime="audio/mpeg",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"⚠️ MP3 conversion unavailable: {str(e)}")

# =============================================================================
# RECENT GENERATIONS GALLERY
# =============================================================================

def create_recent_generations_gallery():
    """Create the recent generations gallery interface."""
    st.markdown("### 📁 Recent Generations")
    
    recent_metadata = load_recent_metadata(RECENTS_LIMIT)
    
    if not recent_metadata:
        st.info("🎭 No generations yet. Create your first voice clone to see it here!")
        return
    
    for metadata in recent_metadata:
        create_generation_entry(metadata)


def create_generation_entry(metadata: Dict[str, Any]):
    """Create a single generation entry in the gallery."""
    # Format creation time
    created_time = metadata.get('created', 'Unknown')
    language = metadata.get('language', 'Unknown')
    style = metadata.get('style', 'Unknown')
    reference = metadata.get('reference', 'Unknown')
    
    # Create expandable entry
    with st.expander(
        f"🕒 {created_time} • {language} • {style} • ref: {reference}"
    ):
        wav_filename = metadata.get("wav", "")
        wav_path = os.path.join(OUTPUTS_DIR, wav_filename)
        
        entry_cols = st.columns([2, 1])
        
        with entry_cols[0]:
            st.markdown("**Audio Preview**")
            if os.path.exists(wav_path):
                st.audio(wav_path)
                
                # Display generation text (truncated)
                generation_text = metadata.get("text", "")
                if generation_text:
                    display_text = (
                        generation_text[:200] + "..." 
                        if len(generation_text) > 200 
                        else generation_text
                    )
                    st.markdown(f"**Text:** {display_text}")
            else:
                st.error("⚠️ Original audio file not found")
        
        with entry_cols[1]:
            # Download button
            if os.path.exists(wav_path):
                with open(wav_path, "rb") as audio_file:
                    st.download_button(
                        label="⬇️ Download WAV",
                        data=audio_file.read(),
                        file_name=wav_filename,
                        mime="audio/wav",
                        use_container_width=True
                    )
            
            # Re-synthesis button
            if st.button(
                "🔁 Re-synthesize", 
                key=f"resynth_{metadata.get('id', 'unknown')}",
                help="Generate again with the same settings"
            ):
                handle_resynthesis(metadata)


def handle_resynthesis(metadata: Dict[str, Any]):
    """Handle re-synthesis of a previous generation."""
    try:
        reference_name = metadata.get("reference")
        if not reference_name:
            st.error("❌ No reference voice information found")
            return
            
        reference_path = os.path.join(VOICES_DIR, reference_name)
        if not os.path.exists(reference_path):
            st.error(f"❌ Reference voice '{reference_name}' not found in gallery")
            return
        
        # Load TTS model and re-synthesize
        tts_model = load_tts_model()
        extras = metadata.get("extras", {})
        
        new_output = synthesize_speech(
            tts_model=tts_model,
            text=metadata.get("text", ""),
            speaker_wav_path=reference_path,
            language=metadata.get("language", "en"),
            style=metadata.get("style", "neutral"),
            speed_override=extras.get("speed"),
            emotion_override=extras.get("emotion")
        )
        
        # Apply normalization if it was used originally
        final_output = new_output
        if extras.get("normalized", False):
            try:
                final_output = normalize_audio_loudness(new_output)
            except Exception:
                final_output = new_output
        
        # Save new metadata
        save_generation_metadata(
            output_wav_path=final_output,
            text=metadata.get("text", ""),
            language=metadata.get("language", "en"),
            style=metadata.get("style", "neutral"),
            reference_name=reference_name,
            extras=extras
        )
        
        st.success("✅ Re-synthesis completed!")
        st.audio(final_output)
        
    except Exception as e:
        st.error(f"❌ Re-synthesis failed: {str(e)}")

# =============================================================================
# HELP AND TIPS SECTION
# =============================================================================

def create_tips_section():
    """Create the tips and help section."""
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **🎯 For Natural-Sounding Voice Clones:**

        **1. Reference Audio Quality**
        - Use 20-30 seconds of clean, single-speaker audio
        - Record in a quiet environment with minimal background noise
        - Ensure consistent volume and clear pronunciation
        - Use a quality microphone if possible

        **2. Text Input Best Practices**
        - Use proper punctuation (periods, commas, question marks)
        - Break very long sentences into shorter, natural phrases
        - Use contractions and natural language ("don't" vs "do not")
        - Match the style of your reference audio (formal vs casual)

        **3. Optimal Settings**
        - **Neutral style** works best for most use cases
        - Adjust **speed** between 0.9-1.1 for natural pacing
        - Use **emotions** sparingly and match your reference tone
        - Increase **voice stability** for consistent output
        - Increase **voice similarity** to match reference more closely

        **4. Reference Speech Guidelines**
        - Use reference audio with similar emotion to desired output
        - Match speaking style (conversational, narrative, etc.)
        - Ensure clear articulation and consistent pacing
        - Avoid background music or multiple speakers

        **5. Browser Recording Tips**
        - Chrome and Firefox work best for recording
        - Allow microphone permissions when prompted  
        - Speak at consistent distance from microphone
        - Record in quiet environment without echo
        - Test your microphone before important recordings

        **6. Troubleshooting**
        - If output sounds robotic, try a longer reference sample
        - If voice doesn't match, increase voice similarity setting
        - For inconsistent results, increase voice stability
        - MP3 export requires FFmpeg installation
        """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    # Application header
    st.title("🗣️ AI Voice Cloner")
    st.caption(
        "Powered by Coqui XTTS v2 • Upload 20-30 seconds of clean reference speech for best results"
    )

    # Text input section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Text to Synthesize")
    
    # Use a global text area that persists across reruns
    if "text_input" not in st.session_state:
        st.session_state.text_input = (
            "Hello! This is my cloned voice running from a Streamlit web application. "
            "The voice should sound natural and expressive with proper intonation."
        )
    
    text_input = st.text_area(
        label="Enter the text you want to convert to speech",
        value=st.session_state.text_input,
        height=120,
        label_visibility="collapsed",
        help="Enter clear, well-punctuated text for best results. Avoid extremely long sentences.",
        key="text_area"
    )
    
    # Store text area value globally for access in other functions
    st.text_area_value = text_input
    st.session_state.text_input = text_input
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Get sidebar settings
    settings = create_sidebar()

    # Voice selection interface
    create_voice_selection_interface()
    st.markdown("---")

    # Speech generation interface
    create_generation_interface(settings)
    st.markdown("---")

    # Recent generations gallery
    create_recent_generations_gallery()
    st.markdown("---")

    # Tips and help
    create_tips_section()


# Run the application
if __name__ == "__main__":
    main()
