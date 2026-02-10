# Drum Transcriber

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentHitmanFaris/DrumTranscriber/blob/main/DrumTranscriber_Colab.ipynb)

Automatic drum transcription system capable of detecting and classifying drum hits into six distinct classes: **Kick, Snare, Hi-Hat (Closed), Tom (High), Ride, and Crash**. The application features an interactive web interface with a synchronized audio player and piano roll visualization.

## Quick Start (Windows Portable)

For the easiest experience on Windows without manual installation:

1.  Clone or download this repository.
2.  Run the **`run_portable.bat`** script.
3.  The application will automatically configure a local Python environment, download dependencies, and launch in your default web browser.

## Cloud Deployment

This project is optimized for Google Colab. Click the badge above to launch the notebook, which handles environment setup and model downloading automatically.

## Manual Installation

If you prefer to configure the environment manually, ensure you have Python 3.10+ and FFmpeg installed.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model
Download the pre-trained model (`drum_transcriber.h5`) and place it in the `model/` directory.
[Download Link](https://drive.google.com/file/d/1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi/view?usp=sharing)

### 3. Run Application
```bash
python gradio_app.py
```

## Features

-   **Multi-Class Transcription**: Accurately identifies 6 different drum components.
-   **Interactive Visualization**: Web-based piano roll with real-time audio synchronization.
-   **Playback Control**: Seek, play, and pause directly from the visual interface.
-   **CSV Export**: Download transcription data for use in DAWs or other analysis tools.
-   **YouTube Integration**: helper to download and transcribe audio directly from YouTube URLs.

## Python API Usage

To use the transcriber programmatically:

```python
from DrumTranscriber import DrumTranscriber
import librosa

# Load audio
samples, sr = librosa.load("path/to/audio.wav")

# Initialize and predict
transcriber = DrumTranscriber()
predictions = transcriber.predict(samples, sr)

print(predictions.head())
```
