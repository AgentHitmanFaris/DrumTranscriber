# Drum Transcriber

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentHitmanFaris/DrumTranscriber/blob/main/DrumTranscriber_Colab.ipynb)

**A production-grade drum transcription system employing deep learning to classify drum hits into six distinct classes.**

This application features a modern, interactive web interface for visualize drum transcriptions in real-time, offering a seamless experience for musicians, producers, and developers.

## Live Demo

Experience the full capabilities of Drum Transcriber instantly via Google Colab. No local installation required.

[**Launch Interactive Demo**](https://colab.research.google.com/github/AgentHitmanFaris/DrumTranscriber/blob/main/DrumTranscriber_Colab.ipynb)

## Key Features

-   **High-Fidelity Transcription**: Detects Kick, Snare, Hi-Hat (Closed), Tom (High), Ride, and Crash with high accuracy.
-   **Interactive Piano Roll**: HTML5/Canvas-based visualization synchronized with audio playback.
-   **Precision Controls**: Seek, play, and pause directly on the timeline.
-   **Data Export**: Download transcription results as structured CSV data.
-   **YouTube Integration**: Built-in support for direct audio extraction from YouTube.

## Technology Stack

Built with a robust stack of open-source technologies:

-   **Core**: Python 3.10+
-   **Machine Learning**: TensorFlow / Keras (CNN Architecture)
-   **Signal Processing**: Librosa, SoundFile
-   **Web Framework**: Gradio (Reactive UI)
-   **Visualization**: Custom HTML5 Canvas & Plotly

## Quick Start (Portability Mode)

For Windows users requiring a zero-setup environment:

1.  Clone this repository.
2.  Execute **`run_portable.bat`**.
3.  The application handles all dependency management and launches the local server automatically.

## Manual Installation

For developers integrating this into existing workflows:

### Prerequisites
-   Python 3.10 or higher
-   FFmpeg installed and available in system PATH

### Installation
1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Setup**
    Download the pre-trained weights (`drum_transcriber.h5`) and place them in the `model/` directory.
    [**Download Weights**](https://drive.google.com/file/d/1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi/view?usp=sharing)

3.  **Launch**
    ```bash
    python gradio_app.py
    ```

## Python API

Integrate the transcription engine directly into your Python scripts:

```python
from DrumTranscriber import DrumTranscriber
import librosa

# Load Audio
audio_path = "path/to/audio.wav"
samples, sr = librosa.load(audio_path, sr=44100)

# Initialize Engine
transcriber = DrumTranscriber()

# Predict
# Returns a Pandas DataFrame with timestamps and confidence scores
predictions = transcriber.predict(samples, sr)

print(predictions.head())
```

---
*v1.0.0 - Production Release*
