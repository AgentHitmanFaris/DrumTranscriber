# Drum Transcriber - Transcribe Drum Audio Clips

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentHitmanFaris/DrumTranscriber/blob/main/DrumTranscriber_Colab.ipynb)

This package helps users transcribes drum audio hits into 6 classes - Hihat, Crash, Kick Drum, Snare, Ride, and Toms.

![demo](https://github.com/AgentHitmanFaris/DrumTranscriber/blob/main/assets/demo.gif?raw=true)

## Quick Start (Windows Portable App)

**Use the portable app for the easiest experience (no installation required):**

1.  Clone or download this repository.
2.  Double-click **`run_portable.bat`**.
    -   This script will automatically set up a local Python environment, download necessary dependencies (including FFmpeg and the model), and launch the app.
3.  The app will open in your default browser.

### Features
-   **Interactive Player**: View drum hits on a piano roll synced with audio.
-   **Playback Control**: Play, pause, and seek by clicking on the timeline.
-   **Volume Control**: Adjust playback volume directly in the player.
-   **Download Predictions**: Export the transcribed drum hits as a CSV file.

## Quick Start (Google Colab)

If you prefer running in the cloud, click the "Open In Colab" badge above to use the Google Colab notebook.

## Dependencies (Manual Setup)

If you are **not** using the portable script and want to set up the environment manually:

1.  Install Python 3.10+.
2.  Install FFmpeg and add it to your system PATH.
3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Download the model:
    -   Ensure `model/drum_transcriber.h5` exists. You can download it [here](https://drive.google.com/file/d/1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi/view?usp=sharing).

## Usage

### Run Gradio App (Recommended)

```bash
python gradio_app.py
```

### Basic Usage (Script)
```Python
from DrumTranscriber import DrumTranscriber
import librosa

samples, sr = librosa.load("PATH/TO/AUDIO.wav")

transcriber = DrumTranscriber()

# pandas dataframe containing probabilities of classes
predictions = transcriber.predict(samples, sr)
print(predictions)
```
