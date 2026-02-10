# Drum Transcriber - Transcribe Drum Audio Clips

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentHitmanFaris/DrumTranscriber/blob/main/DrumTranscriber_Colab.ipynb)

This package helps users transcribes drum audio hits into 6 classes - Hihat, Crash, Kick Drum, Snare, Ride, and Toms.

![demo](https://github.com/AgentHitmanFaris/DrumTranscriber/blob/main/assets/demo.gif?raw=true)

## Quick Start (Google Colab)

The easiest way to use this is with the Google Colab notebook. Click the "Open In Colab" badge above to get started.

## Dependencies

Run the following to install the python dependencies:

```
pip install librosa tensorflow numpy pandas scikit-learn streamlit streamlit-player gradio yt-dlp plotly
```

## Usage

### Using Gradio (Recommended)

Run the **Gradio** app for a better UI and YouTube support:

```bash
python gradio_app.py
```

### Basic Usage (Python)
```Python
from DrumTranscriber import DrumTranscriber
import librosa

samples, sr = librosa.load(PATH/TO/AUDIO/CLIP)

transcriber = DrumTranscriber()

# pandas dataframe containing probabilities of classes
predictions = transcriber.predict(samples, sr)
```

### For Streamlit (Legacy)

cd to the parent directory and run the following command:
```
streamlit run frontend.py
```
A localhost website will appear with the demo app.


## Getting Started Locally

1. Clone/Zip the directory
2. Redownload the model .h5 file from `/model/drum_transcriber.h5` (https://drive.google.com/file/d/1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi/view?usp=sharing)

**Note**: There is an issue with Github zipping the .h5 model file. To properly get the model to work, I suggest downloading the model file from the Google Drive above and directly to replace the model from the clone/zipped folder. 
