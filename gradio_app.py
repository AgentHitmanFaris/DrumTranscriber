
import gradio as gr
import yt_dlp
import os
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from DrumTranscriber import DrumTranscriber
from utils.config import SETTINGS

# Initialize transcriber globally
transcriber = None

def load_model():
    global transcriber
    if transcriber is not None:
        return transcriber
        
    try:
        print("Loading model...")
        transcriber = DrumTranscriber()
        print("Model loaded successfully.")
        return transcriber
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'model/drum_transcriber.h5' exists. If on Colab, check the download step.")
        return None

# Try loading initially (optional, but good if model already exists)
load_model()

def download_audio(url, progress=gr.Progress()):
    progress(0, desc="Starting download...")
    
    def progress_hook(d):
        if d['status'] == 'downloading':
            try:
                p = d.get('_percent_str', '0%').replace('%','')
                progress(float(p)/100, desc=f"Downloading: {d.get('_percent_str', '')}")
            except:
                pass
        elif d['status'] == 'finished':
            progress(1.0, desc="Download complete, converting...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress_hook]
    }
    
    # Remove existing temp file if it exists
    if os.path.exists('temp_audio.wav'):
        os.remove('temp_audio.wav')
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return 'temp_audio.wav'
        except Exception as e:
            return None, f"Error downloading video: {e}"

def process_audio(audio_file, start_time, duration=30, progress=gr.Progress()):
    if not audio_file:
        return None, None, None, "No audio file provided."
    
    try:
        progress(0.1, desc="Loading Audio...")
        # Load audio
        samples, sr = librosa.load(audio_file, sr=44100, offset=start_time, duration=duration)
    except Exception as e:
        return None, None, None, f"Error loading audio: {e}"

    # Ensure model is loaded
    model = load_model()
    if model is None:
        return None, None, None, "Transcriber model not loaded. Please ensure 'model/drum_transcriber.h5' exists."

    # Predict
    try:
        progress(0.4, desc="Transcribing (this may take a moment)...")
        preds = model.predict(samples, sr)
    except Exception as e:
        return None, None, None, f"Error during prediction: {e}"

    # Process predictions
    progress(0.8, desc="Processing Results...")
    top_indices = np.argmax(preds[list(SETTINGS['LABELS_INDEX'].values())].to_numpy(), axis=1)
    labelled_preds = [SETTINGS['LABELS_INDEX'][i] for i in top_indices]
    
    preds['prediction'] = labelled_preds
    preds['confidence'] = preds.apply(lambda x: x[x['prediction']], axis=1)
    
    return samples, sr, preds, None

def create_plot(preds, duration):
    # Create a "Drum Roll" view (Piano Roll style)
    label_order = ['kick_drum', 'snare', 'tom_h', 'hihat_c', 'ride', 'crash']
    
    fig = go.Figure()

    for label in label_order:
        subset = preds[preds['prediction'] == label]
        if subset.empty:
            continue
            
        fig.add_trace(go.Scatter(
            x=subset['time'],
            y=[label] * len(subset),
            mode='markers',
            name=label,
            marker=dict(
                size=12,
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=subset['confidence'].apply(lambda x: f"Conf: {x:.2%}"),
            hoverinfo='x+y+text'
        ))

    fig.update_layout(
        title="Drum Roll View",
        xaxis_title="Time (s)",
        yaxis_title="Drum Component",
        yaxis={'categoryorder': 'array', 'categoryarray': label_order},
        hovermode="closest",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def run_pipeline(url, file_upload, start_time, progress=gr.Progress()):
    audio_path = None
    status_msg = ""
    
    if url:
        status_msg += "Downloading from YouTube... "
        # progress object is passed automatically by Gradio if argument is named 'progress' 
        # but we can also pass it explicitly if we want to be sure, though Gradio handles the context.
        # We'll pass it to verify.
        audio_path_result = download_audio(url, progress)
        
        if isinstance(audio_path_result, tuple): 
            # It returned (None, error_message)
            return None, None, None, audio_path_result[1]
        
        audio_path = audio_path_result
        
    elif file_upload:
        audio_path = file_upload
    else:
        return None, None, None, "Please provide a YouTube URL or upload an audio file."

    status_msg += "Processing Audio... "
    samples, sr, preds, error = process_audio(audio_path, start_time, duration=30, progress=progress)
    
    if error:
        return None, None, None, error

    progress(0.9, desc="Generating Plot...")
    fig = create_plot(preds, 30)
    
    csv_path = "predictions.csv"
    preds.to_csv(csv_path, index=False)
    
    progress(1.0, desc="Done!")
    return (sr, samples), fig, csv_path, "Done!"


# Gradio UI
with gr.Blocks(title="Drum Transcriber", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Drum Transcriber")
    gr.Markdown("Transcribe drum hits from audio to MIDI-like data.")
    
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            file_input = gr.Audio(label="Or Upload Audio File", type="filepath")
            start_time = gr.Number(label="Start Time (seconds)", value=0, precision=1)
            btn = gr.Button("Transcribe", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            audio_out = gr.Audio(label="Processed Audio Segment")
            
    with gr.Row():
        plot_out = gr.Plot(label="Drum Roll View")
        
    with gr.Row():
        csv_out = gr.File(label="Download Predictions CSV")

    btn.click(fn=run_pipeline, 
              inputs=[url_input, file_input, start_time], 
              outputs=[audio_out, plot_out, csv_out, status])

if __name__ == "__main__":
    demo.launch(share=True)
