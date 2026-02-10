
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
try:
    transcriber = DrumTranscriber()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'model/drum_transcriber.h5' exists. If on Colab, check the download step.")
    transcriber = None

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'temp_audio',
        'quiet': True,
        'no_warnings': True,
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

def process_audio(audio_file, start_time, duration=30):
    if not audio_file:
        return None, None, None, "No audio file provided."
    
    # If audio_file is a tuple (sr, data) from Gradio mic/upload, we'd need to save it or handle it.
    # But for now assuming file path.
    
    try:
        # Load audio
        # librosa.load can take a file path
        samples, sr = librosa.load(audio_file, sr=44100, offset=start_time, duration=duration)
    except Exception as e:
        return None, None, None, f"Error loading audio: {e}"

    if transcriber is None:
        return None, None, None, "Transcriber model not loaded."

    # Predict
    try:
        preds = transcriber.predict(samples, sr)
    except Exception as e:
        return None, None, None, f"Error during prediction: {e}"

    # Process predictions
    top_indices = np.argmax(preds[list(SETTINGS['LABELS_INDEX'].values())].to_numpy(), axis=1)
    labelled_preds = [SETTINGS['LABELS_INDEX'][i] for i in top_indices]
    
    preds['prediction'] = labelled_preds
    preds['confidence'] = preds.apply(lambda x: x[x['prediction']], axis=1) # Get confidence of the predicted class
    
    # Filter for hits (optional: could filter by confidence threshold if needed, but showing all for now)
    # The original code just returns everything. Let's keep it consistent but maybe cleaner.
    
    return samples, sr, preds, None

def create_plot(preds, duration):
    # Create a "Drum Roll" view (Piano Roll style)
    
    # Map labels to Y-axis positions for better ordering
    # Kick usually at bottom, Cymbals at top
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

def run_pipeline(url, file_upload, start_time):
    # Determine input source
    audio_path = None
    status_msg = ""
    
    if url:
        status_msg += "Downloading from YouTube... "
        audio_path = download_audio(url)
        if isinstance(audio_path, tuple): # Error tuple
            return None, None, None, audio_path[1]
    elif file_upload:
        audio_path = file_upload
    else:
        return None, None, None, "Please provide a YouTube URL or upload an audio file."

    status_msg += "Processing Audio... "
    samples, sr, preds, error = process_audio(audio_path, start_time)
    
    if error:
        return None, None, None, error

    # Save processed audio segment to temp file for playback if needed, 
    # but Gradio can take numpy array for audio output: (sr, samples)
    
    # Visualization
    fig = create_plot(preds, 30) # Assuming 30s max duration as per input
    
    # CSV
    csv_path = "predictions.csv"
    preds.to_csv(csv_path, index=False)
    
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
