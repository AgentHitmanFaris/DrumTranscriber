
import sys
import os


# Ensure current directory is in sys.path so we can import local modules

# This is needed for the portable embedded python version
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


import gradio as gr
import yt_dlp
import librosa
import numpy as np
import pandas as pd
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

    
    # Explicitly set ffmpeg location for portable usage
    base_dir = os.path.dirname(__file__)
    ffmpeg_dirs = ['ffmpeg', 'bin']
    ffmpeg_path = None
    
    for relative_dir in ffmpeg_dirs:
        possible_path = os.path.join(base_dir, relative_dir)
        if os.path.exists(os.path.join(possible_path, 'ffmpeg.exe')):
            ffmpeg_path = possible_path
            break
            
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
    
    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = ffmpeg_path
    
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

def create_interactive_player(preds, samples, sr):
    """Create an interactive HTML piano roll with synced audio playback and playhead.
    
    Uses an iframe with srcdoc to guarantee JavaScript execution, since
    Gradio's gr.HTML component does not execute <script> tags directly.
    """
    import base64
    import io
    import html as html_module
    import soundfile as sf
    import json
    
    # Encode audio to base64 WAV
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
    audio_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Prepare drum hit data as JSON
    label_order = ['crash', 'ride', 'hihat_c', 'tom_h', 'snare', 'kick_drum']
    label_display = {
        'kick_drum': 'Kick', 'snare': 'Snare', 'tom_h': 'Tom',
        'hihat_c': 'Hi-Hat', 'ride': 'Ride', 'crash': 'Crash'
    }
    label_colors = {
        'kick_drum': '#ff4444', 'snare': '#44aaff', 'tom_h': '#ff8800',
        'hihat_c': '#ffdd00', 'ride': '#44ff88', 'crash': '#ff44ff'
    }
    
    duration = len(samples) / sr
    
    hits_json = []
    for _, row in preds.iterrows():
        hits_json.append({
            'time': float(row['time']),
            'label': row['prediction'],
            'confidence': float(row.get('confidence', 0.5))
        })
    
    hits_str = json.dumps(hits_json)
    labels_str = json.dumps(label_order)
    display_str = json.dumps(label_display)
    colors_str = json.dumps(label_colors)
    
    # Build a self-contained HTML document for the iframe
    inner_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
  #controls {{ display:flex; align-items:center; gap:12px; padding:12px 16px; }}
  #playBtn {{
    background:#ff4444; color:white; border:none; border-radius:50%;
    width:44px; height:44px; font-size:18px; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
  }}
  #playBtn:hover {{ background:#ff6666; }}
  #timeDisplay {{ color:#ccc; font-size:14px; font-variant-numeric:tabular-nums; }}
  #volumeSlider {{ width:80px; accent-color:#ff4444; }}
  #pianoRoll {{ display:block; width:100%; cursor:crosshair; }}
</style>
</head>
<body>
  <div id="controls">
    <button id="playBtn">&#9654;</button>
    <span id="timeDisplay">0:00.0 / {duration:.1f}s</span>
    <input id="volumeSlider" type="range" min="0" max="100" value="80" title="Volume">
  </div>
  <canvas id="pianoRoll"></canvas>
  <audio id="drumAudio" src="data:audio/wav;base64,{audio_b64}"></audio>
<script>
(function() {{
  const hits = {hits_str};
  const labelOrder = {labels_str};
  const labelDisplay = {display_str};
  const labelColors = {colors_str};
  const duration = {duration};

  const canvas = document.getElementById('pianoRoll');
  const ctx = canvas.getContext('2d');
  const audio = document.getElementById('drumAudio');
  const playBtn = document.getElementById('playBtn');
  const timeDisp = document.getElementById('timeDisplay');
  const volSlider = document.getElementById('volumeSlider');

  const dpr = window.devicePixelRatio || 1;
  const LANE_H = 40;
  const TOP_PAD = 10;
  const BOTTOM_PAD = 25;
  const LEFT_PAD = 60;
  const NUM_LANES = labelOrder.length;
  const CANVAS_H = TOP_PAD + NUM_LANES * LANE_H + BOTTOM_PAD;

  let lastW = 0;
  function resize() {{
    const w = document.body.clientWidth;
    if (w === lastW) return;
    lastW = w;
    canvas.style.width = w + 'px';
    canvas.style.height = CANVAS_H + 'px';
    canvas.width = w * dpr;
    canvas.height = CANVAS_H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}

  resize();
  window.addEventListener('resize', resize);

  volSlider.addEventListener('input', () => {{ audio.volume = volSlider.value / 100; }});
  audio.volume = 0.8;

  playBtn.addEventListener('click', () => {{
    if (audio.paused) {{
      audio.play();
      playBtn.innerHTML = '&#9646;&#9646;';
    }} else {{
      audio.pause();
      playBtn.innerHTML = '&#9654;';
    }}
  }});

  audio.addEventListener('ended', () => {{ playBtn.innerHTML = '&#9654;'; }});

  canvas.addEventListener('click', (e) => {{
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const w = rect.width;
    const t = ((x - LEFT_PAD) / (w - LEFT_PAD - 10)) * duration;
    if (t >= 0 && t <= duration) {{
      audio.currentTime = t;
    }}
  }});

  function draw() {{
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const rollW = w - LEFT_PAD - 10;

    ctx.clearRect(0, 0, w, h);

    // Draw lanes
    for (let i = 0; i < NUM_LANES; i++) {{
      const y = TOP_PAD + i * LANE_H;
      ctx.fillStyle = i % 2 === 0 ? '#16213e' : '#1a1a2e';
      ctx.fillRect(LEFT_PAD, y, rollW, LANE_H);

      ctx.strokeStyle = '#2a2a4a';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(LEFT_PAD, y + LANE_H);
      ctx.lineTo(w - 10, y + LANE_H);
      ctx.stroke();

      ctx.fillStyle = labelColors[labelOrder[i]] || '#ccc';
      ctx.font = 'bold 12px Segoe UI, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelDisplay[labelOrder[i]] || labelOrder[i], LEFT_PAD - 8, y + LANE_H / 2);
    }}

    // Time markers
    const totalH = NUM_LANES * LANE_H;
    const step = duration > 20 ? 5 : duration > 10 ? 2 : 1;
    ctx.font = '10px Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    for (let t = 0; t <= duration; t += step) {{
      const x = LEFT_PAD + (t / duration) * rollW;
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, TOP_PAD);
      ctx.lineTo(x, TOP_PAD + totalH);
      ctx.stroke();
      ctx.fillStyle = '#777';
      ctx.fillText(t + 's', x, TOP_PAD + totalH + 14);
    }}

    // Draw hits
    for (const hit of hits) {{
      const laneIdx = labelOrder.indexOf(hit.label);
      if (laneIdx < 0) continue;
      const x = LEFT_PAD + (hit.time / duration) * rollW;
      const y = TOP_PAD + laneIdx * LANE_H + LANE_H / 2;
      const r = 4 + hit.confidence * 6;
      const color = labelColors[hit.label] || '#fff';

      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.25 + hit.confidence * 0.5;
      ctx.fillRect(x - 2, TOP_PAD + laneIdx * LANE_H + 4, 4, LANE_H - 8);
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }}

    // Playhead
    const ct = audio.currentTime;
    if (ct > 0 || !audio.paused) {{
      const px = LEFT_PAD + (ct / duration) * rollW;
      ctx.strokeStyle = '#ff4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, TOP_PAD);
      ctx.lineTo(px, TOP_PAD + totalH);
      ctx.stroke();
      ctx.fillStyle = '#ff4444';
      ctx.beginPath();
      ctx.moveTo(px - 6, TOP_PAD);
      ctx.lineTo(px + 6, TOP_PAD);
      ctx.lineTo(px, TOP_PAD + 8);
      ctx.closePath();
      ctx.fill();
    }}

    // Time display
    const mins = Math.floor(ct / 60);
    const secs = (ct % 60).toFixed(1);
    const pad = parseFloat(secs) < 10 ? '0' : '';
    timeDisp.textContent = mins + ':' + pad + secs + ' / ' + duration.toFixed(1) + 's';

    requestAnimationFrame(draw);
  }}

  requestAnimationFrame(draw);
}})();
</script>
</body>
</html>"""

    # Escape for srcdoc attribute — browser unescapes entities before parsing
    srcdoc_escaped = html_module.escape(inner_html, quote=True)
    iframe_height = CANVAS_H + 70  # canvas + controls
    
    return f'<iframe srcdoc="{srcdoc_escaped}" style="width:100%; height:{iframe_height}px; border:none; border-radius:12px; background:#1a1a2e;" allowfullscreen></iframe>'


# Pre-calculate constant for iframe height
CANVAS_H = 10 + 6 * 40 + 25  # TOP_PAD + NUM_LANES * LANE_H + BOTTOM_PAD

def run_pipeline(url, file_upload, start_time, progress=gr.Progress()):
    audio_path = None
    status_msg = ""
    
    if url:
        status_msg += "Downloading from YouTube... "
        audio_path_result = download_audio(url, progress)
        
        if isinstance(audio_path_result, tuple): 
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

    progress(0.9, desc="Generating Piano Roll...")
    player_html = create_interactive_player(preds, samples, sr)
    
    csv_path = "predictions.csv"
    preds.to_csv(csv_path, index=False)
    
    progress(1.0, desc="Done!")
    return player_html, csv_path, None, "Done!"


# Gradio UI
with gr.Blocks(title="Drum Transcriber") as demo:
    gr.Markdown("# 🥁 Drum Transcriber")
    gr.Markdown("Transcribe drum hits from audio. Play the result and watch the playhead move in real-time.")
    
    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            file_input = gr.Audio(label="Or Upload Audio File", type="filepath")
            start_time = gr.Number(label="Start Time (seconds)", value=0, precision=1)
            btn = gr.Button("🎵 Transcribe", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            csv_out = gr.File(label="Download Predictions CSV")
        
    with gr.Row():
        player_out = gr.HTML(label="Drum Roll Player")
    
    # Hidden component for error passthrough
    error_out = gr.Textbox(visible=False)

    btn.click(fn=run_pipeline, 
              inputs=[url_input, file_input, start_time], 
              outputs=[player_out, csv_out, error_out, status])

if __name__ == "__main__":
    demo.launch(share=True)

